import asyncio
from concurrent.futures import ThreadPoolExecutor
from enum import Enum, auto
from functools import partial
from typing import Any, Awaitable, Callable, Dict, Generic, List, Optional, TypeVar, Union

T = TypeVar('T')  # Input type
R = TypeVar('R')  # Attribution result type


class AttributionState(Enum):
    STOP = auto()
    CONTINUE = auto()


class NVRxAttribution(Generic[T, R]):
    """A class that implements a three-step attribution process.
    This class is designed to be used in a pipeline of attribution modules.
    The output of one attribution module can be used as the input to the next attribution module.

    This class handles:
    1. Input preprocessing - can handle single objects or lists of objects
    2. Attribution computation
    3. Output handling
    """

    # Shared loop for all instances
    _shared_loop = None
    _loop_lock = asyncio.Lock()

    @classmethod
    def get_shared_loop(cls):
        """Get or create the shared event loop."""
        if cls._shared_loop is None or cls._shared_loop.is_closed():
            cls._shared_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(cls._shared_loop)
        return cls._shared_loop

    def __init__(
        self,
        preprocess_input: Callable[[Union[T, List[T]]], Any],
        attribution: Callable[[Any], R],
        output_handler: Callable[[R], None],
        attribution_kwargs: Optional[Dict[str, Any]] = None,
        thread_pool: Optional[ThreadPoolExecutor] = None,
    ):
        """Initialize the attribution module.

        Args:
            preprocess_input: Function to preprocess the input data. Can handle single objects or lists.
            attribution: Function to perform the attribution computation
            output_handler: Function to handle the attribution results
            attribution_kwargs: Optional keyword arguments to pass to the attribution function
            thread_pool: Optional thread pool for running sync functions
        """
        self._preprocess_input = preprocess_input
        self._attribution = attribution
        self._output_handler = output_handler
        self.attribution_kwargs = attribution_kwargs or {}
        self._thread_pool = thread_pool or ThreadPoolExecutor(max_workers=2)

        # Get the shared loop and set the thread pool
        self._loop = self.get_shared_loop()
        self._loop.set_default_executor(self._thread_pool)

    async def _run_sync_in_thread(self, func: Callable, *args, **kwargs) -> Any:
        """Run a synchronous function in a thread pool.

        Args:
            func: The synchronous function to run
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            The result of the function
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._thread_pool, partial(func, *args, **kwargs))

    async def _preprocess_input_inner(
        self, input_data: Union[T, List[T], Awaitable[Union[T, List[T]]]]
    ) -> tuple[Any, AttributionState]:
        """Preprocess the input data.

        Args:
            input_data: The raw input data to be preprocessed. Can be:
                - A single object of type T
                - A list of objects of type T
                - An awaitable that resolves to either of the above

        Returns:
            Preprocessed data ready for attribution, and a flag to indicate if the attribution should continue.
            If the flag is AttributionState.STOP, the attribution should stop and the preprocessed data should be returned.
            If the flag is AttributionState.CONTINUE, the attribution should continue.
        """
        # Handle awaitable inputs (e.g., from other attribution modules)
        # Await on awaitable objects in the list input_data
        awaited_input_data = None
        if isinstance(input_data, Awaitable):
            awaited_input_data = await input_data

        if isinstance(input_data, list):
            awaited_input_data = []
            for item in input_data:
                awaited_item = None
                if isinstance(item, Awaitable):
                    awaited_item = await item
                    if awaited_item[1] == AttributionState.STOP:
                        return awaited_item[0], awaited_item[1]
                    else:
                        awaited_input_data.append(awaited_item[0])
                else:
                    awaited_input_data.append(item)

        else:
            awaited_input_data = input_data
        # Check if preprocess_input is async
        if asyncio.iscoroutinefunction(self._preprocess_input):
            return await self._preprocess_input(awaited_input_data), AttributionState.CONTINUE
        else:
            return (
                await self._run_sync_in_thread(self._preprocess_input, awaited_input_data),
                AttributionState.CONTINUE,
            )

    async def do_attribution(self, preprocessed_data: Any) -> R:
        """Perform the attribution computation.

        Args:
            preprocessed_data: The preprocessed input data

        Returns:
            The attribution results
        """
        # Check if attribution is async
        if asyncio.iscoroutinefunction(self._attribution):
            return await self._attribution(preprocessed_data, **self.attribution_kwargs)
        else:
            return await self._run_sync_in_thread(
                self._attribution, preprocessed_data, **self.attribution_kwargs
            )

    async def output_handler(self, attribution_result: R) -> R:
        """Handle the attribution results.

        Args:
            attribution_result: The results from the attribution computation
        """
        # Check if output_handler is async
        if asyncio.iscoroutinefunction(self._output_handler):
            return await self._output_handler(attribution_result)
        else:
            return await self._run_sync_in_thread(self._output_handler, attribution_result)

    async def run(self, input_data: Union[T, List[T], Awaitable[Union[T, List[T]]]]) -> R:
        """Run the complete attribution pipeline.

        Args:
            input_data: The raw input data to process. Can be:
                - A single object of type T
                - A list of objects of type T
                - An awaitable that resolves to either of the above

        Returns:
            The attribution results of type R
        """
        loop = asyncio.get_running_loop()

        async def _run_pipeline():
            preprocessed_data, flag_to_proceed = await self._preprocess_input_inner(input_data)
            if flag_to_proceed == AttributionState.CONTINUE:
                attribution_result = await self.do_attribution(preprocessed_data)
                final_output = await self.output_handler(attribution_result)
                return final_output
            else:
                return preprocessed_data

        return await loop.create_task(_run_pipeline())

    def run_sync(self, input_data: Union[T, List[T], Awaitable[Union[T, List[T]]]]) -> R:
        """Run the attribution pipeline synchronously.

        Args:
            input_data: The raw input data to process. Can be:
                - A single object of type T
                - A list of objects of type T
                - An awaitable that resolves to either of the above

        Returns:
            The attribution results of type R
        """
        loop = self._loop

        try:
            return loop.run_until_complete(self.run(input_data))
        finally:
            # Don't close the shared loop, just clean up if needed
            pass

    def __del__(self):
        """Cleanup thread pool on deletion."""
        if hasattr(self, '_thread_pool'):
            self._thread_pool.shutdown(wait=False)
