import argparse
import asyncio
import inspect
import logging
from concurrent.futures import ThreadPoolExecutor
from enum import Enum, auto
from functools import partial
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

T = TypeVar('T')  # Input type
R = TypeVar('R')  # Attribution result type

logger = logging.getLogger(__name__)


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
        stop_on_stop: bool = False,
    ):
        """Initialize the attribution module.

        Args:
            preprocess_input: Function to preprocess the input data. Can handle single objects or lists.
            attribution: Function to perform the attribution computation
            output_handler: Function to handle the attribution results
            attribution_kwargs: Optional keyword arguments to pass to the attribution function
            thread_pool: Optional thread pool for running sync functions
            stop_on_stop: Whether to stop the attribution pipeline if the output handler returns STOP state
        """
        self.register_attr_pipeline(
            {
                'preprocess_input': preprocess_input,
                'attribution': attribution,
                'output_handler': output_handler,
            }
        )
        self.attribution_kwargs = attribution_kwargs or {}
        self._thread_pool = thread_pool or ThreadPoolExecutor(max_workers=4)
        self._stop_on_stop = stop_on_stop
        self.args = None
        # Get the shared loop and set the thread pool
        self._loop = self.get_shared_loop()
        self._loop.set_default_executor(self._thread_pool)

    def inspect_type_consistency(self, attr_pipeline: Dict[str, Callable]):
        """Inspect the type consistency of the attribution pipeline."""
        # Check if the output handler returns a tuple of (result, AttributionState)
        prev_step = attr_pipeline['preprocess_input']
        for step in [attr_pipeline['attribution'], attr_pipeline['output_handler']]:
            if not isinstance(step, Callable):
                return False
            prev_step_sig = inspect.signature(prev_step)
            prev_step_params = list(prev_step_sig.parameters.values())
            prev_step_output_type = prev_step_sig.return_annotation

            step_sig = inspect.signature(step)
            step_params = list(step_sig.parameters.values())
            step_input = step_params[0].annotation if step_params else inspect.Parameter.empty

            if prev_step_output_type != step_input:
                raise ValueError(
                    f"The attribution pipeline is not type consistent. prev_step_output_type: {prev_step_output_type}, step_input: {step_input}"
                )
            prev_step = step

    def register_attr_pipeline(self, attr_pipeline: Dict[str, Callable]):
        """Register the attribution routines."""
        try:
            self.inspect_type_consistency(attr_pipeline)
        except ValueError as e:
            logger.error(f"The attribution pipeline is not type consistent. {e}")
        finally:
            self._preprocess_input = attr_pipeline['preprocess_input']
            self._attribution = attr_pipeline['attribution']
            self._output_handler = attr_pipeline['output_handler']

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

    async def _preprocess_input_inner(self) -> Any:
        """Preprocess the input data.

        Args:
        """
        if asyncio.iscoroutinefunction(self._preprocess_input):
            return await self._preprocess_input()
        else:
            return await self._run_sync_in_thread(self._preprocess_input)

    async def do_attribution(self, preprocessed_data: Any) -> R:
        """Perform the attribution computation.

        Args:
            preprocessed_data: The preprocessed input data

        Returns:
            The attribution results
        """
        # Check if attribution is async
        if asyncio.iscoroutinefunction(self._attribution):
            return await self._attribution(preprocessed_data)
        else:
            return await self._run_sync_in_thread(self._attribution, preprocessed_data)

    async def output_handler(self, attribution_result: R) -> tuple[R, AttributionState]:
        """Handle the attribution results.

        Args:
            attribution_result: The results from the attribution computation
        """
        # Check if output_handler is async
        if asyncio.iscoroutinefunction(self._output_handler):
            return await self._output_handler(attribution_result)
        else:
            return await self._run_sync_in_thread(self._output_handler, attribution_result)

    async def run(
        self, args: Union[argparse.Namespace, Dict[str, Any]]
    ) -> tuple[R, AttributionState]:
        """Run the complete attribution pipeline.

        Args:
            args: The keyword arguments to pass to the attribution pipeline

        Returns:
            The attribution results of type R
        """
        loop = asyncio.get_running_loop()
        # Set self.args from the provided args
        try:
            # Always replace args because the caller is responsible for setting the correct arguments.
            self.args = args if isinstance(args, argparse.Namespace) else argparse.Namespace(**args)
        except Exception as e:
            logger.error(f"Error setting self.args: {e}")
            raise ValueError(f"Invalid type of args: {type(args)}")

        async def _run_pipeline():
            preprocessed_data = await self._preprocess_input_inner()
            attribution_result = await self.do_attribution(preprocessed_data)
            return await self.output_handler(attribution_result)

        return await loop.create_task(_run_pipeline())

    def run_sync(self, args: Union[argparse.Namespace, Dict[str, Any]]) -> R:
        """Run the attribution pipeline synchronously.

        Args:
            kwargs: The keyword arguments to pass to the attribution pipeline

        Returns:
            The attribution results of type R
        """
        loop = self._loop

        try:
            return loop.run_until_complete(self.run(args))
        finally:
            # Don't close the shared loop, just clean up if needed
            pass

    def __del__(self):
        """Cleanup thread pool on deletion."""
        if hasattr(self, '_thread_pool'):
            self._thread_pool.shutdown(wait=False)
