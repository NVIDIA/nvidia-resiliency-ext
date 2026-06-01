import argparse
import asyncio
import inspect
import logging
import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from enum import Enum, auto
from functools import partial
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

T = TypeVar('T')  # Input type
R = TypeVar('R')  # Attribution result type

logger = logging.getLogger(__name__)


def normalize_attribution_args(args: Any) -> Dict[str, Any]:
    """Normalize CLI or service arguments to a plain dict (library carrier).

    Accepts, in order:

    - :class:`argparse.Namespace` (CLIs)
    - Any :class:`collections.abc.Mapping` (preferred for services)
    - A length-2 :class:`list` or :class:`tuple` (``[log_result, fr_result]`` →
      ``{"input_data": [log_result, fr_result]}`` for :class:`~nvidia_resiliency_ext.attribution.combined_log_fr.combined_log_fr.CombinedLogFR`)
    - Other objects with a writable ``__dict__`` (e.g. simple test doubles; discouraged for new code)
    """
    if isinstance(args, argparse.Namespace):
        return dict(vars(args))
    if isinstance(args, Mapping):
        return dict(args)
    if isinstance(args, (list, tuple)) and len(args) == 2:
        return {"input_data": [args[0], args[1]]}
    if getattr(args, "__dict__", None) is not None:
        return dict(vars(args))
    raise TypeError(
        "run() / run_sync() args must be Namespace, mapping, length-2 list/tuple, "
        "or simple object with __dict__, "
        f"not {type(args).__name__}"
    )


_attribution_run_ctx: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "nvidia_resiliency_ext_attribution_run_args",
    default=None,
)


def peek_attribution_run_args() -> Optional[Dict[str, Any]]:
    """Return the dict for the active :meth:`NVRxAttribution.run` call, or ``None`` if not in a run."""
    return _attribution_run_ctx.get()


def current_attribution_run_args() -> Dict[str, Any]:
    """Return the normalized dict for the active run; raises if :meth:`run` / :meth:`run_sync` is not active."""
    v = peek_attribution_run_args()
    if v is None:
        raise RuntimeError(
            "Attribution run arguments are only available while NVRxAttribution.run() is active."
        )
    return v


def effective_run_or_init_config(init_config: Mapping[str, Any]) -> Dict[str, Any]:
    """Use the active run dict when inside :meth:`run`; otherwise constructor config (for direct step calls)."""
    run = peek_attribution_run_args()
    if run is not None:
        return run
    return dict(init_config)


def merged_attribution_config(init_config: Mapping[str, Any]) -> Dict[str, Any]:
    """``{**init, **run}`` while a run is active; run keys override. If no run is active, return init only."""
    base = dict(init_config)
    run = peek_attribution_run_args()
    if run is None:
        return base
    return {**base, **run}


def _callable_arity(fn: Callable[..., Any]) -> int:
    """Parameter count for plain functions; ``self`` not counted for bound methods."""
    return len(inspect.signature(fn).parameters)


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

    # One event loop per thread (not process-global). A class-level loop breaks when
    # ``CollectiveAnalyzer`` is constructed inside ``run_in_executor``: two loops in one
    # thread (ephemeral ``new_event_loop`` vs ``get_shared_loop``) fight over
    # ``set_event_loop`` and corrupt shared state.
    _loop_local = threading.local()

    @classmethod
    def get_shared_loop(cls):
        """Get or create this thread's event loop and bind it with ``set_event_loop``."""
        loop = getattr(cls._loop_local, "loop", None)
        if loop is None or loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            cls._loop_local.loop = loop
        return loop

    @classmethod
    def reset_thread_event_loop(cls) -> None:
        """Close and drop the loop for the **current** thread (one-shot executor workers)."""
        if not hasattr(cls._loop_local, "loop"):
            return
        loop = cls._loop_local.loop
        try:
            if not loop.is_closed():
                loop.close()
        finally:
            delattr(cls._loop_local, "loop")
            asyncio.set_event_loop(None)

    def __init__(
        self,
        preprocess_input: Callable[[Union[T, List[T]]], Any],
        attribution: Callable[[Any], R],
        output_handler: Callable[[R], None],
        thread_pool: Optional[ThreadPoolExecutor] = None,
    ):
        """Initialize the attribution module.

        Args:
            preprocess_input: Function to preprocess the input data. Can handle single objects or lists.
            attribution: Function to perform the attribution computation
            output_handler: Function to handle the attribution results
            thread_pool: Optional thread pool for running sync functions
        """
        self.register_attr_pipeline(
            {
                'preprocess_input': preprocess_input,
                'attribution': attribution,
                'output_handler': output_handler,
            }
        )
        self._thread_pool = thread_pool or ThreadPoolExecutor(max_workers=4)
        # Keep a loop handle for synchronous callers; sync steps use self._thread_pool explicitly.
        self._loop = self.get_shared_loop()

    def inspect_type_consistency(self, attr_pipeline: Dict[str, Callable]) -> None:
        """Inspect the type consistency of the attribution pipeline.

        Steps are assumed callable (constructor / :meth:`register_attr_pipeline`); invalid objects
        raise :exc:`TypeError` from :func:`inspect.signature`.

        Raises:
            ValueError: If annotations do not chain consistently.
            TypeError: If a step is not a valid callable for :func:`inspect.signature`.
        """
        prev_step = attr_pipeline['preprocess_input']
        for step in (attr_pipeline['attribution'], attr_pipeline['output_handler']):
            prev_step_sig = inspect.signature(prev_step)
            prev_step_output_type = prev_step_sig.return_annotation

            step_sig = inspect.signature(step)
            step_params = list(step_sig.parameters.values())
            step_input = step_params[0].annotation if step_params else inspect.Parameter.empty

            if prev_step_output_type != step_input:
                raise ValueError(
                    "The attribution pipeline is not type consistent. "
                    f"prev_step_output_type: {prev_step_output_type}, step_input: {step_input}"
                )
            prev_step = step

    def register_attr_pipeline(self, attr_pipeline: Dict[str, Callable]) -> None:
        """Register the attribution routines after type checks; raises if invalid."""
        try:
            self.inspect_type_consistency(attr_pipeline)
        except ValueError as e:
            logger.error("The attribution pipeline is not type consistent. %s", e)
            raise
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

    async def _preprocess_input_inner(self, run_args: Dict[str, Any]) -> Any:
        """Run ``preprocess_input`` with zero args, or with ``run_args`` if it takes one parameter."""
        fn = self._preprocess_input
        n = _callable_arity(fn)
        if n > 1:
            raise TypeError(f"preprocess_input must accept 0 or 1 arguments, not {n}")
        if inspect.iscoroutinefunction(fn):
            if n == 0:
                return await fn()
            return await fn(run_args)
        if n == 0:
            return await self._run_sync_in_thread(fn)
        return await self._run_sync_in_thread(fn, run_args)

    async def do_attribution(self, preprocessed_data: Any) -> R:
        """Perform the attribution computation.

        Args:
            preprocessed_data: The preprocessed input data

        Returns:
            The attribution results
        """
        # Check if attribution is async
        if inspect.iscoroutinefunction(self._attribution):
            return await self._attribution(preprocessed_data)
        else:
            return await self._run_sync_in_thread(self._attribution, preprocessed_data)

    async def output_handler(self, attribution_result: R) -> tuple[R, AttributionState]:
        """Handle the attribution results.

        Args:
            attribution_result: The results from the attribution computation
        """
        # Check if output_handler is async
        if inspect.iscoroutinefunction(self._output_handler):
            return await self._output_handler(attribution_result)
        else:
            return await self._run_sync_in_thread(self._output_handler, attribution_result)

    async def run(self, args: Any) -> tuple[R, AttributionState]:
        """Run the complete attribution pipeline.

        Args:
            args: Run parameters as a :class:`dict` (preferred) or :class:`argparse.Namespace`
                (CLI). Exposed to pipeline steps via :func:`peek_attribution_run_args` /
                :func:`effective_run_or_init_config` for the duration of this call.

        Returns:
            The attribution results of type R
        """
        try:
            run_args = normalize_attribution_args(args)
        except TypeError:
            raise
        except Exception as e:
            logger.error("Error normalizing run args: %s", e)
            raise ValueError(f"Invalid run args: {type(args)}") from e

        token = _attribution_run_ctx.set(run_args)
        try:
            preprocessed_data = await self._preprocess_input_inner(run_args)
            attribution_result = await self.do_attribution(preprocessed_data)
            return await self.output_handler(attribution_result)
        finally:
            _attribution_run_ctx.reset(token)

    def run_sync(self, args: Any) -> R:
        """Run the attribution pipeline synchronously.

        Args:
            args: Same as :meth:`run` — mapping or :class:`argparse.Namespace`.

        Returns:
            The attribution results of type R
        """
        loop = self.get_shared_loop()
        if loop is not self._loop:
            self._loop = loop

        try:
            return loop.run_until_complete(self.run(args))
        finally:
            # Don't close the shared loop, just clean up if needed
            pass

    def __del__(self):
        """Cleanup thread pool on deletion."""
        if hasattr(self, '_thread_pool'):
            self._thread_pool.shutdown(wait=False)
