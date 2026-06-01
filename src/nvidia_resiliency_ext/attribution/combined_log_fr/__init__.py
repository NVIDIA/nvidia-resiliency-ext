from importlib import import_module

_EXPORTS = {
    "CombinedLogFR": ".combined_log_fr",
    "merge_log_fr_llm": ".llm_merge",
    "unpack_run_result": ".llm_merge",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
