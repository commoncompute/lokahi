"""Miki Engine public facade; legacy ``ollm`` remains source compatible.

Importing this facade does not load MLX, Transformers, models or native code.
The existing implementation is reused, not forked or renamed underneath callers.
"""

__all__ = ["Inference", "TensorTracer", "TextStreamer", "TraceEvent", "file_get_contents"]


def __getattr__(name):
    if name in __all__:
        import ollm

        return getattr(ollm, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
