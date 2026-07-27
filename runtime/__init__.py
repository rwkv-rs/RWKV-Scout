"""Project-owned model runtimes.

The application talks to the small backend contract exposed here.  HTTP
compatibility is an adapter, not the model abstraction; the direct RWKV
backend can load a checkpoint and run inference without a model API server.
"""

from runtime.backend import BackendResponse, ModelBackend
from runtime.factory import get_model_backend, reset_model_backend

__all__ = [
    "BackendResponse",
    "ModelBackend",
    "get_model_backend",
    "reset_model_backend",
]
