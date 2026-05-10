"""coach — interactive UI that walks users through the mindxtrain pipeline.

Mounted under /coach by mindxtrain.operator.app. Static assets at /coach/static/*.
"""

from mindxtrain.operator.coach.api import router

__all__ = ["router"]
