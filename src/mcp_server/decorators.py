"""Decorators for MCP Server handlers.

This module provides decorators for common handler patterns like error handling.
"""

import logging
from functools import wraps
from typing import Any, Callable, TypeVar, ParamSpec

from .constants import ResponseStatus, ErrorCode, ErrorMessage

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


def handle_errors(func: Callable[P, T]) -> Callable[P, dict[str, Any]]:
    """Decorator untuk handle errors di async handler functions.
    
    Automatically catches exceptions and returns consistent error response format.
    Removes need for repetitive try-except blocks in handlers.
    
    Args:
        func: Async handler function to wrap
        
    Returns:
        Wrapped function that returns dict with status and error info on exception
    """
    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> dict[str, Any]:
        try:
            result = await func(*args, **kwargs)
            # Ensure result is a dict (for consistency)
            if not isinstance(result, dict):
                logger.warning(
                    f"Handler {func.__name__} returned non-dict result: {type(result)}"
                )
                return {
                    "status": ResponseStatus.SUCCESS.value,
                    "result": result,
                }
            return result
        except ValueError as e:
            logger.error(f"Invalid input in {func.__name__}: {e}", exc_info=False)
            return {
                "status": ResponseStatus.ERROR.value,
                "error": str(e),
                "error_code": ErrorCode.INVALID_INPUT.value,
                "function": func.__name__,
            }
        except RuntimeError as e:
            error_msg = str(e)
            # Check for specific runtime errors
            if "LLM provider" in error_msg or "provider" in error_msg.lower():
                error_code = ErrorCode.LLM_PROVIDER_UNAVAILABLE.value
            elif "resource manager" in error_msg.lower():
                error_code = ErrorCode.RESOURCE_MANAGER_UNAVAILABLE.value
            elif "context" in error_msg.lower():
                error_code = ErrorCode.RUNTIME_ERROR.value
            else:
                error_code = ErrorCode.RUNTIME_ERROR.value
            
            logger.error(f"Runtime error in {func.__name__}: {e}", exc_info=False)
            return {
                "status": ResponseStatus.ERROR.value,
                "error": error_msg,
                "error_code": error_code,
                "function": func.__name__,
            }
        except Exception as e:
            logger.error(
                f"Unexpected error in {func.__name__}: {e}",
                exc_info=True
            )
            return {
                "status": ResponseStatus.ERROR.value,
                "error": f"{ErrorMessage.UNEXPECTED_ERROR}: {e}",
                "error_code": ErrorCode.UNEXPECTED_ERROR.value,
                "function": func.__name__,
            }
    
    return wrapper
