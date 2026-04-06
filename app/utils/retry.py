import logging
import time
from typing import Callable, TypeVar

T = TypeVar('T')

logger = logging.getLogger("J.A.R.V.I.S")

def with_retry(
    max_retries: int = 3,
    initial_delay: float = 0.5,
) -> Callable:
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args, **kwargs) -> T:
            last_exc = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    
                    if attempt < max_retries - 1:
                        logger.debug("[RETRY] Attempt %d/%d failed: %s - retrying in %.1fs", attempt + 1, max_retries, e, initial_delay)
                        time.sleep(initial_delay)
                    else:
                        logger.debug("[RETRY] Attempt %d/%d failed: %s - giving up", attempt + 1, max_retries, e)
                        
            raise last_exc
        return wrapper
    return decorator