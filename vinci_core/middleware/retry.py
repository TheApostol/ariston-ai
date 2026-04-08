import asyncio
from functools import wraps

def with_retry(max_retries: int = 3, base_delay: float = 1.0):
    """
    Async middleware decorator to retry failed model inference calls due to rate limiting or timeouts.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    err_message = str(e).lower()
                    
                    # Target transient HTTP exceptions (Rate Limit, Timeout)
                    if "429" in err_message or "timeout" in err_message or "502" in err_message or "503" in err_message:
                        if retries == max_retries:
                            print(f"[Middleware] Max retries ({max_retries}) exhausted: {e}")
                            raise e
                            
                        delay = base_delay * (2 ** (retries - 1))  # Exponential backoff
                        print(f"[Middleware] Caught transient error: {e}. Retrying in {delay}s... (Attempt {retries}/{max_retries})")
                        await asyncio.sleep(delay)
                    else:
                        # Non-transient errors like Auth failures should crash immediately
                        raise e
            return await func(*args, **kwargs)
        return wrapper
    return decorator
