import time
from functools import wraps

from src.utils.logger import get_logger

logger = get_logger(__name__)


def timeit(func):
    """Decorator to log the execution time of pipeline steps."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{func.__name__} finished in {elapsed:.2f}s")
        return result
    return wrapper


def risk_band(prob: float, bands: dict) -> str:
    """Map a predicted default probability to a business-readable risk band."""
    for label, (low, high) in bands.items():
        if low <= prob < high:
            return label
    return "Unknown"


def safe_div(a, b):
    return a / b if b not in (0, None) else 0.0
