import time
from contextlib import contextmanager

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

@contextmanager
def time_block(label="Block"):
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        print(f"{label} execution time: {end_time - start_time:.4f} seconds")
