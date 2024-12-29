"""

"""
from typing import Optional
import time


class Timer:
    def __init__(self, text: Optional[str] = None):
        self.text = text

    def __enter__(self):
        self.start = time.perf_counter()
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()
        secs = round(self.end - self.start, 2)

        text = self.text + '. ' if self.text else ''
        print(f'{text}Second to execute: {secs}')
