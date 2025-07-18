"""

"""
import os
from typing import List


def clear_line(n: int = 1) -> None:
    # TODO: Delete me?
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for _ in range(n):
        print(LINE_UP, end=LINE_CLEAR, flush=True)


def recursively_search_files(d) -> List[str]:
    """
    :param d: a directory path
    :return:
    """
    result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(d) for f in filenames]
    return result
