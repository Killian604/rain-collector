"""

"""
def clear_line(n: int = 1) -> None:
    # TODO: Delete me?
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for _ in range(n):
        print(LINE_UP, end=LINE_CLEAR, flush=True)
