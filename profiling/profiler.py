import cProfile
import pstats


def profile(fnc, n):
    """A decorator that uses cProfile to profile a function"""

    pr = cProfile.Profile()
    pr.enable()

    fnc(n)
    pr.disable()
    ps = pstats.Stats(pr).sort_stats("cumulative")
    ps.print_stats()
    return
