import profilehooks as ph

def profile(fnc):
    """A decorator that uses cProfile to profile a function"""

    def outer(*args, **kwargs):
        """The outer function"""

        # profilehooks decorator
        @ph.profile(immediate=True,entries=1000)
        def inner(*args, **kwargs):
            """The inner function"""
            # run the function
            fnc(*args, **kwargs)
            return True

        print("Profiling: {}".format(fnc.__name__))
        
        # returns the time of execution
        return print(inner(*args, **kwargs))

    return outer