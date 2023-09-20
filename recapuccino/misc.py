def in_ipython():
    try:
        return __IPYTHON__
    except NameError:
        return False
