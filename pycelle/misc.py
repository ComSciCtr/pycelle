from __future__ import division

from math import ceil, log

__all__ = [
    'base_expansion',
]

def base_expansion(n, to_base, from_base=10, zfill=None):
    """
    Converts the number `n` from base `from_base` to base `to_base`.

    Parameters
    ----------
    n : list | int
        The number to convert. If an integer, then each digit is treated as
        a value in base `from_base`. That is, 30 is broken up into [3, 0]. Note
        that [3, 10] is a valid input that cannot be passed in as an integer,
        since it require a `from_base` of at least 11.
    to_base : int
        The desired base.
    from_base: int
        The base of `n`.
    zfill : int | None
        If an integer, then the expansion is padded with zeros on the left
        until the length of the expansion is equal to `zfill`.

    Returns
    -------
    out : list
        The `to_base` expansion of `n`.

    Examples
    --------
    >>> base_expansion(31, 2)
    [1, 1, 1, 1, 1]
    >>> base_expansion(31, 3)
    [1, 0, 1, 1]
    >>> base_expansion([1, 1, 1, 1, 1], 3, from_base=2)
    [1, 0, 1, 1]
    >>> base_expansion([1, 1, 1, 1, 1], 3, from_base=10)
    [1, 2, 0, 0, 2, 0, 1, 1, 2]

    """
    # Based on: http://code.activestate.com/recipes/577939-base-expansionconversion-algorithm-python/
    try:
        len(n)
    except TypeError:
        n = list(map(int, str(n)))

    if n == [0]:
        return n

    if max(n) >= from_base:
        raise Exception('Input `n` is not consistent with `from_base`.')

    L = len(n)
    base10 = sum([( from_base ** (L - k - 1) ) * n[k] for k in range(L)])
    j = int(ceil(log(base10 + 1, to_base)))
    out = [( base10 // (to_base ** (j - p)) ) % to_base
           for p in range(1, j + 1)]

    if zfill is not None:
        out = [0] * (zfill - len(out)) + out

    return out

