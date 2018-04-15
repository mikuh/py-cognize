"""
the code reference the thinkbayes.py at https://github.com/AllenDowney/ThinkBayes/blob/master/code/thinkbayes.py
"""
__author__ = 'jsyj'
__mail__ = '853934146@qq.com'

import random
import bisect
import math
import copy
import numpy as np


def random_seed(x):
    """Initialize the random and numpy.random generators.
    x: int seed
    """
    random.seed(x)
    np.random.seed(x)


def odds(p):
    """Computes odds for a given probability.
    Example: p=0.75 means 75 for and 25 against, or 3:1 odds in favor.
    Note: when p=1, the formula for odds divides by zero, which is
    normally undefined.  But I think it is reasonable to define Odds(1)
    to be infinity, so that's what this function does.
    p: float 0-1
    Returns: float odds
    """
    if p == 1:
        return float('inf')
    return p / (1 - p)


def probability(yes, no=1):
    """Computes the probability corresponding to given odds.
    Example: yes=2, no=1 means 2:1 odds in favor, or 2/3 probability.

    yes, no: int or float odds in favor
    """
    return float(yes) / (yes + no)


# 线性插值函数
class Interpolator(object):
    """Represents a mapping between sorted sequences; performs linear interp.
    Attributes:
        xs: sorted list
        ys: sorted list
    """

    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def lookup(self, x):
        """Looks up x and returns the corresponding value of y."""
        return self._bisect(x, self.xs, self.ys)

    def reverse(self, y):
        """Looks up y and returns the corresponding value of x."""
        return self._bisect(y, self.ys, self.xs)

    def _bisect(self, x, xs, ys):
        """Helper function."""
        if x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:
            return ys[-1]
        i = bisect.bisect(xs, x)
        frac = 1.0 * (x - xs[i - 1]) / (xs[i] - xs[i - 1])  # 线性函数 x 和 y 线性相关
        y = ys[i - 1] + frac * 1.0 * (ys[i] - ys[i - 1])
        return y


class _DictProbWrapper(dict):
    """An object that used as a dictionary for prob
    """
    def __init__(self, values=None, name=''):
        """Initialize the prior probability.
        """
        self.name = name

        # flag whether the distribution is under a log transform
        self.log = False

        if values is None:
            return

        init_methods = [
            self.init_pmf,
            self.init_mapping,
            self.init_sequence,
            self.init_failure,
        ]

        for method in init_methods:
            try:
                method(values)
                break
            except (AttributeError, TypeError):
                continue

        if len(self) > 0:
            self.normalize()

    def init_pmf(self, values):
        """Initializes with a Pmf.
        values: Pmf object likes:{'a':0.5, 'b':0.5}
        """
        super(_DictProbWrapper, self).__init__(**values)

    def init_sequence(self, values):
        """Initializes with a sequence of equally-likely values.
        values: sequence of values
        """
        for value in values:
            self.set(value, 1)

    def init_mapping(self, values):
        """Initializes with a map from value to probability.
        values: map from value to probability
        """
        super(_DictProbWrapper, self).__init__(**values)

    def init_failure(self, values):
        """Raises an error."""
        raise ValueError('None of the initialization methods worked.')

    def set(self, x, y=0):
        """Sets the freq/prob associated with the value x.
        Args:
            x: name of random variable
            y: number freq or prob
        """
        self[x] = y

    def copy(self, name=None):
        """Returns a copy.
        Make a shallow copy of d.  If you want a deep copy of d,
        use copy.deepcopy on the whole object.
        Args:
            name: string name for the new Hist
        """
        new = copy.copy(self)
        new.name = name if name is not None else self.name
        return new

    def scale(self, factor):
        """Multiplies the values by a factor.
        factor: what to multiply by
        Returns: new object
        """
        new = self.copy()
        new.clear()

        for val, prob in self.items():
            new.set(val * factor, prob)
        return new

    def log(self, m=None):
        """Log transforms the probabilities.

        Removes values with probability 0.
        Normalizes so that the largest logprob is 0.
        """
        if self.log:
            raise ValueError("Pmf/Hist already under a log transform")
        self.log = True

        if m is None:
            m = self.max_like()

        for x, p in self.items():
            if p:
                self.set(x, math.log(p / m))
            else:
                self.remove(x)

    def exp(self, m=None):
        """Exponentiation the probabilities.
        m: how much to shift the ps before exponentiation
        If m is None, normalizes so that the largest prob is 1.
        """
        if not self.log:
            raise ValueError("Pmf/Hist not under a log transform")
        self.log = False

        if m is None:
            m = self.max_like()

        for x, p in self.items():
            self.set(x, math.exp(p - m))

    def render(self):
        """Generates a sequence of points suitable for plotting.
        Returns:
            tuple of (sorted value sequence, freq/prob sequence)
        """
        return zip(*sorted(self.items()))

    def print(self):
        """Prints the values and freqs/probs in ascending order."""
        for val, prob in sorted(self.items()):
            print(val, prob)

    def set_dict(self, d):
        """Sets the dictionary."""
        for value, prob in d.items():
            self.set(value, prob)

    def remove(self, x):
        """Removes a value.
        Throws an exception if the value is not there.
        Args:
            x: value to remove
        """
        del self[x]

    def max_like(self):
        """Returns the largest frequency/probability in the map."""
        return max(self.values())

    def incr(self, x, term=1):
        """Increments the freq/prob associated with the value x.
        Args:
            x: number value
            term: how much to increment by
        """
        self[x] = self.get(x, 0) + term

    def mult(self, x, factor):
        """Scales the freq/prob associated with the value x.
        Args:
            x: number value
            factor: how much to multiply by
        """
        self[x] = self.get(x, 0) * factor

    def total(self):
        """Returns the total of the frequencies/probabilities in the map."""
        return sum(self.values())

    def normalize(self):
        """Normalization the probability
        """
        pass


class Hist(_DictProbWrapper):
    """Represents a histogram, which is a map from values to frequencies.
    Values can be any hashable type; frequencies are integer counters.
    """

    def freq(self, x):
        """Gets the frequency associated with the value x.
        Args:
            x: number value
        Returns:
            int frequency
        """
        return self.get(x, 0)

    def freqs(self, xs):
        """Gets frequencies for a sequence of values."""
        return [self.freq(x) for x in xs]

    def is_subset(self, other):
        """Checks whether the values in this histogram are a subset of
        the values in the given histogram."""
        for val, freq in self.items():
            if freq > other.freq(val):
                return False
        return True

    def subtract(self, other):
        """Subtracts the values in the given histogram from this histogram."""
        for val, freq in other.items():
            self.incr(val, -freq)


if __name__ == '__main__':
    # test code
    dp = _DictProbWrapper(values='abcde')
    dp2 = _DictProbWrapper(values=dp)
    dp3 = _DictProbWrapper(values={'a': 0.5, 'b': 0.5})
    dp4 = dp.copy()
    print(dp)
    print(dp2)
    print(dp3)
    print(dp4 is dp)
    print(len(dp), len(dp3))