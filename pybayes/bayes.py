"""
the code reference the thinkbayes.py at https://github.com/AllenDowney/ThinkBayes/blob/master/code/thinkbayes.py
"""
__author__ = 'jsyj'
__email__ = '853934146@qq.com'

import random
import bisect
import scipy.stats
import math
import copy
import numpy as np
import logging

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
            x:  number value represent random variable
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
            x: random variable
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

    def __hash__(self):
        return hash(self.name)


class Hist(_DictProbWrapper):
    """Represents a histogram, which is a map from values to frequencies.
    Values can be any hashable type; frequencies are integer counters.
    """

    def freq(self, x):
        """Gets the frequency associated with the value x.
        Args:
            x: random variable
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


class Pmf(_DictProbWrapper):
    """Represents a probability mass function.

    Values can be any hashable type; probabilities are floating-point.
    Pmfs are not necessarily normalized.
    """

    def prob(self, x, default=0):
        """Gets the probability associated with the value x.
        Args:
            x: random variable
            default: value to return if the key is not there
        Returns:
            float probability
        """
        return self.get(x, default)

    def probs(self, xs):
        """Gets probabilities for a sequence of values."""
        return [self.prob(x) for x in xs]

    def make_cdf(self, name=None):
        """Makes a Cdf."""
        return MakeDistribution('cdf').from_pmf(self, name=name)

    def prob_greater(self, x):
        """Probability that a sample from this Pmf exceeds x.
        x: number
        returns: float probability
        """
        t = [prob for (val, prob) in self.items() if val > x]
        return sum(t)

    def prob_less(self, x):
        """Probability that a sample from this Pmf is less than x.
        x: number
        returns: float probability
        """
        t = [prob for (val, prob) in self.items() if val < x]
        return sum(t)

    def __lt__(self, obj):
        """Less than.
        obj: number or _DictWrapper
        returns: float probability
        """
        if isinstance(obj, _DictProbWrapper):
            return pmf_prob_less(self, obj)
        else:
            return self.prob_less(obj)

    def __gt__(self, obj):
        """Greater than.
        obj: number or _DictProbWrapper
        returns: float probability
        """
        if isinstance(obj, _DictProbWrapper):
            return pmf_prob_greater(self, obj)
        else:
            return self.prob_greater(obj)

    def __ge__(self, obj):
        """Greater than or equal.
        obj: number or _DictWrapper
        returns: float probability
        """
        return 1 - (self < obj)

    def __le__(self, obj):
        """Less than or equal.
        obj: number or _DictWrapper
        returns: float probability
        """
        return 1 - (self > obj)

    def __eq__(self, obj):
        """Equal to.
        obj: number or _DictWrapper
        returns: float probability
        """
        if isinstance(obj, _DictProbWrapper):
            return pmf_prob_equal(self, obj)
        else:
            return self.prob(obj)

    def __ne__(self, obj):
        """Not equal to.
        obj: number or _DictWrapper
        returns: float probability
        """
        return 1 - (self == obj)

    def normalize(self, fraction=1.0):
        """Normalizes this PMF so the sum of all probs is fraction.
        Args:
            fraction: what the total should be after normalization
        Returns: the total probability before normalizing
        """
        if self.log:
            raise ValueError("Pmf is under a log transform")

        total = self.total()
        if total == 0.0:
            raise ValueError('total probability is zero.')
            logging.warning('Normalize: total probability is zero.')
            return total

        factor = float(fraction) / total
        for x in self:
            self[x] *= factor

        return total

    def random(self):
        """Chooses a random element from this PMF.
        Returns:
            float value from the Pmf
        """
        if len(self) == 0:
            raise ValueError('Pmf contains no values.')

        target = random.random()
        total = 0.0
        for x, p in self.items():
            total += p
            if total >= target:
                return x

        # we shouldn't get here
        assert False

    def mean(self):
        """Computes the mean of a PMF.
        Returns:
            float mean
        """
        mu = 0.0
        for x, p in self.items():
            mu += p * x
        return mu

    def var(self, mu=None):
        """Computes the variance of a PMF.
        Args:
            mu: the point around which the variance is computed;
                if omitted, computes the mean
        Returns:
            float variance
        """
        if mu is None:
            mu = self.mean()

        var = 0.0
        for x, p in self.items():
            var += p * (x - mu) ** 2
        return var

    def maximum_likelihood(self):
        """Returns the value with the highest probability.
        Returns: float probability
        """
        prob, val = max((prob, val) for val, prob in self.items())
        return val

    def credible_interval(self, percentage=90):
        """Computes the central credible interval.
        If percentage=90, computes the 90% CI.
        Args:
            percentage: float between 0 and 100
        Returns:
            sequence of two floats, low and high
        """
        cdf = self.make_cdf()
        return cdf.credible_interval(percentage)

    def __add__(self, other):
        """Computes the Pmf of the sum of values drawn from self and other.
        other: another Pmf
        returns: new Pmf
        """
        try:
            return self.add_pmf(other)
        except AttributeError:
            # if other is a number
            return self.add_constant(other)

    def add_pmf(self, other):
        """Computes the Pmf of the sum of values drawn from self and other.
        other: another Pmf
        returns: new Pmf
        """
        pmf = Pmf()
        for v1, p1 in self.items():
            for v2, p2 in other.items():
                pmf.incr(v1 + v2, p1 * p2)
        return pmf

    def add_constant(self, other):
        """Computes the Pmf of the sum a constant and  values from self.
        other: a number
        returns: new Pmf
        """
        pmf = Pmf()
        for v1, p1 in self.items():
            pmf.set(v1 + other, p1)
        return pmf

    def __sub__(self, other):
        """Computes the Pmf of the diff of values drawn from self and other.
        other: another Pmf
        returns: new Pmf
        """
        pmf = Pmf()
        for v1, p1 in self.items():
            for v2, p2 in other.Items():
                pmf.incr(v1 - v2, p1 * p2)
        return pmf

    def max(self, k):
        """Computes the CDF of the maximum of k selections from this dist.
        k: int
        returns: new Cdf
        """
        cdf = self.make_cdf()
        cdf.ps = [p ** k for p in cdf.ps]
        return cdf


class Cdf(object):
    """Represents a cumulative distribution function.
    Attributes:
        xs: sequence of values
        ps: sequence of probabilities
        name: string used as a graph label.
    """

    def __init__(self, xs=None, ps=None, name=''):
        self.xs = [] if xs is None else xs
        self.ps = [] if ps is None else ps
        self.name = name

    def copy(self, name=None):
        """Returns a copy of this Cdf.
        Args:
            name: string name for the new Cdf
        """
        if name is None:
            name = self.name
        return Cdf(list(self.xs), list(self.ps), name)

    def make_pmf(self, name=None):
        """Makes a Pmf."""
        return MakeDistribution('pmf').from_cdf(self, name=name)

    def values(self):
        """Returns a sorted list of values.
        """
        return self.xs

    def items(self):
        """Returns a sorted sequence of (value, probability) pairs.
        Note: in Python3, returns an iterator.
        """
        return zip(self.xs, self.ps)

    def append(self, x, p):
        """Add an (x, p) pair to the end of this CDF.
        Note: this us normally used to build a CDF from scratch, not
        to modify existing CDFs.  It is up to the caller to make sure
        that the result is a legal CDF.
        """
        self.xs.append(x)
        self.ps.append(p)

    def shift(self, term):
        """Adds a term to the xs.
        term: how much to add
        """
        new = self.copy()
        new.xs = [x + term for x in self.xs]
        return new

    def scale(self, factor):
        """Multiplies the xs by a factor.
        factor: what to multiply by
        """
        new = self.copy()
        new.xs = [x * factor for x in self.xs]
        return new

    def prob(self, x):
        """Returns CDF(x), the probability that corresponds to value x.
        Args:
            x: number
        Returns:
            float probability
        """
        if x < self.xs[0]:
            return 0.0
        index = bisect.bisect(self.xs, x)
        p = self.ps[index - 1]
        return p

    def value(self, p):
        """Returns InverseCDF(p), the value that corresponds to probability p.
        Args:
            p: number in the range [0, 1]
        Returns:
            number value
        """
        if p < 0 or p > 1:
            raise ValueError('Probability p must be in range [0, 1]')

        if p == 0:
            return self.xs[0]
        if p == 1:
            return self.xs[-1]
        index = bisect.bisect(self.ps, p)
        if p == self.ps[index - 1]:
            return self.xs[index - 1]
        else:
            return self.xs[index]

    def percentile(self, p):
        """Returns the value that corresponds to percentile p.
        Args:
            p: number in the range [0, 100]
        Returns:
            number value
        """
        return self.value(p / 100.0)

    def random(self):
        """Chooses a random value from this distribution."""
        return self.value(random.random())

    def sample(self, n):
        """Generates a random sample from this distribution.

        Args:
            n: int length of the sample
        """
        return [self.random() for _ in range(n)]

    def mean(self):
        """Computes the mean of a CDF.
        Returns:
            float mean
        """
        old_p = 0
        total = 0.0
        for x, new_p in zip(self.xs, self.ps):
            p = new_p - old_p
            total += p * x
            old_p = new_p
        return total

    def credible_interval(self, percentage=90):
        """Computes the central credible interval.
        If percentage=90, computes the 90% CI.
        Args:
            percentage: float between 0 and 100
        Returns:
            sequence of two floats, low and high
        """
        prob = (1 - percentage / 100.0) / 2
        interval = self.value(prob), self.value(1 - prob)
        return interval

    def _round(self, multiplier=1000.0):
        """
        An entry is added to the cdf only if the percentile differs
        from the previous value in a significant digit, where the number
        of significant digits is determined by multiplier.  The
        default is 1000, which keeps log10(1000) = 3 significant digits.
        """
        # TODO(write this method)
        raise UnimplementedMethodException()

    def render(self):
        """Generates a sequence of points suitable for plotting.
        An empirical CDF is a step function; linear interpolation can be misleading.
        Returns:
            tuple of (xs, ps)
        """
        xs = [self.xs[0]]
        ps = [0.0]
        for i, p in enumerate(self.ps):
            xs.append(self.xs[i])
            ps.append(p)

            try:
                xs.append(self.xs[i + 1])
                ps.append(p)
            except IndexError:
                pass
        return xs, ps

    def max(self, k):
        """Computes the CDF of the maximum of k selections from this dist.
        k: int
        returns: new Cdf
        """
        cdf = self.copy()
        cdf.ps = [p ** k for p in cdf.ps]
        return cdf




class MakeDistribution(object):
    """Make a Prob Distribution
    """
    def __call__(self, distribution):
        if distribution == 'cdf':
            return MakeCdf()
        elif distribution == 'pmf':
            return MakePmf()
        elif distribution == 'hist':
            return MakeHist()
        elif distribution == 'suite':
            return MakeSuite()


class MakePmf(object):
    """Make a Pmf
    """
    def from_list(self, l, name=''):
        """Makes a PMF from an unsorted sequence of values.
           Args:
               t: sequence of numbers
               name: string name for this PMF
           Returns:
               Pmf object
           """
        hist = MakeDistribution('hist').from_list(l)
        pmf = Pmf(hist, name)
        pmf.normalize()
        return pmf

    def from_dict(self, d, name=''):
        """Makes a PMF from a map from values to probabilities.
           Args:
               d: dictionary that maps values to probabilities
               name: string name for this PMF
           Returns:
               Pmf object
           """
        pmf = Pmf(d, name)
        pmf.normalize()
        return pmf

    def from_items(self, name='', **kwargs):
        """Makes a PMF from a sequence of value-probability pairs
            Args:
                name: string name for this PMF
                kwargs: sequence of value-probability pairs
            Returns:
                Pmf object
            """
        pmf = Pmf(dict(kwargs), name)
        pmf.normalize()
        return pmf

    def from_hist(self, hist, name=None):
        """Makes a normalized PMF from a Hist object.
            Args:
                hist: Hist object
                name: string name
            Returns:
                Pmf object
            """
        if name is None:
            name = hist.name

        pmf = Pmf(hist, name)
        pmf.normalize()
        return pmf

    def from_cdf(self, cdf, name=None):
        """Makes a normalized Pmf from a Cdf object.
            Args:
                cdf: Cdf object
                name: string name for the new Pmf
            Returns:
                Pmf object
            """
        if name is None:
            name = cdf.name

        pmf = Pmf(name=name)

        prev = 0.0
        for val, prob in cdf.items():
            pmf.incr(val, prob - prev)
            prev = prob
        return pmf


class MakeHist(object):
    """Make a Hist
    """
    def from_list(self, l, name=''):
        """Makes a histogram from an unsorted sequence of values.
           Args:
               l: sequence of numbers
               name: string name for this histogram
           Returns:
               Hist object
           """
        hist = Hist(name=name)
        [hist.incr(x) for x in l]
        return hist

    def from_dict(self, d, name=''):
        """Makes a histogram from a map from values to frequencies.
            Args:
                d: dictionary that maps values to frequencies
                name: string name for this histogram
            Returns:
                Hist object
            """
        return Hist(d, name)

    def from_pmf(self, pmf, name=None):
        if name is None:
            name = pmf.name
        return Hist(pmf, name)





class MakeCdf(object):
    """Make a Cdf.
    """
    def from_items(self, items, name=''):
        """Makes a cdf from an unsorted sequence of (value, frequency) pairs.
        Args:
            items: unsorted sequence of (value, frequency) pairs
            name: string name for this CDF
        Returns:
            cdf: list of (value, fraction) pairs
        """
        runsum = 0
        xs = []
        cs = []

        for value, count in sorted(items):
            runsum += count
            xs.append(value)
            cs.append(runsum)

        total = float(runsum)
        ps = [c / total for c in cs]

        cdf = Cdf(xs, ps, name)
        return cdf

    def from_dict(self, d, name=''):
        """Makes a CDF from a dictionary that maps values to frequencies.
           Args:
              d: dictionary that maps values to frequencies.
              name: string name for the data.
           Returns:
               Cdf object
           """
        return self.from_items(d.items(), name)

    def from_hist(self, hist, name=''):
        """Makes a CDF from a Hist object.
            Args:
               hist: Pmf.Hist object
               name: string name for the data.
            Returns:
                Cdf object
            """
        return self.from_items(hist.items(), name)

    def from_pmf(self, pmf, name=None):
        """Makes a CDF from a Pmf object.
           Args:
              pmf: Pmf.Pmf object
              name: string name for the data.
           Returns:
               Cdf object
           """
        if name is None:
            name = pmf.name
        return self.from_items(pmf.items(), name)

    def from_list(self, l, name=''):
        """Creates a CDF from an unsorted sequence.
            Args:
                l: unsorted sequence of sortable values
                name: string name for the cdf
            Returns:
               Cdf object
        """
        hist = MakeDistribution('hist').from_list(l)
        return self.from_hist(hist, name)

class MakeJoint(object):
    """Joint distribution of values from pmf1 and pmf2.
        Args:
            pmf1: Pmf object
            pmf2: Pmf object
        Returns:
            Joint pmf of value pairs
    """
    def __call__(self, pmf1, pmf2):
        joint = Joint()
        for v1, p1 in pmf1.Items():
            for v2, p2 in pmf2.Items():
                joint.set((v1, v2), p1 * p2)
        return joint



class Joint(Pmf):
    """Represents a joint distribution.
        The values are sequences (usually tuples)
        """

    def marginal(self, i, name=''):
        """Gets the marginal distribution of the indicated variable.
        i: index of the variable we want
        Returns: Pmf
        """
        pmf = Pmf(name=name)
        for vs, prob in self.items():
            pmf.incr(vs[i], prob)
        return pmf

    def conditional(self, i, j, val, name=''):
        """Gets the conditional distribution of the indicated variable.
        Distribution of vs[i], conditioned on vs[j] = val.
        i: index of the variable we want
        j: which variable is conditioned on
        val: the value the jth variable has to have
        Returns: Pmf
        """
        pmf = Pmf(name=name)
        for vs, prob in self.items():
            if vs[j] != val:
                continue
            pmf.incr(vs[i], prob)

        pmf.normalize()
        return pmf

    def max_like_interval(self, percentage=90):
        """Returns the maximum-likelihood credible interval.
        If percentage=90, computes a 90% CI containing the values
        with the highest likelihoods.
        percentage: float between 0 and 100
        Returns: list of values from the suite
        """
        interval = []
        total = 0

        t = [(prob, val) for val, prob in self.items()]
        t.sort(reverse=True)

        for prob, val in t:
            interval.append(val)
            total += prob
            if total >= percentage / 100.0:
                break

        return interval


class UnimplementedMethodException(Exception):
    """Exception if someone calls a method that should be overridden."""


class Suite(Pmf):
    """Represents a suite of hypotheses and their probabilities."""

    def update(self, data):
        """Updates each hypothesis based on the data.
        data: any representation of the data
        returns: the normalizing constant
        """
        for hypo in self.values():
            like = self.likelihood(data, hypo)
            self.mult(hypo, like)
        return self.normalize()

    def log_update(self, data):
        """Updates a suite of hypotheses based on new data.
        Modifies the suite directly; if you want to keep the original, make
        a copy.
        Note: unlike Update, LogUpdate does not normalize.
        Args:
            data: any representation of the data
        """
        for hypo in self.values():
            like = self.log_likelihood(data, hypo)
            self.incr(hypo, like)

    def update_set(self, dataset):
        """Updates each hypothesis based on the dataset.
        This is more efficient than calling Update repeatedly because
        it waits until the end to Normalize.
        Modifies the suite directly; if you want to keep the original, make
        a copy.
        dataset: a sequence of data
        returns: the normalizing constant
        """
        for data in dataset:
            for hypo in self.values():
                like = self.likelihood(data, hypo)
                self.mult(hypo, like)
        return self.normalize()

    def log_update_set(self, dataset):
        """Updates each hypothesis based on the dataset.
        Modifies the suite directly; if you want to keep the original, make
        a copy.
        dataset: a sequence of data
        returns: None
        """
        for data in dataset:
            self.log_update(data)

    def likelihood(self, data, hypo):
        """Computes the likelihood of the data under the hypothesis.
        hypo: some representation of the hypothesis
        data: some representation of the data
        """
        raise UnimplementedMethodException()

    def log_likelihood(self, data, hypo):
        """Computes the log likelihood of the data under the hypothesis.
        hypo: some representation of the hypothesis
        data: some representation of the data
        """
        raise UnimplementedMethodException()

    def print(self):
        """Prints the hypotheses and their probabilities."""
        for hypo, prob in sorted(self.items()):
            print(hypo, prob)

    def make_odds(self):
        """Transforms from probabilities to odds.
        Values with prob=0 are removed.
        """
        for hypo, prob in self.items():
            if prob:
                self.set(hypo, odds(prob))
            else:
                self.remove(hypo)

    def make_probs(self):
        """Transforms from odds to probabilities."""
        for hypo, odds in self.items():
            self.set(hypo, probability(odds))

class MakeSuite():
    """make a suite"""

    def from_list(self, l, name=''):
        """Makes a suite from an unsorted sequence of values.
            Args:
                t: sequence of numbers
                name: string name for this suite
            Returns:
                Suite object
            """
        hist = MakeDistribution('hist').from_list(l)
        return self.from_dict(hist)

    def from_hist(self, hist, name=None):
        """Makes a normalized suite from a Hist object.
            Args:
                hist: Hist object
                name: string name
            Returns:
                Suite object
            """
        if name is None:
            name = hist.name

        return self.from_dict(hist, name)

    def from_dict(self, d, name=''):
        """Makes a suite from a map from values to probabilities.
         Args:
             d: dictionary that maps values to probabilities
             name: string name for this suite
         Returns:
             Suite object
         """
        suite = Suite(d, name=name)
        suite.normalize()
        return suite

    def from_cdf(self, cdf, name=None):
        """Makes a normalized Suite from a Cdf object.
          Args:
              cdf: Cdf object
              name: string name for the new Suite
          Returns:
              Suite object
          """
        if name is None:
            name = cdf.name

        suite = Suite(name=name)

        prev = 0.0
        for val, prob in cdf.Items():
            suite.incr(val, prob - prev)
            prev = prob

        return suite


class Pdf(object):
    """Represents a probability density function (PDF)."""

    def density(self, x):
        """Evaluates this Pdf at x.
                Returns: float probability density
                """
        raise UnimplementedMethodException()

    def make_pmf(self, xs, name=''):
        """Makes a discrete version of this Pdf, evaluated at xs.
               xs: equally-spaced sequence of values
               Returns: new Pmf
        """
        pmf = Pmf(name=name)
        for x in xs:
            pmf.set(x, self.density(x))
        pmf.normalize()
        return pmf


class GaussianPdf(Pdf):
    """Represents the PDF of a Gaussian distribution."""

    def __init__(self, mu, sigma):
        """Constructs a Gaussian Pdf with given mu and sigma.
        mu: mean
        sigma: standard deviation
        """
        self.mu = mu
        self.sigma = sigma

    def density(self, x):
        """Evaluates this Pdf at x.
        Returns: float probability density
        """
        return eval_gaussian_pdf(x, self.mu, self.sigma)


class EstimatedPdf(Pdf):
    """Represents a PDF estimated by KDE."""

    def __init__(self, sample):
        """Estimates the density function based on a sample.
        sample: sequence of data
        """
        self.kde = scipy.stats.gaussian_kde(sample)

    def density(self, x):
        """Evaluates this Pdf at x.
        Returns: float probability density
        """
        return self.kde.evaluate(x)

    def make_pmf(self, xs, name=''):
        ps = self.kde.evaluate(xs)
        pmf = MakeDistribution('pmf').from_items(zip(xs, ps), name=name)
        return pmf


def percentile(pmf, percentage):
    """Computes a percentile of a given Pmf.
    percentage: float 0-100
    """
    p = percentage / 100.0
    total = 0
    for val, prob in pmf.Items():
        total += prob
        if total >= p:
            return val


def gredible_interval(pmf, percentage=90):
    """Computes a credible interval for a given distribution.
    If percentage=90, computes the 90% CI.
    Args:
        pmf: Pmf object representing a posterior distribution
        percentage: float between 0 and 100
    Returns:
        sequence of two floats, low and high
    """
    cdf = pmf.make_cdf()
    prob = (1 - percentage / 100.0) / 2
    interval = cdf.value(prob), cdf.value(1 - prob)
    return interval


def pmf_prob_less(pmf1, pmf2):
    """Probability that a value from pmf1 is less than a value from pmf2.
    Args:
        pmf1: Pmf object
        pmf2: Pmf object
    Returns:
        float probability
    """
    total = 0.0
    for v1, p1 in pmf1.items():
        for v2, p2 in pmf2.items():
            if v1 < v2:
                total += p1 * p2
    return total


def pmf_prob_greater(pmf1, pmf2):
    """Probability that a value from pmf1 is less than a value from pmf2.
    Args:
        pmf1: Pmf object
        pmf2: Pmf object
    Returns:
        float probability
    """
    total = 0.0
    for v1, p1 in pmf1.Items():
        for v2, p2 in pmf2.Items():
            if v1 > v2:
                total += p1 * p2
    return total


def pmf_prob_equal(pmf1, pmf2):
    """Probability that a value from pmf1 equals a value from pmf2.
    Args:
        pmf1: Pmf object
        pmf2: Pmf object
    Returns:
        float probability
    """
    total = 0.0
    for v1, p1 in pmf1.Items():
        for v2, p2 in pmf2.Items():
            if v1 == v2:
                total += p1 * p2
    return total


def eval_gaussian_pdf(x, mu, sigma):
    """Computes the unnormalized PDF of the normal distribution.
    x: value
    mu: mean
    sigma: standard deviation

    returns: float probability density
    """
    return scipy.stats.norm.pdf(x, mu, sigma)


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