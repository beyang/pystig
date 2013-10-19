#!/usr/bin/env python

'''
Probability distributions with sampling routines.
'''

import numpy as np
import random
import stig


class Distribution(object):
    def sample(self, rng):
        stig.abs_impl()


class DefaultRNGDistribution(Distribution):
    def __init__(self, dist, rng):
        self.dist = dist
        self.rng = rng

    def sample(self, rng=None):
        rng = rng or self.rng
        return self.dist.sample(rng=rng)


class Bernoulli(Distribution):
    def __init__(self, p):
        self.p = p

    def sample(self, rng=random):
        return stig.coinflip(self.p, rng=rng)


class Multinomial(Distribution):
    def __init__(self, ps, cdf=False):
        if cdf:
            self.cdf = ps
        else:
            self.cdf = stig.cumsum(ps)

    def sample(self, rng=random):
        return stig.diceroll(self.cdf, rng=rng)


class Gaussian(Distribution):
    def __init__(self, mean, cov):
        self.mean = np.array(mean)
        self.cov = np.array(cov)

    def sample(self, rng=random, from_np=list):
        return from_np(np.random.multivariate_normal(self.mean, self.cov))


if __name__ == '__main__':
    # TODO: unit tests
    pass
