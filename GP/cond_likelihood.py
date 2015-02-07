import scipy

__author__ = 'AT'


def normal_likelihood(epsilon):
    def ll(f, y):
        return scipy.stats.norm.logpdf(y, loc=f[0], scale=epsilon)

    return ll


def multivariate_likelihood(sigma):
    def ll(f, y):
        return scipy.stats.multivariate_normal.logpdf(y, mean=f, cov=sigma)

    return ll
