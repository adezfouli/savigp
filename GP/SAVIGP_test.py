from SAVIGP import *
import copy
from DerApproximator import check_d1
import numpy as np

__author__ = 'AT'

class SAVIGP_test:
    @staticmethod
    def _check_dent_dpi():
        s1 = SAVIGP_test._sample_instance()
        s2 = copy.deepcopy(s1)

        def approx(pi):
            s2.MoG.pi = pi
            s2._update()
            return s2.l_ent()

        def exact(pi):
            s1.MoG.pi = pi
            s1._update()
            return s1._d_ent_d_pi()

        samples = 20
        m = np.random.uniform(low=0.1, high=5.0, size=(samples, s1.num_MoG_comp))
        for s in range(samples):
            check_d1(approx, exact, m[s])


    @staticmethod
    def _check_dent_dS():
        s1 = SAVIGP_test._sample_instance()
        s2 = copy.deepcopy(s1)

        def approx(s):
            s2.MoG.s_from_array(s)
            s2._update()
            return s2.l_ent()

        def exact(s):
            s1.MoG.s_from_array(s)
            s1._update()
            return s1._d_ent_d_S().flatten()

        samples = 1
        m = np.random.uniform(low=0.1, high=1.0, size=(samples, s1.num_MoG_comp *  s1.num_latent_proc * s1.num_inducing))
        for s in range(samples):
            check_d1(approx, exact, m[s])

    @staticmethod
    def _check_dent_dm():
        s1 = SAVIGP_test._sample_instance()
        s2 = copy.deepcopy(s1)

        def approx(m):
            s2.MoG.m_from_array(m)
            s2._update()
            return s2.l_ent()

        def exact(m):
            s1.MoG.m_from_array(m)
            s1._update()
            return s1._d_ent_d_m().flatten()

        samples = 1
        m = np.random.uniform(low=-1.0, high=1.0, size=(samples, s1.num_MoG_comp *  s1.num_latent_proc * s1.num_inducing))
        for s in range(samples):
            check_d1(approx, exact, m[s])


    @staticmethod
    def _check_dcorss_dm():
        s1 = SAVIGP_test._sample_instance()
        s2 = copy.deepcopy(s1)

        def approx(m):
            s2.MoG.m_from_array(m)
            s2._update()
            c, tmp = s2._cross_dcorss_dpi(1)
            return c

        def exact(m):
            s1.MoG.m_from_array(m)
            return s1._dcorss_dm().flatten()

        samples = 20
        m = np.random.uniform(low=-1.0, high=1.0, size=(samples, s1.num_MoG_comp *  s1.num_latent_proc * s1.num_inducing))
        for s in range(samples):
            check_d1(approx, exact, m[s])

    @staticmethod
    def _check_dcorss_ds():
        s1 = SAVIGP_test._sample_instance()
        s2 = copy.deepcopy(s1)

        def approx(s):
            s2.MoG.s_from_array(s)
            s2._update()
            c, tmp = s2._cross_dcorss_dpi(1)
            return c

        def exact(s):
            s1.MoG.s_from_array(s)
            return s1._dcorss_dS().flatten()

        samples = 20
        m = np.random.uniform(low=0.1, high=5.0, size=(samples, s1.num_MoG_comp *  s1.num_latent_proc * s1.num_inducing))
        for s in range(samples):
            check_d1(approx, exact, m[s])


    @staticmethod
    def _check_dcorss_dpi():
        s1 = SAVIGP_test._sample_instance()
        s2 = copy.deepcopy(s1)

        def approx(pi):
            s2.MoG.pi = pi
            s2._update()
            c, tmp = s2._cross_dcorss_dpi(1)
            return c

        def exact(pi):
            s1.MoG.pi = pi
            c, dpi = s1._cross_dcorss_dpi(1)
            return dpi

        samples = 20
        m = np.random.uniform(low=0.1, high=5.0, size=(samples, s1.num_MoG_comp))
        for s in range(samples):
            check_d1(approx, exact, m[s])


    @staticmethod
    def generate_samples(num_samples, input_dim):
        # seed=1000
        noise=0.02
        # np.random.seed(seed=seed)
        X = np.random.uniform(low=-1.0, high=1.0, size=(num_samples, input_dim))
        X.sort(axis=0)
        rbf = GPy.kern.rbf(input_dim, variance=1., lengthscale=np.array((0.25,)))
        white = GPy.kern.white(input_dim, variance=noise)
        kernel = rbf + white
        Y = np.sin([X.sum(axis=1).T]).T + np.random.randn(num_samples, 1) * 0.05
        return X, Y, kernel, noise


    @staticmethod
    def _sample_instance():
        num_input_samples = 20
        input_dim = 5
        num_inducing = 6
        num_MoG = 3
        num_latent_proc = 3
        num_samples = 1000

        X, Y, kernel, noise = SAVIGP_test.generate_samples(num_input_samples, input_dim)
        np.random.seed()
        return SAVIGP(X, Y, num_inducing, num_MoG, num_latent_proc, SAVIGP.normal_likelihood(1), False)

if __name__ == '__main__':
    SAVIGP_test._check_dcorss_dm()