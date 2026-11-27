"""ES math against hand calculations: shaping, antithetic symmetry, gradient, Adam/SGD, AWD."""
import unittest

import numpy as np

from es.data import derive_seed
from es.es_loop import (Optimizer, apply_awd, centered_ranks, es_coefficients, es_update, gradient_from_coeffs,
                        lr_at, noise, noise_seeds, shape_fitness)

EPS = {11: np.array([1.0, 0.0, -1.0], dtype=np.float32), 22: np.array([0.5, 2.0, 0.0], dtype=np.float32)}


def fixed_noise(seed, dim):
    return EPS[seed]


class ShapingTests(unittest.TestCase):
    def test_centered_ranks_plain(self):
        self.assertTrue(np.allclose(centered_ranks([3, 1, 0, 2]), [0.5, -1 / 6, -0.5, 1 / 6]))

    def test_centered_ranks_average_ties(self):
        self.assertTrue(np.allclose(centered_ranks([1, 1, 3, 0]), [0.0, 0.0, 0.5, -0.5]))
        self.assertTrue(np.allclose(centered_ranks([2, 2, 2, 2]), [0, 0, 0, 0]))

    def test_centered_ranks_sum_to_zero(self):
        x = np.random.default_rng(0).integers(0, 5, size=33)
        self.assertAlmostEqual(float(centered_ranks(x).sum()), 0.0, places=12)

    def test_zscore(self):
        up, um = shape_fitness([1.0, 3.0], [1.0, 3.0], "zscore")
        self.assertTrue(np.allclose(up, [-1, 1]) and np.allclose(um, [-1, 1]))


class AntitheticTests(unittest.TestCase):
    def test_equal_pair_fitness_gives_zero_gradient(self):
        f = [0.3, -0.2, 0.9, 0.9]
        up, um = shape_fitness(f, f)
        c = es_coefficients(up, um, 0.05)
        self.assertTrue(np.array_equal(c, np.zeros(4)))
        self.assertTrue(np.array_equal(gradient_from_coeffs(c, noise_seeds(1, 0, 4), 50), np.zeros(50)))

    def test_swapping_signs_negates_coefficients(self):
        fp, fm = [0.1, 0.7, 0.4], [0.5, 0.2, 0.4]
        c1 = es_coefficients(*shape_fitness(fp, fm), 0.1)
        c2 = es_coefficients(*shape_fitness(fm, fp), 0.1)
        self.assertTrue(np.allclose(c1, -c2))
        self.assertEqual(c1[2], 0.0)

    def test_noise_is_reproducible_and_distinct(self):
        s = noise_seeds(7, 3, 4)
        self.assertEqual(s, noise_seeds(7, 3, 4))
        self.assertEqual(len(set(s + noise_seeds(7, 4, 4))), 8)
        self.assertTrue(np.array_equal(noise(s[0], 1000), noise(s[0], 1000)))
        big = noise(123, 200000)
        self.assertEqual(big.dtype, np.float32)
        self.assertLess(abs(float(big.mean())), 0.01)
        self.assertLess(abs(float(big.std()) - 1.0), 0.01)

    def test_known_answers(self):
        # Changing the seed derivation or the noise generator would silently break resume of
        # journals written on the pod; these values pin both.
        self.assertEqual(derive_seed(0, "noise", 0, 0), 2884587683080375433)
        self.assertEqual(noise_seeds(20261128, 3, 2), [4810673841713909737, 8037089853988272763])
        self.assertTrue(np.allclose(noise(12345, 4), [-1.2160426378250122, -0.3693346381187439,
                                                      1.8661257028579712, -0.22194042801856995]))


class UpdateHandCalculation(unittest.TestCase):
    """D = 3, N = 2, sigma = 0.5, eps_1 = [1, 0, -1], eps_2 = [0.5, 2, 0], f+ = [3, 1], f- = [0, 2].

    Ranks of [3, 1, 0, 2] are [3, 1, 0, 2]; u = rank / 3 - 0.5 gives u+ = [1/2, -1/6], u- = [-1/2, 1/6].
    c_j = (u+_j - u-_j) / (2 N sigma) = [1/2, -1/6].
    g = 1/2 [1, 0, -1] - 1/6 [0.5, 2, 0] = [5/12, -1/3, -1/2].
    """
    G = np.array([5 / 12, -1 / 3, -1 / 2])

    def coeffs(self):
        up, um = shape_fitness([3.0, 1.0], [0.0, 2.0])
        self.assertTrue(np.allclose(up, [0.5, -1 / 6]) and np.allclose(um, [-0.5, 1 / 6]))
        c = es_coefficients(up, um, 0.5)
        self.assertTrue(np.allclose(c, [0.5, -1 / 6]))
        return c

    def test_gradient(self):
        g = gradient_from_coeffs(self.coeffs(), [11, 22], 3, fixed_noise)
        self.assertTrue(np.allclose(g, self.G, atol=1e-12))

    def test_sgd_with_awd(self):
        opt = Optimizer({"kind": "sgd"}, 3)
        theta = np.zeros(3, dtype=np.float32)
        new, info = es_update(theta, self.coeffs(), [11, 22], opt, lr=0.1, awd=0.1, noise_fn=fixed_noise)
        # theta1 = 0.9 * 0.1 * g = [0.0375, -0.03, -0.045]
        self.assertTrue(np.allclose(new, [0.0375, -0.03, -0.045], atol=1e-7))
        self.assertEqual(new.dtype, np.float32)
        self.assertAlmostEqual(info["grad_norm"], float(np.linalg.norm(self.G)), places=9)

    def test_sgd_momentum(self):
        opt = Optimizer({"kind": "sgd", "momentum": 0.5}, 3)
        theta = np.zeros(3, dtype=np.float32)
        t1, _ = es_update(theta, self.coeffs(), [11, 22], opt, 0.1, 0.0, noise_fn=fixed_noise)
        t2, _ = es_update(t1, self.coeffs(), [11, 22], opt, 0.1, 0.0, noise_fn=fixed_noise)
        # buffer: g, then 1.5 g; theta2 = 0.1 g + 0.15 g = 0.25 g
        self.assertTrue(np.allclose(t2, 0.25 * self.G, atol=1e-7))

    def test_adam_two_steps(self):
        opt = Optimizer({"kind": "adam", "beta1": 0.9, "beta2": 0.999, "eps": 1e-8}, 3)
        theta = np.zeros(3, dtype=np.float32)
        t1, _ = es_update(theta, self.coeffs(), [11, 22], opt, 0.1, 0.0, noise_fn=fixed_noise)
        # step 1: mhat = g, vhat = g^2, update = lr * sign(g)
        self.assertTrue(np.allclose(t1, [0.1, -0.1, -0.1], atol=1e-6))
        # step 2 with the reversed gradient: m = 0.09 g - 0.1 g = -0.01 g, mhat = -0.01 / 0.19 g,
        # v = 0.001999 g^2, vhat = g^2, so theta2 = 0.1 sign(g) (1 - 1/19)
        t2, _ = es_update(t1, -self.coeffs(), [11, 22], opt, 0.1, 0.0, noise_fn=fixed_noise)
        self.assertTrue(np.allclose(t2, np.array([1, -1, -1]) * 0.1 * (18 / 19), atol=1e-6))
        self.assertEqual(opt.t, 2)

    def test_awd_and_clip(self):
        self.assertTrue(np.allclose(apply_awd(np.array([1.0, -2.0]), 0.25), [0.75, -1.5]))
        self.assertTrue(np.allclose(apply_awd(np.array([1.0]), 0.5, anchor=np.array([3.0])), [2.0]))
        opt = Optimizer({"kind": "sgd"}, 3)
        new, _ = es_update(np.zeros(3, dtype=np.float32), self.coeffs(), [11, 22], opt, 10.0, 0.0,
                           theta_min=-0.95, theta_max=1.0, noise_fn=fixed_noise)
        self.assertTrue(np.allclose(new, [1.0, -0.95, -0.95]))

    def test_max_update_norm(self):
        opt = Optimizer({"kind": "sgd", "max_update_norm": 0.01}, 3)
        new, _ = es_update(np.zeros(3, dtype=np.float32), self.coeffs(), [11, 22], opt, 1.0, 0.0,
                           noise_fn=fixed_noise)
        self.assertAlmostEqual(float(np.linalg.norm(new)), 0.01, places=6)

    def test_lr_from_sigma(self):
        es = {"optimizer": {"lr": None, "lr_sigma2_coef": 1.7}, "sigma": 0.05, "lr_schedule": "constant"}
        self.assertAlmostEqual(lr_at(es, 7), 1.7 * 0.0025)

    def test_cosine_schedule(self):
        es = {"optimizer": {"lr": 0.01}, "lr_schedule": "cosine", "lr_min_frac": 0.1, "generations": 100}
        self.assertAlmostEqual(lr_at(es, 0), 0.01)
        self.assertAlmostEqual(lr_at(es, 100), 0.001)
        self.assertAlmostEqual(lr_at(es, 50), 0.0055)


if __name__ == "__main__":
    unittest.main()
