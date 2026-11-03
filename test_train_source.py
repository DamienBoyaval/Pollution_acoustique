import unittest
import numpy as np
import _env
import train_source as train
import comparaison_materiaux as comp
import processing as pro


class TrainSourceTests(unittest.TestCase):
    def test_spectral_peaks_and_speed_scaling(self):
        frequencies = np.array([500., 1425., 2250., 3000.])
        rolling, aero = train.spectral_amplitudes(frequencies)
        self.assertEqual(np.argmax(rolling), 2)
        self.assertEqual(np.argmax(aero), 1)
        self.assertAlmostEqual(rolling[2], 1.)
        self.assertAlmostEqual(aero[1], 1.)
        for index, center in enumerate([2250., 1425.]):
            values = train.spectral_amplitudes(np.array([center/2, center, center*2]))[index]
            np.testing.assert_allclose(values[0], values[2])
        r2, a2 = train.spectral_amplitudes(frequencies, speed_kmh=600.)
        np.testing.assert_allclose((r2 / rolling)**2, 8.)
        np.testing.assert_allclose((a2 / aero)**2, 64.)
        np.testing.assert_array_equal(train.spectral_amplitudes(500., speed_kmh=0.), [0., 0.])

    def test_source_frequency_and_grid_coordinates(self):
        d = np.full((100, 50), _env.NODE_INTERIOR)
        d[0] = _env.NODE_DIRICHLET
        s500, _ = train.gaussian_train_source(d, .02, 500.)
        s2000, _ = train.gaussian_train_source(d, .02, 2000.)
        self.assertFalse(np.allclose(s500, s2000))
        self.assertTrue(np.all(s500[0] == 0))
        fine = np.full((200, 100), _env.NODE_INTERIOR)
        fine[0] = _env.NODE_DIRICHLET
        sfine, _ = train.gaussian_train_source(fine, .01, 500.)
        np.testing.assert_allclose(sfine[::2, ::2], s500)

    def test_linear_solver_energy_scaling(self):
        d, h, args, chi = comp.make_problem(16)
        _, fd, fn, fr, bp, ap, ad, bn, br = args
        fd = np.zeros_like(fd)
        source, _ = train.gaussian_train_source(d, h, 500.)
        k = 2 * np.pi * 500 / 340
        fields = []
        for scale in [1., 2.]:
            fields.append(pro.solve_helmholtz(
                d, h, k, scale * source, fd, fn, fr,
                bp, ap, ad, bn, br, (10-10j)*chi))
        np.testing.assert_allclose(fields[1], 2*fields[0], atol=1e-12)
        e0, e1 = [np.sum(abs(u[d == _env.NODE_INTERIOR])**2) for u in fields]
        self.assertAlmostEqual(e1/e0, 4.)

    def test_invalid_inputs(self):
        for frequency in [0., -1., np.nan, np.inf]:
            with self.assertRaises(ValueError):
                train.spectral_amplitudes(frequency)
        for kwargs in [dict(speed_kmh=-1), dict(rolling_width_octaves=0),
                       dict(aero_peak_hz=0), dict(aero_gain=-1)]:
            with self.assertRaises(ValueError):
                train.spectral_amplitudes(500., **kwargs)


if __name__ == '__main__':
    unittest.main()
