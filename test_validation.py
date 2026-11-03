"""Run with: python -m unittest -v test_validation.py"""
import contextlib
import io
import unittest
import numpy as np
import _env
import preprocessing as pre
import processing as pro
import demo_control_polycopie2024 as demo
import etudes
import tempfile
from pathlib import Path


def problem(N=16, level=0):
    M, h = 2*N, 1/N
    domain, x, y, _, _ = pre._set_geometry_of_domain(M, N, level)
    bp, ap, ad, bn, ar, br = pre._set_coefficients_of_pde(M, N)
    f, fd, fn, fr = pre._set_rhs_of_pde(M, N)
    fd[0, :] = 1
    chi = pre.set2zero(pre._set_chi(M, N, x, y), domain)
    return domain, h, 10., f, fd, fn, fr, bp, ap, ad, bn, br, 10-10j, chi


def state(q, chi=None):
    domain, h, omega, f, fd, fn, fr, bp, ap, ad, bn, br, Alpha, initial = q
    if chi is None:
        chi = initial
    return pro.solve_helmholtz(domain, h, omega, f, fd, fn, fr,
                              bp, ap, ad, bn, br, Alpha*chi, return_system=True)


def gradient(q, chi):
    d, h, _, _, _, _, fr, bp, _, _, _, br, Alpha, _ = q
    u, A, b = state(q, chi)
    p = demo.your_compute_adjoint(d, u, h, A)
    return demo.your_compute_gradient(d, u, p, h, fr, bp, br, Alpha, chi)


class Validation(unittest.TestCase):
    def test_multifrequency(self):
        problem = etudes.make_problem(16,1)
        d, _, _, chi = problem
        volume = chi.sum()/np.count_nonzero(d == _env.NODE_ROBIN)
        chi = demo.your_project_density(chi*.4+.2,d,volume)
        ks, alphas, weights = [3.,7.], [8-6j,10-10j], [.3,.7]
        J, _, grad = etudes.evaluate(problem,chi,ks,alphas,weights,True)
        direction = np.zeros_like(chi)
        mask = d == _env.NODE_ROBIN
        direction[mask] = np.random.default_rng(9).normal(size=mask.sum())
        direction[mask] -= direction[mask].mean()
        direction /= np.linalg.norm(direction)
        eps = 1e-4
        fd = (etudes.evaluate(problem,chi+eps*direction,ks,alphas,weights)[0]-
              etudes.evaluate(problem,chi-eps*direction,ks,alphas,weights)[0])/(2*eps)
        adj = np.sum(grad*direction)
        rel = abs(fd-adj)/max(abs(fd),abs(adj),1e-12)
        print('multifrequency adj=%g fd=%g relative_error=%g'%(adj,fd,rel))
        self.assertLess(rel,1e-6)
        final, history, grad = etudes.optimize_frequencies(problem,chi,ks,alphas,volume,max_iter=12,weights=weights)
        self.assertTrue(np.all(np.diff(history) < 0))
        self.assertLessEqual(history[-1],J)
        self.assertAlmostEqual(final.sum(),chi.sum(),places=9)
        with self.assertRaises(ValueError):
            etudes.evaluate(problem,chi,[100],[10-10j])

    def test_material_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp)/'material.csv'
            file.write_text('frequency_hz,alpha_real_per_m,alpha_imag_per_m\n250,1,-2\n4000,3,-6\n')
            np.testing.assert_allclose(etudes.material_coefficients(file,[250,2125,4000],2),[2-4j,4-8j,6-12j])
            with self.assertRaises(ValueError):
                etudes.material_coefficients(file,[100],1)

    def test_objective(self):
        d = np.full((10, 10), _env.NODE_INTERIOR)
        u = np.ones((10, 10), complex)*(1+2j)
        self.assertAlmostEqual(demo.your_compute_objective_function(d, u, .1), 5)
        d[0] = _env.NODE_COMPLEMENTARY
        self.assertAlmostEqual(demo.your_compute_objective_function(d, u, .1), 4.5)

    def test_projection(self):
        d = problem()[0]
        mask = d == _env.NODE_ROBIN
        z = np.random.default_rng(3).normal(size=d.shape)*100
        for v in (0, .13, .4, 1):
            chi = demo.your_project_density(z, d, v)
            self.assertTrue(np.all((chi >= 0) & (chi <= 1)))
            self.assertTrue(np.all(chi[~mask] == 0))
            self.assertAlmostEqual(chi.sum(), v*mask.sum(), places=10)
            np.testing.assert_allclose(demo.your_project_density(chi, d, v), chi, atol=1e-12)
            # KKT condition: a common shift on every unsaturated component.
            free = mask & (chi > 1e-10) & (chi < 1-1e-10)
            if free.sum() > 1:
                self.assertLess(np.ptp((chi-z)[free]), 1e-10)
        with self.assertRaises(ValueError):
            demo.your_project_density(z, d, 1.1)

    def test_linear_system_and_backward_compatibility(self):
        q = problem()
        u, A, b = state(q)
        self.assertLess(np.linalg.norm(A @ u.ravel()-b.ravel()), 1e-10)
        d,h,w,f,fd,fn,fr,bp,ap,ad,bn,br,Alpha,chi = q
        np.testing.assert_allclose(u, pro.solve_helmholtz(d,h,w,f,fd,fn,fr,bp,ap,ad,bn,br,Alpha*chi))
        p = demo.your_compute_adjoint(d,u,h,A)
        g = h*h*u
        self.assertLess(np.linalg.norm(A.conj().T @ p.ravel()-g.ravel()), 1e-10)

    def test_adjoint_finite_differences(self):
        # Four geometries/configurations, three independent directions,
        # several epsilons; include nonzero Robin forcing and complex PDE data.
        for N, level, forced in [(16,0,False),(16,1,False),(32,2,False),(16,1,True)]:
            q = list(problem(N, level))
            d,h = q[:2]
            mask = d == _env.NODE_ROBIN
            rng = np.random.default_rng(42)
            chi = np.zeros(d.shape)
            chi[mask] = rng.uniform(.2,.8,mask.sum())
            if forced:
                q[3][d == _env.NODE_INTERIOR] = .2+.1j
                q[6][mask] = .3-.2j
                q[7][:] = 1.1+.1j
                q[11][:] = .9+.2j
            grad = gradient(q, chi)
            for trial in range(3):
                direction = np.zeros_like(chi)
                direction[mask] = rng.normal(size=mask.sum())
                if trial != 0:
                    direction[mask] -= direction[mask].mean()
                direction /= np.linalg.norm(direction)
                adj = np.sum(grad*direction)
                errors = []
                for eps in (1e-3, 1e-4, 1e-5):
                    up = state(q,chi+eps*direction)[0]
                    um = state(q,chi-eps*direction)[0]
                    fd = (demo.your_compute_objective_function(d,up,h)-
                          demo.your_compute_objective_function(d,um,h))/(2*eps)
                    rel = abs(fd-adj)/max(abs(fd),abs(adj),1e-12)
                    errors.append(rel)
                print('gradient N=%d level=%d forced=%s direction=%d adj=%.12g errors=%s'
                      % (N,level,forced,trial,adj,errors))
                self.assertLess(min(errors), 2e-6)

    def test_optimization(self):
        for N,level,volume,mu in [(16,0,.4,5),(16,1,.4,5),(16,0,0,5),
                                 (16,0,1,5),(16,0,.4,0),(16,0,.4,1e6)]:
            q = problem(N,level)
            d,h,w,f,fd,fn,fr,bp,ap,ad,bn,br,Alpha,chi = q
            with contextlib.redirect_stdout(io.StringIO()):
                result = demo.your_optimization_procedure(d,h,w,f,fd,fn,fr,bp,ap,ad,bn,br,Alpha*chi,Alpha,mu,chi,volume)
            final, history, u, grad = result
            self.assertTrue(np.all(np.isfinite(history)))
            self.assertTrue(np.all(np.diff(history[:,0]) < 0))
            self.assertAlmostEqual(final.sum(), volume*np.count_nonzero(d == _env.NODE_ROBIN), places=9)
            self.assertAlmostEqual(history[-1,0],demo.your_compute_objective_function(d,u,h))
            np.testing.assert_allclose(u,state(q,final)[0])
            np.testing.assert_allclose(grad,gradient(q,final),atol=1e-12)
            print('optimization N=%d level=%d volume=%g mu=%g accepted=%d E0=%.12g Ef=%.12g'
                  % (N,level,volume,mu,len(history)-1,history[0,0],history[-1,0]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
