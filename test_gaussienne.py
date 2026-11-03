import unittest
import numpy as np
import comparaison_materiaux as comp
import essai_source_gaussienne as experiment
import processing


class GaussianTests(unittest.TestCase):
    def test_only_interior_source_changes(self):
        base=comp.make_problem(16)
        snapshots=[a.copy() for a in base[2]]
        changed,gaussian=experiment.add_gaussian(base)
        np.testing.assert_array_equal(base[0],changed[0])
        np.testing.assert_array_equal(base[-1],changed[-1])
        np.testing.assert_array_equal(base[2][0],snapshots[0])
        np.testing.assert_array_equal(changed[2][0],snapshots[0]+gaussian)
        for old,new in zip(snapshots[1:],changed[2][1:]):
            np.testing.assert_array_equal(old,new)
        self.assertTrue(np.all(gaussian[base[0]!=-1]==0))
        self.assertAlmostEqual(gaussian[4,8],1.)

    def test_linear_response_and_residual(self):
        base=comp.make_problem(16)
        changed,g=experiment.add_gaussian(base)
        d,h,args,chi=base
        k,Alpha=8.,19-22j
        initial=processing.solve_helmholtz(d,h,k,*args,Alpha*chi)
        after,A,b=processing.solve_helmholtz(d,h,k,*changed[2],Alpha*chi,return_system=True)
        self.assertLess(np.linalg.norm(A@after.ravel()-b.ravel()),1e-10)
        # Difference must equal the Gaussian response with homogeneous boundaries.
        gaussian_only=list(args)
        gaussian_only[0]=g.astype(complex)
        gaussian_only[1]=np.zeros_like(args[1])
        isolated=processing.solve_helmholtz(d,h,k,*gaussian_only,Alpha*chi)
        np.testing.assert_allclose(after-initial,isolated,rtol=1e-9,atol=1e-12)
        self.assertGreater(np.linalg.norm(after-initial),1e-5)


if __name__=='__main__':
    unittest.main(verbosity=2)
