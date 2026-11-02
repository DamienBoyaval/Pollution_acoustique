import unittest
import numpy as np
import material_alpha as mat
import etudes
import demo_control_polycopie2024 as demo
import _env
import tempfile
import shutil
from pathlib import Path


class MaterialIntegration(unittest.TestCase):
    def test_accented_windows_directory(self):
        with tempfile.TemporaryDirectory(prefix='Système-') as directory:
            for file in (Path(mat.__file__).parent/'data_alpha').glob('*.mtx'):
                shutil.copy2(file,Path(directory)/file.name)
            for material in mat.MATERIALS:
                expected=mat.load_material(material)
                actual=mat.load_material(material,directory)
                for a,b in zip(expected,actual):
                    np.testing.assert_array_equal(a,b)

    def test_tables_and_scaling(self):
        grids=[]
        for material in mat.MATERIALS:
            omega,alpha,error=mat.load_material(material)
            self.assertEqual(len(omega),1000)
            np.testing.assert_allclose(omega/(2*np.pi),np.linspace(500,4000,1000))
            k,a=mat.solver_parameters(omega,alpha,length_m=2,sound_speed=343)
            np.testing.assert_allclose(k,omega*2/343)
            np.testing.assert_allclose(a,alpha*2)
            grids.append(omega)
        np.testing.assert_array_equal(*grids)

    def test_material_gradients_and_descent(self):
        problem=etudes.make_problem(16,0)
        domain,_,_,chi=problem
        mask=domain == _env.NODE_ROBIN
        chi=np.zeros_like(chi)
        chi[mask]=.4
        rng=np.random.default_rng(52)
        direction=np.zeros_like(chi)
        direction[mask]=rng.normal(size=mask.sum())
        direction[mask]-=direction[mask].mean()
        direction/=np.linalg.norm(direction)
        for material in mat.MATERIALS:
            omega,alpha,_=mat.load_material(material)
            ks,alphas=mat.solver_parameters(omega[:3],alpha[:3])
            J,_,grad=etudes.evaluate(problem,chi,ks,alphas,gradient=True)
            eps=1e-4
            fd=(etudes.evaluate(problem,chi+eps*direction,ks,alphas)[0]-
                etudes.evaluate(problem,chi-eps*direction,ks,alphas)[0])/(2*eps)
            adj=np.sum(grad*direction)
            error=abs(fd-adj)/max(abs(fd),abs(adj),1e-12)
            self.assertLess(error,2e-6)
            final,history,_=etudes.optimize_frequencies(problem,chi,ks,alphas,.4,max_iter=5)
            self.assertLess(history[-1],J)
            self.assertAlmostEqual(final.sum(),chi.sum(),places=10)
            print(material,'gradient relative error',error,'energy',J,'->',history[-1])


if __name__=='__main__':
    unittest.main(verbosity=2)
