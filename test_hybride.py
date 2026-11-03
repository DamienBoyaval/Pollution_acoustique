import unittest
import numpy as np
import geometrie_hybride as geo
import comparaison_materiaux as comp
import material_alpha
import processing
import essai_source_gaussienne as gaussian
import essai_fractal_resonateur as experiment
import plot_pics_hybrides as spectral
import _env


class HybridTests(unittest.TestCase):
    def test_geometry_and_boundary_families(self):
        for level in [0,1,2]:
            flat,mask0,meta0=geo.build_problem(128,level,False)
            mixed,mask1,meta1=geo.build_problem(128,level,True)
            d,h,args,chi=mixed
            np.testing.assert_array_equal(mask0,mask1)
            np.testing.assert_array_equal(flat[0]==_env.NODE_ROBIN,d==_env.NODE_ROBIN)
            np.testing.assert_array_equal(flat[0]==_env.NODE_DIRICHLET,d==_env.NODE_DIRICHLET)
            np.testing.assert_array_equal(flat[2][1],args[1])
            np.testing.assert_array_equal(flat[3],chi)
            self.assertEqual(meta1['air_components'],1)
            self.assertGreater(meta1['total_area_normalized'],meta0['total_area_normalized'])
            self.assertAlmostEqual(meta1['quantity_discrete'],.4,places=10)
            self.assertTrue(np.all(chi[d!=_env.NODE_ROBIN]==0))
            changed,g=gaussian.add_gaussian(mixed)
            for a,b in zip(args[1:],changed[2][1:]):
                np.testing.assert_array_equal(a,b)

    def test_padding_reproduces_original_domain(self):
        original=comp.make_problem(128,1)
        padded,mask,meta=geo.build_problem(128,1,False)
        d,h,args,raw=original
        chi=comp.demo.your_project_density(raw,d,.4/(h*np.count_nonzero(d==_env.NODE_ROBIN)))
        u=processing.solve_helmholtz(d,h,9.3,*args,(19-22j)*chi)
        v=processing.solve_helmholtz(padded[0],h,9.3,*padded[2],(19-22j)*padded[3])
        np.testing.assert_allclose(v[:,:128],u,rtol=1e-9,atol=1e-10)
        self.assertTrue(np.all(v[:,128:]==0))

    def test_hybrid_gradient_and_spatial_energy(self):
        problem,main_mask,meta=geo.build_problem(128,1,True)
        problem,g=gaussian.add_gaussian(problem)
        d,h,args,chi=problem
        evaluator=comp.Evaluator(problem)
        rng=np.random.default_rng(77)
        direction=np.zeros_like(chi)
        mask=d==_env.NODE_ROBIN
        direction[mask]=rng.normal(size=mask.sum())
        direction[mask]-=direction[mask].mean()
        direction/=np.linalg.norm(direction)
        for material in material_alpha.MATERIALS:
            omega,alpha,_=material_alpha.load_material(material)
            ks,alphas=material_alpha.solver_parameters(omega[[0,32]],alpha[[0,32]],sound_speed=340)
            energies,grad=evaluator.evaluate(chi,ks,alphas,True)
            adj=float(np.sum(grad*direction))
            errors=[]
            for epsilon in [1e-4,1e-5]:
                plus,_=evaluator.evaluate(chi+epsilon*direction,ks,alphas)
                minus,_=evaluator.evaluate(chi-epsilon*direction,ks,alphas)
                fd=float((plus.mean()-minus.mean())/(2*epsilon))
                errors.append(abs(fd-adj)/max(abs(fd),abs(adj),1e-12))
            self.assertLess(min(errors),2e-6)
            total,main,fields=experiment.spatial_metrics(problem,chi,ks,alphas,main_mask)
            np.testing.assert_allclose(total,energies,rtol=1e-9,atol=1e-10)
            self.assertTrue(np.all(total>=main-1e-12))
            for i,(k,Alpha) in enumerate(zip(ks,alphas)):
                t,m=spectral.energy_components(evaluator,chi,k,Alpha,main_mask)
                np.testing.assert_allclose([t,m],[total[i],main[i]],rtol=1e-9,atol=1e-10)
            print(material,'hybrid gradient relative errors:',errors)


if __name__=='__main__':
    unittest.main(verbosity=2)
