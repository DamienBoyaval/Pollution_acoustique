import unittest
import tempfile
import numpy as np
import material_alpha
import comparaison_materiaux as comp
import processing
import demo_control_polycopie2024 as demo


class ComparisonTests(unittest.TestCase):
    def test_current_tables(self):
        grids=[]
        for material in material_alpha.MATERIALS:
            omega,alpha,error=material_alpha.load_material(material)
            self.assertEqual(len(omega),1000)
            np.testing.assert_allclose(omega/(2*np.pi),np.linspace(500,4000,1000))
            grids.append(omega)
        np.testing.assert_array_equal(*grids)

    def test_cached_solver_matches_professor(self):
        for N,level in [(16,0),(16,1),(32,2)]:
            q=comp.make_problem(N,level)
            d,h,args,chi=q
            # Nonzero complex forcing also exercises differentiated Robin RHS.
            args[0][d==-1]=.1+.2j
            args[3][:]=.2-.1j
            args[4][:]=1.1+.1j
            evaluator=comp.Evaluator(q)
            for material in material_alpha.MATERIALS:
                omega,alpha,_=material_alpha.load_material(material)
                ks,alphas=material_alpha.solver_parameters(omega[:1],alpha[:1],sound_speed=340)
                k,A=ks[0],alphas[0]
                u,mat,rhs=processing.solve_helmholtz(d,h,k,*args,A*chi,return_system=True)
                p=demo.your_compute_adjoint(d,u,h,mat)
                grad=demo.your_compute_gradient(d,u,p,h,args[3],args[4],args[-1],A,chi)
                values,actual=evaluator.evaluate(chi,ks,alphas,True)
                self.assertAlmostEqual(values[0],demo.your_compute_objective_function(d,u,h),places=10)
                np.testing.assert_allclose(actual,grad,rtol=1e-9,atol=1e-10)

    def test_complete_small_comparison(self):
        with tempfile.TemporaryDirectory() as out:
            rows,config=comp.run_comparison(N=16,indices=[0,1,2],max_iter=5,output=out)
            self.assertEqual(len(rows),2)
            self.assertEqual(len(config['table_indices']),3)
            self.assertAlmostEqual(rows[0]['material_quantity'],rows[1]['material_quantity'],places=10)
            for row in rows:
                self.assertLess(row['target_final_mean'],row['target_initial_mean'])
                self.assertTrue(np.isfinite(row['energy_final_max']))


if __name__=='__main__':
    unittest.main(verbosity=2)
