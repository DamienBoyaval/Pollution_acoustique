"""Separate Gaussian volume-source experiment, preserving every boundary condition."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import _env
import comparaison_materiaux as comp
import material_alpha
import processing
import demo_control_polycopie2024 as demo


def add_gaussian(problem, amplitude=1., center=(.5,.25), sigma=.08):
    """Coordinates and sigma are in the professor's normalized geometry.

    The original plane-wave Dirichlet boundary remains present. Only f changes.
    """
    d,h,args,chi=problem
    if not np.isfinite(sigma) or sigma<=0 or not np.isfinite(amplitude) or not np.all(np.isfinite(center)):
        raise ValueError('Require finite source parameters and sigma > 0.')
    y,x=np.indices(d.shape,dtype=float)
    x*=h;y*=h
    gaussian=amplitude*np.exp(-((x-center[0])**2+(y-center[1])**2)/(2*sigma**2))
    gaussian[d!=_env.NODE_INTERIOR]=0
    new_args=list(args)
    new_args[0]=args[0].copy()+gaussian
    return (d,h,tuple(new_args),chi),gaussian


def run(N=50,level=0,frequency_index=0,amplitude=1.,sigma=.08,
        center=(.5,.25),length_m=1.,sound_speed=340.,output=None):
    base=comp.make_problem(N,level)
    changed,gaussian=add_gaussian(base,amplitude,center,sigma)
    d,h,args,chi=base
    # Preserve values and boundary types exactly, not merely their names.
    boundary_snapshots=[a.copy() for a in args[1:]]
    domain_snapshot=d.copy()
    out=Path(output) if output else Path(__file__).resolve().parent/'resultats_source_gaussienne'
    out.mkdir(parents=True,exist_ok=True)
    summaries=[]
    print('TEST SEPARE : onde plane au bord + source gaussienne dans le volume.',flush=True)
    print(f'Gaussienne : amplitude={amplitude:g}, centre={center}, sigma={sigma:g} (coordonnees normalisees).',flush=True)
    for material in material_alpha.MATERIALS:
        omega,alpha,_=material_alpha.load_material(material)
        if not 0<=frequency_index<len(omega):
            raise ValueError('Invalid frequency table index.')
        k,Alpha=material_alpha.solver_parameters(omega[frequency_index],alpha[frequency_index],length_m,sound_speed)
        k,Alpha=float(k),complex(Alpha)
        if k*h>2*np.pi/10:
            raise ValueError('Increase N to maintain at least 10 points per wavelength.')
        before=processing.solve_helmholtz(d,h,k,*args,Alpha*chi)
        after=processing.solve_helmholtz(d,h,k,*changed[2],Alpha*chi)
        E0=demo.your_compute_objective_function(d,before,h)
        E1=demo.your_compute_objective_function(d,after,h)
        row={'material':material,'frequency_hz':float(omega[frequency_index]/(2*np.pi)),
             'omega_rad_s':float(omega[frequency_index]),'k_solver':k,
             'alpha_solver_real':Alpha.real,'alpha_solver_imag':Alpha.imag,
             'energy_plane_boundary':float(E0),'energy_plane_boundary_plus_gaussian':float(E1),
             'field_difference_norm':float(np.linalg.norm(after-before)),
             'material_quantity':float(chi.sum())}
        summaries.append(row)
        print(f"{material}, f={row['frequency_hz']:.6f} Hz : E_plan={E0:.12g}, E_plan+gaussienne={E1:.12g}, ||difference du champ||={row['field_difference_norm']:.6e}",flush=True)
        np.savez(out/(material+'.npz'),domain=d,f_initial=args[0],f_gaussian=changed[2][0],
                 gaussian=gaussian,f_dir=args[1],f_neu=args[2],f_rob=args[3],
                 chi=chi,u_plane=before,u_plane_plus_gaussian=after)
        fig,axes=plt.subplots(1,4,figsize=(13,4))
        for ax,values,title in zip(axes,[gaussian,np.abs(before),np.abs(after),np.abs(after-before)],
                ['Source gaussienne f','|u| : bord plan','|u| : bord plan + gaussienne','|difference de u|']):
            shown=np.where(d==_env.NODE_INTERIOR,values,np.nan)
            im=ax.imshow(shown,cmap='viridis',origin='upper')
            ax.set_title(title,fontsize=9)
            ax.set_xlabel('Indice x');ax.set_ylabel('Indice y')
            fig.colorbar(im,ax=ax,shrink=.7)
        fig.suptitle(f"{material}, {row['frequency_hz']:.3f} Hz, conditions aux bords identiques")
        fig.tight_layout();fig.savefig(out/(material+'.png'),dpi=160);plt.close(fig)
    np.testing.assert_array_equal(d,domain_snapshot)
    for old,current,gaussian_case in zip(boundary_snapshots,args[1:],changed[2][1:]):
        np.testing.assert_array_equal(old,current)
        np.testing.assert_array_equal(old,gaussian_case)
    report={'N':N,'level':level,'length_m':length_m,'sound_speed':sound_speed,
            'amplitude':amplitude,'center_normalized':list(center),'sigma_normalized':sigma,
            'boundary_conditions_unchanged':True,
            'experiment':'add Gaussian volume forcing to the existing plane boundary excitation; fixed chi, no optimization',
            'results':summaries}
    (out/'bilan_gaussienne.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Conditions aux bords inchangees : verifie. Resultats :',out.resolve(),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--N',type=int,default=50)
    parser.add_argument('--level',type=int,default=0)
    parser.add_argument('--frequency-index',type=int,default=0)
    parser.add_argument('--amplitude',type=float,default=1.)
    parser.add_argument('--sigma',type=float,default=.08)
    parser.add_argument('--center',type=float,nargs=2,default=[.5,.25])
    parser.add_argument('--length-m',type=float,default=1.)
    parser.add_argument('--c',type=float,default=340.)
    parser.add_argument('--output')
    args=parser.parse_args()
    run(args.N,args.level,args.frequency_index,args.amplitude,args.sigma,args.center,args.length_m,args.c,args.output)
