"""Compare professor fractal levels, both materials and both source cases at fixed quantity."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import argparse
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import comparaison_materiaux as comp
import essai_source_gaussienne as gaussian_test
import material_alpha
import demo_control_polycopie2024 as demo
import _env


def run(levels=(0,1,2),N=None,indices=(0,),quantity=.4,length_m=1.,sound_speed=340.,
        max_iter=100,amplitude=1.,sigma=.08,center=(.5,.25),output=None):
    """quantity = L*h*sum(chi), the same discrete material length for every level."""
    levels=tuple(sorted(set(levels)))
    if not levels or any(int(l)!=l or l<0 for l in levels):
        raise ValueError('Nonnegative integer fractal levels required.')
    indices=np.asarray(indices,dtype=int)
    tables={name:material_alpha.load_material(name) for name in material_alpha.MATERIALS}
    omega_all=tables['ROCK_WOOL'][0]
    np.testing.assert_array_equal(omega_all,tables['RECYCLED_RUBBER'][0])
    if not len(indices) or np.any(indices<0) or np.any(indices>=len(omega_all)) or len(np.unique(indices))!=len(indices):
        raise ValueError('Choose distinct valid table indices.')
    indices=np.sort(indices)
    omega=omega_all[indices]
    ks,_=material_alpha.solver_parameters(omega,np.ones(len(omega)),length_m,sound_speed)
    frequencies=omega/(2*np.pi)
    multiple=4**max(levels)
    required=max(4,int(np.ceil(10*max(ks)/(2*np.pi))))
    N=int(np.ceil(max(required,4*multiple)/multiple)*multiple) if N is None else N
    if N<required or N%multiple:
        raise ValueError(f'Use N >= {required} and divisible by {multiple}.')
    if quantity<=0 or not np.isfinite(quantity) or max_iter<0:
        raise ValueError('Positive finite quantity and max_iter >= 0 required.')
    out=Path(output) if output else Path(__file__).resolve().parent/'resultats_niveaux_fractals'
    out.mkdir(parents=True,exist_ok=True)
    print(f'NIVEAUX {levels}, N={N}, frequences={frequencies}, quantite discrete commune={quantity:g}',flush=True)
    print('Source 1: onde plane au bord. Source 2: memes bords + gaussienne dans le volume.',flush=True)
    rows=[]
    spectral=[]
    domains=[]
    for level in levels:
        d,h,args,raw_chi=comp.make_problem(N,level)
        mask=d==_env.NODE_ROBIN
        target_sum=quantity/(h*length_m)
        fraction=target_sum/mask.sum()
        if fraction>1:
            raise ValueError('Requested quantity exceeds available Robin boundary.')
        # Reuse the professor's initial pattern, projected to a common quantity.
        chi0=demo.your_project_density(raw_chi,d,fraction)
        base=(d,h,args,chi0)
        with_gaussian,g=gaussian_test.add_gaussian(base,amplitude,center,sigma)
        snapshots=[a.copy() for a in args[1:]]
        area=float(h*h*np.count_nonzero(d==_env.NODE_INTERIOR))
        source_integral=float(h*h*g.sum())
        domains.append(d.copy())
        for source,problem in [('plane',base),('plane_plus_gaussian',with_gaussian)]:
            evaluator=comp.Evaluator(problem)
            for material in material_alpha.MATERIALS:
                alphas=tables[material][1][indices]
                _,alphas=material_alpha.solver_parameters(omega,alphas,length_m,sound_speed)
                initial,_=evaluator.evaluate(chi0,ks,alphas)
                final_chi,history,reason=comp.optimize(evaluator,chi0,ks,alphas,fraction,
                                      max_iter=max_iter,label=f'niveau {level} {material} {source}')
                final,_=evaluator.evaluate(final_chi,ks,alphas)
                actual_quantity=float(h*length_m*final_chi.sum())
                np.testing.assert_allclose(actual_quantity,quantity,rtol=0,atol=1e-10)
                row={'level':level,'material':material,'source':source,'N':N,
                     'area_normalized':area,'robin_nodes':int(mask.sum()),
                     'material_quantity_discrete':actual_quantity,'boundary_fraction':float(fraction),
                     'gaussian_volume_integral':source_integral if source!='plane' else 0.,
                     'energy_initial_mean':float(initial.mean()),'energy_final_mean':float(final.mean()),
                     'energy_final_mean_per_area':float(final.mean()/area),
                     'energy_initial_max':float(initial.max()),'energy_final_max':float(final.max()),
                     'reduction_mean_percent':float(100*(1-final.mean()/initial.mean())),
                     'accepted_iterations':len(history)-1,'stop_reason':reason}
                rows.append(row)
                for i,freq in enumerate(frequencies):
                    spectral.append([level,material,source,float(freq),float(initial[i]),float(final[i]),float(final[i]/area)])
                np.savez(out/(f'level_{level}_{material}_{source}.npz'),domain=d,chi_initial=chi0,
                         chi_final=final_chi,omega_rad_s=omega,frequency_hz=frequencies,k_solver=ks,
                         alpha_solver=alphas,energy_initial=initial,energy_final=final,
                         history=history,f=problem[2][0],f_dir=args[1],f_neu=args[2],f_rob=args[3])
                print(f"RESULTAT niveau={level} {material} {source}: E={initial.mean():.9g} -> {final.mean():.9g}; baisse={row['reduction_mean_percent']:.3f}%; quantite={actual_quantity:.9g}",flush=True)
        for expected,actual,gauss_value in zip(snapshots,args[1:],with_gaussian[2][1:]):
            np.testing.assert_array_equal(expected,actual)
            np.testing.assert_array_equal(expected,gauss_value)
        # Checkpoint after each completed level.
        with (out/'comparaison_niveaux.csv').open('w',newline='',encoding='utf-8') as file:
            writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (out/'energies_par_frequence.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.writer(file)
        writer.writerow(['level','material','source','frequency_hz','energy_initial','energy_final','energy_final_per_area'])
        writer.writerows(spectral)
    configuration={'levels':list(levels),'N':N,'table_indices':indices.tolist(),'frequency_hz':frequencies.tolist(),
                   'quantity_definition':'quantity = L*h*sum(chi), same for all levels',
                   'quantity':quantity,'length_m':length_m,'sound_speed':sound_speed,
                   'gaussian_amplitude':amplitude,'gaussian_sigma_normalized':sigma,
                   'gaussian_center_normalized':list(center),'max_iter':max_iter,
                   'chi_tolerance':1e-6,'same_boundary_data_between_source_cases':True,
                   'objective':'mean energy over exactly the selected frequencies; no full-spectrum claim',
                   'geometry':'professor preprocessing._set_geometry_of_domain, unmodified motif',
                   'results':rows}
    (out/'bilan_niveaux.json').write_text(json.dumps(configuration,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,source,title in zip(axes,['plane','plane_plus_gaussian'],['Onde plane au bord','Bord plan + gaussienne dans le volume']):
        for material,color in zip(material_alpha.MATERIALS,['tab:blue','tab:orange']):
            selected=[r for r in rows if r['material']==material and r['source']==source]
            ax.plot(levels,[r['energy_initial_mean'] for r in selected],'--o',color=color,label=material+' initial')
            ax.plot(levels,[r['energy_final_mean'] for r in selected],'-s',color=color,label=material+' optimise')
        ax.set(xlabel='Niveau de fractale',ylabel='Energie moyenne sur les frequences choisies',title=title)
        ax.set_xticks(levels);ax.grid(alpha=.25);ax.legend(fontsize=7)
    fig.suptitle(f'Meme quantite = {quantity:g}, N={N}, f={frequencies[0]:.1f} a {frequencies[-1]:.1f} Hz')
    fig.tight_layout();fig.savefig(out/'comparaison_niveaux.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,len(levels),figsize=(4*len(levels),4),squeeze=False)
    for ax,level,d in zip(axes[0],levels,domains):
        inner=d==_env.NODE_INTERIOR
        ax.imshow(np.where(inner,1.,np.nan),cmap='Greys',vmin=0,vmax=3,origin='upper')
        i,j=np.where(d==_env.NODE_ROBIN)
        ax.scatter(j,i,s=3,color='tab:red',label='Robin')
        i,j=np.where(d==_env.NODE_DIRICHLET)
        ax.scatter(j,i,s=3,color='tab:blue',label='Onde plane')
        bottom=np.max(np.where(inner)[0])+3
        ax.set_ylim(bottom,-1);ax.set_aspect('equal');ax.set_title(f'Niveau {level}');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'geometries.png',dpi=160);plt.close(fig)
    print('Resultats :',out.resolve(),flush=True)
    return configuration


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--levels',type=int,nargs='+',default=[0,1,2])
    parser.add_argument('--N',type=int)
    parser.add_argument('--indices',type=int,nargs='+',default=[0])
    parser.add_argument('--quantity',type=float,default=.4)
    parser.add_argument('--length-m',type=float,default=1.)
    parser.add_argument('--c',type=float,default=340.)
    parser.add_argument('--max-iter',type=int,default=100)
    parser.add_argument('--amplitude',type=float,default=1.)
    parser.add_argument('--sigma',type=float,default=.08)
    parser.add_argument('--center',type=float,nargs=2,default=[.5,.25])
    parser.add_argument('--output')
    a=parser.parse_args()
    run(a.levels,a.N,a.indices,a.quantity,a.length_m,a.c,a.max_iter,a.amplitude,a.sigma,a.center,a.output)
