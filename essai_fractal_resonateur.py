"""Controlled comparison: flat, fractal, resonator, and fractal plus resonator."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import argparse
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import geometrie_hybride as geo
import comparaison_materiaux as comp
import essai_source_gaussienne as gaussian
import material_alpha
import processing
import _env


def spatial_metrics(problem,chi,ks,alphas,main_mask):
    d,h,args,_=problem
    total,main,fields=[],[],[]
    for k,Alpha in zip(ks,alphas):
        u=processing.solve_helmholtz(d,h,k,*args,Alpha*chi)
        total.append(h*h*np.sum(np.abs(u[d==_env.NODE_INTERIOR])**2))
        main.append(h*h*np.sum(np.abs(u[main_mask])**2))
        fields.append(u)
    return np.array(total),np.array(main),np.array(fields)


def run(N=128,level=1,indices=None,quantity=.4,length_m=1.,sound_speed=340.,
        max_iter=100,amplitude=1.,sigma=.08,center=(.5,.25),output=None):
    if level<1:
        raise ValueError('Choose a fractal level >=1 for the mixed case.')
    tables={name:material_alpha.load_material(name) for name in material_alpha.MATERIALS}
    omega_all=tables['ROCK_WOOL'][0]
    np.testing.assert_array_equal(omega_all,tables['RECYCLED_RUBBER'][0])
    variants=[('plat',0,False),('fractal',level,False),('resonateur',0,True),('fractal_resonateur',level,True)]
    problems={name:geo.build_problem(N,l,res,quantity,length_m,sound_speed) for name,l,res in variants}
    estimate=problems['resonateur'][2]['helmholtz_estimate_hz_no_end_correction']
    if indices is None:
        if not omega_all[0]/(2*np.pi)<=estimate<=omega_all[-1]/(2*np.pi):
            raise ValueError('Estimated frequency is outside the Alpha tables. Adapt dimensions or select explicit available indices.')
        indices=np.unique([0,int(np.argmin(np.abs(omega_all/(2*np.pi)-estimate)))])
    indices=np.asarray(indices,dtype=int)
    if not len(indices) or np.any(indices<0) or np.any(indices>=len(omega_all)) or len(np.unique(indices))!=len(indices):
        raise ValueError('Distinct valid frequency indices required.')
    indices=np.sort(indices)
    omega=omega_all[indices]
    frequencies=omega/(2*np.pi)
    ks,_=material_alpha.solver_parameters(omega,np.ones(len(omega)),length_m,sound_speed)
    if ks.max()/N>2*np.pi/10:
        raise ValueError('Increase N for at least 10 points per wavelength.')
    if max_iter<0:
        raise ValueError('max_iter must be nonnegative.')
    out=Path(output) if output else Path(__file__).resolve().parent/'resultats_fractal_resonateur'
    out.mkdir(parents=True,exist_ok=True)
    print(f'MIX FRACTAL + RESONATEUR: N={N}, niveau={level}, quantite={quantity:g}',flush=True)
    print(f'Estimation cavite-col sans correction des extremites: {estimate:.3f} Hz (pas une resonance couplee mesuree).',flush=True)
    print('Frequences exactes des tables utilisees:',frequencies,flush=True)
    print('Optimisation de l energie TOTALE; energie de la zone principale mesuree separement.',flush=True)
    rows=[]
    spectra=[]
    configuration={'N':N,'fractal_level':level,'quantity':quantity,'length_m':length_m,'sound_speed':sound_speed,
                   'table_indices':indices.tolist(),'frequencies_hz':frequencies.tolist(),'max_iter':max_iter,
                   'gaussian_amplitude':amplitude,'gaussian_sigma':sigma,'gaussian_center':list(center),
                   'objective':'mean total-domain energy at the selected frequencies',
                   'cavity_walls':'rigid Neumann; absorption only on original Robin boundary',
                   'geometries':{name:data[2] for name,data in problems.items()}}
    (out/'configuration.json').write_text(json.dumps(configuration,indent=2),encoding='utf-8')
    plot_geometries(problems,out)
    for name,_,_ in variants:
        base,main_mask,metadata=problems[name]
        d,h,args,chi0=base
        fraction=quantity/(h*length_m*np.count_nonzero(d==_env.NODE_ROBIN))
        with_gaussian,g=gaussian.add_gaussian(base,amplitude,center,sigma)
        for source,problem in [('plane',base),('plane_plus_gaussian',with_gaussian)]:
            evaluator=comp.Evaluator(problem)
            for material in material_alpha.MATERIALS:
                _,alphas=material_alpha.solver_parameters(omega,tables[material][1][indices],length_m,sound_speed)
                initial,_=evaluator.evaluate(chi0,ks,alphas)
                final_chi,history,reason=comp.optimize(evaluator,chi0,ks,alphas,fraction,
                                       max_iter=max_iter,label=f'{name} {material} {source}')
                final,_=evaluator.evaluate(final_chi,ks,alphas)
                check0,main0,u0=spatial_metrics(problem,chi0,ks,alphas,main_mask)
                check1,main1,u1=spatial_metrics(problem,final_chi,ks,alphas,main_mask)
                np.testing.assert_allclose(check0,initial,rtol=1e-9,atol=1e-10)
                np.testing.assert_allclose(check1,final,rtol=1e-9,atol=1e-10)
                np.testing.assert_allclose(h*length_m*final_chi.sum(),quantity,atol=1e-10,rtol=0)
                row={'geometry':name,'material':material,'source':source,'N':N,
                     'energy_total_initial':float(initial.mean()),'energy_total_final':float(final.mean()),
                     'energy_main_initial':float(main0.mean()),'energy_main_final':float(main1.mean()),
                     'energy_added_air_final':float((final-main1).mean()),
                     'energy_total_final_max':float(final.max()),'energy_main_final_max':float(main1.max()),
                     'main_area':metadata['main_area_normalized'],'total_area':metadata['total_area_normalized'],
                     'energy_main_final_per_area':float(main1.mean()/metadata['main_area_normalized']),
                     'material_quantity':float(h*length_m*final_chi.sum()),'accepted_iterations':len(history)-1,
                     'stop_reason':reason}
                rows.append(row)
                for i,f in enumerate(frequencies):
                    spectra.append([name,material,source,float(f),float(initial[i]),float(final[i]),float(main0[i]),float(main1[i])])
                np.savez(out/(f'{name}_{material}_{source}.npz'),domain=d,main_mask=main_mask,chi_initial=chi0,
                         chi_final=final_chi,omega_rad_s=omega,alpha_solver=alphas,k_solver=ks,
                         energy_total_initial=initial,energy_total_final=final,energy_main_initial=main0,
                         energy_main_final=main1,u_initial=u0,u_final=u1,history=history,f=problem[2][0],
                         f_dir=args[1],f_neu=args[2],f_rob=args[3])
                print(f"RESULTAT {name} {material} {source}: E_total={initial.mean():.8g}->{final.mean():.8g}; E_zone_principale={main0.mean():.8g}->{main1.mean():.8g}",flush=True)
                write_csv(out/'comparaison_hybride.csv',rows)
    with (out/'energies_par_frequence.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.writer(file)
        writer.writerow(['geometry','material','source','frequency_hz','total_initial','total_final','main_initial','main_final'])
        writer.writerows(spectra)
    configuration['results']=rows
    (out/'bilan_hybride.json').write_text(json.dumps(configuration,indent=2),encoding='utf-8')
    plot_results(rows,[v[0] for v in variants],out)
    print('Resultats :',out.resolve(),flush=True)
    return configuration


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def plot_geometries(problems,out):
    fig,axes=plt.subplots(1,4,figsize=(14,4))
    for ax,(name,(q,main,meta)) in zip(axes,problems.items()):
        d,h,_,_=q
        interior=d==_env.NODE_INTERIOR
        shown=np.where(interior,1.,np.nan)
        extent=[0,(d.shape[1]-1)*h,(d.shape[0]-1)*h,0]
        ax.imshow(shown,extent=extent,cmap='Greys',vmin=0,vmax=3,origin='upper')
        for code,color,label in [(_env.NODE_ROBIN,'tab:red','Robin'),(_env.NODE_DIRICHLET,'tab:blue','Onde plane')]:
            i,j=np.where(d==code);ax.scatter(j*h,i*h,s=2,color=color,label=label)
        bottom=(np.max(np.where(interior)[0])+3)*h
        ax.set(xlim=(-.02,extent[1]+.02),ylim=(bottom,-.02),title=name,aspect='equal')
        ax.set_xlabel('x/L');ax.set_ylabel('y/L');ax.legend(fontsize=6)
        if meta['resonator']:
            inset=ax.inset_axes([.48,.48,.50,.30])
            inset.imshow(shown,extent=extent,cmap='Greys',vmin=0,vmax=3,origin='upper')
            x=meta['mouth_x_normalized'];y=meta['center_y_normalized']
            inset.set_xlim(x-.03,x+meta['neck_length_normalized']+meta['cavity_width_normalized']+.02)
            inset.set_ylim(y+.06,y-.06);inset.set_title('Zoom col + cavite',fontsize=6)
            inset.tick_params(labelsize=5)
    fig.tight_layout();fig.savefig(out/'geometries_hybrides.png',dpi=170);plt.close(fig)


def plot_results(rows,names,out):
    for region in ['total','main']:
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for ax,source in zip(axes,['plane','plane_plus_gaussian']):
            for k,(material,color) in enumerate(zip(material_alpha.MATERIALS,['tab:blue','tab:orange'])):
                selected=[next(row for row in rows if row['geometry']==name and row['material']==material and row['source']==source) for name in names]
                x=np.arange(len(names))+(k-.5)*.32
                ax.bar(x,[r[f'energy_{region}_final'] for r in selected],width=.3,color=color,label=material+' final')
                ax.scatter(x,[r[f'energy_{region}_initial'] for r in selected],marker='x',color='black',s=20,label='Initial' if k==0 else None)
            ax.set_xticks(range(len(names)),['Plat','Fractal','Resonateur','Mix'],rotation=15)
            ax.set_ylabel('Energie moyenne (frequences selectionnees)')
            ax.set_title('Onde plane' if source=='plane' else 'Onde plane + gaussienne')
            ax.grid(axis='y',alpha=.2);ax.legend(fontsize=7)
        fig.suptitle('Domaine complet' if region=='total' else 'Zone principale seulement (hors col et cavite)')
        fig.tight_layout();fig.savefig(out/(f'comparaison_energie_{region}.png'),dpi=160);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--N',type=int,default=128)
    parser.add_argument('--level',type=int,default=1)
    parser.add_argument('--indices',type=int,nargs='+')
    parser.add_argument('--quantity',type=float,default=.4)
    parser.add_argument('--length-m',type=float,default=1.)
    parser.add_argument('--c',type=float,default=340.)
    parser.add_argument('--max-iter',type=int,default=100)
    parser.add_argument('--amplitude',type=float,default=1.)
    parser.add_argument('--sigma',type=float,default=.08)
    parser.add_argument('--center',type=float,nargs=2,default=[.5,.25])
    parser.add_argument('--output')
    parser.add_argument('--no-spectral-plot',action='store_true',help='Skip the additional 500-900 Hz spectral diagnostic.')
    a=parser.parse_args()
    run(a.N,a.level,a.indices,a.quantity,a.length_m,a.c,a.max_iter,a.amplitude,a.sigma,a.center,a.output)
    if not a.no_spectral_plot and a.max_iter>0:
        from plot_pics_hybrides import run as plot_peaks
        directory=Path(a.output) if a.output else Path(__file__).resolve().parent/'resultats_fractal_resonateur'
        plot_peaks(input_dir=directory,output=directory/'pics_spectraux')
