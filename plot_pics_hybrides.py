"""Plot sampled spectral peaks for the four hybrid-test geometries, at fixed densities."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import argparse
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.signal import find_peaks
import comparaison_materiaux as comp
import material_alpha
import preprocessing as pre


def energy_components(evaluator,chi,k,Alpha,main_mask):
    """Same reduced direct system as Evaluator, with two observation regions."""
    ratio=Alpha*chi[evaluator.a,evaluator.b]*evaluator.h/evaluator.beta
    denom=evaluator.betaR-ratio
    diagonal=k*k*evaluator.mass.copy()
    np.add.at(diagonal,evaluator.rows,ratio/denom)
    rhs=evaluator.b0.copy()
    np.add.at(rhs,evaluator.rows,(evaluator.h/denom-evaluator.h/evaluator.betaR)*evaluator.frob)
    u=splu(evaluator.A0+sparse.diags(diagonal,format='csc')).solve(rhs)
    main=u[main_mask.ravel()[evaluator.active]]
    return float(evaluator.h**2*np.vdot(u,u).real),float(evaluator.h**2*np.vdot(main,main).real)


def run(input_dir=None,output=None,source='plane',frequency_min=500.,frequency_max=900.,stride=1):
    root=Path(__file__).resolve().parent
    inp=Path(input_dir) if input_dir else root/'resultats_fractal_resonateur'
    out=Path(output) if output else root/'resultats_pics_hybrides'
    config=json.loads((inp/'configuration.json').read_text(encoding='utf-8'))
    N,L,c=config['N'],config['length_m'],config['sound_speed']
    if stride<1 or frequency_min>=frequency_max:
        raise ValueError('Positive stride and an increasing frequency range required.')
    tables={name:material_alpha.load_material(name) for name in material_alpha.MATERIALS}
    w=tables['ROCK_WOOL'][0]
    np.testing.assert_array_equal(w,tables['RECYCLED_RUBBER'][0])
    available=w/(2*np.pi)
    chosen=np.flatnonzero((available>=frequency_min-1e-9)&(available<=frequency_max+1e-9))
    if len(chosen)<3:
        raise ValueError('At least 3 available frequencies required.')
    selected=chosen[::stride]
    selected=np.unique(np.r_[selected,chosen[-1]])
    omega=w[selected]
    frequencies=omega/(2*np.pi)
    ks,_=material_alpha.solver_parameters(omega,np.ones(len(omega)),L,c)
    if ks.max()/N>2*np.pi/10:
        raise ValueError('Stored mesh is too coarse for this range. Rerun hybrid experiments on a finer mesh first.')
    out.mkdir(parents=True,exist_ok=True)
    variants=['plat','fractal','resonateur','fractal_resonateur']
    curves={}
    summaries=[]
    peak_rows=[]
    print(f'PICS : {len(selected)} frequences exactes de {frequencies[0]:.3f} a {frequencies[-1]:.3f} Hz ; source={source}.',flush=True)
    print('Densites avant/apres conservees a chaque frequence, sans nouvelle optimisation.',flush=True)
    for name in variants:
        for material in material_alpha.MATERIALS:
            with np.load(inp/f'{name}_{material}_{source}.npz') as data:
                d=data['domain'].copy();mask=data['main_mask'].copy()
                chi0,chi1=data['chi_initial'].copy(),data['chi_final'].copy()
                f,fd,fn,fr=[data[key].copy() for key in ['f','f_dir','f_neu','f_rob']]
            bp,ap,ad,bn,ar,br=pre._set_coefficients_of_pde(*d.shape)
            evaluator=comp.Evaluator((d,1/N,(f,fd,fn,fr,bp,ap,ad,bn,br),chi0))
            _,alphas=material_alpha.solver_parameters(omega,tables[material][1][selected],L,c)
            values=np.empty((len(ks),4))
            for i,(k,Alpha) in enumerate(zip(ks,alphas)):
                t0,m0=energy_components(evaluator,chi0,k,Alpha,mask)
                t1,m1=energy_components(evaluator,chi1,k,Alpha,mask)
                values[i]=t0,t1,m0,m1
                if i==0 or (i+1)%10==0 or i+1==len(ks):
                    print(f'{name} {material} : {i+1}/{len(ks)}, f={frequencies[i]:.3f} Hz, E_principale_finale={m1:.8g}',flush=True)
            curves[(name,material)]=values
            with (out/f'{name}_{material}.csv').open('w',newline='',encoding='utf-8') as file:
                writer=csv.writer(file)
                writer.writerow(['frequency_hz','total_initial','total_final','main_initial','main_final'])
                writer.writerows(np.c_[frequencies,values])
            for column,region,state in [(0,'total','initial'),(1,'total','final'),(2,'main','initial'),(3,'main','final')]:
                local=find_peaks(values[:,column])[0]
                maximum=int(np.argmax(values[:,column]))
                for i in np.unique(np.r_[local,maximum]):
                    peak_rows.append({'geometry':name,'material':material,'source':source,'region':region,'state':state,
                                      'frequency_hz':float(frequencies[i]),'energy':float(values[i,column]),
                                      'sampled_local_peak':bool(i in local),'sampled_global_maximum':bool(i==maximum)})
            i=int(np.argmax(values[:,3]))
            summaries.append({'geometry':name,'material':material,'source':source,
                              'main_peak_initial':float(values[:,2].max()),'main_peak_final':float(values[i,3]),
                              'main_peak_final_frequency_hz':float(frequencies[i]),
                              'main_mean_final':float(values[:,3].mean()),
                              'total_peak_final':float(values[:,1].max())})
    for name,records in [('pics.csv',peak_rows),('synthese_pics.csv',summaries)]:
        with (out/name).open('w',newline='',encoding='utf-8') as file:
            writer=csv.DictWriter(file,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    for region,columns in [('main',(2,3)),('total',(0,1))]:
        fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
        for ax,name in zip(axes.ravel(),variants):
            for material,color in zip(material_alpha.MATERIALS,['tab:blue','tab:orange']):
                data=curves[(name,material)]
                for column,style,state in [(columns[0],'--','initial'),(columns[1],'-','optimise')]:
                    energy=data[:,column]
                    ax.plot(frequencies,energy,style,color=color,label=material+' '+state)
                    indices=np.unique(np.r_[find_peaks(energy)[0],np.argmax(energy)]).astype(int)
                    ax.scatter(frequencies[indices],energy[indices],marker='o' if state=='initial' else '*',
                               color=color,s=15 if state=='initial' else 45,zorder=4)
            ax.set_title(name);ax.set_xlabel('Frequence (Hz)');ax.set_ylabel('Energie principale' if region=='main' else 'Energie totale')
            ax.grid(alpha=.25);ax.legend(fontsize=6)
        fig.suptitle(f'Pics echantillonnes : {frequencies[0]:.1f}-{frequencies[-1]:.1f} Hz, source {source}, N={N}\nPointilles : initial ; traits pleins : apres optimisation aux frequences cibles')
        fig.tight_layout();fig.savefig(out/f'pics_{region}.png',dpi=170);plt.close(fig)
    # Separate final-only view: large initial resonances must not hide residual peaks.
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,material in zip(axes,material_alpha.MATERIALS):
        for name in variants:
            energy=curves[(name,material)][:,3]
            line,=ax.plot(frequencies,energy,label=f'{name} (max={energy.max():.3g})')
            maxima=np.unique(np.r_[find_peaks(energy)[0],np.argmax(energy)]).astype(int)
            ax.scatter(frequencies[maxima],energy[maxima],color=line.get_color(),marker='*',s=40)
        ax.set(title=material,xlabel='Frequence (Hz)',ylabel='Energie principale apres optimisation')
        ax.grid(alpha=.25);ax.legend(fontsize=7)
    fig.suptitle(f'Comparaison des pics restants, {frequencies[0]:.1f}-{frequencies[-1]:.1f} Hz, N={N}')
    fig.tight_layout();fig.savefig(out/'pics_principaux_optimises.png',dpi=170);plt.close(fig)
    report={'input_directory':str(inp.resolve()),'source':source,'N':N,'stride':stride,
            'frequencies_hz':frequencies.tolist(),'initial_optimization_frequencies_hz':config['frequencies_hz'],
            'interpretation':'fixed initial/optimized densities; sampled local band only, peaks may be missed between samples',
            'results':summaries}
    (out/'bilan_pics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Pics traces et resultats :',out.resolve(),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir')
    p.add_argument('--output')
    p.add_argument('--source',choices=['plane','plane_plus_gaussian'],default='plane')
    p.add_argument('--frequency-min',type=float,default=500.)
    p.add_argument('--frequency-max',type=float,default=900.)
    p.add_argument('--stride',type=int,default=1)
    a=p.parse_args()
    run(a.input_dir,a.output,a.source,a.frequency_min,a.frequency_max,a.stride)
