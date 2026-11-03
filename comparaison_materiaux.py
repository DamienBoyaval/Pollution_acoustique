"""Compare both materials on a common frequency grid and with equal material quantity."""
import argparse
import csv
import json
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.signal import find_peaks
import _env
import preprocessing as pre
import processing as pro
import material_alpha
import demo_control_polycopie2024 as demo


def make_problem(N, level=0):
    M,h=2*N,1/N
    d,x,y,_,_=pre._set_geometry_of_domain(M,N,level)
    bp,ap,ad,bn,ar,br=pre._set_coefficients_of_pde(M,N)
    f,fd,fn,fr=pre._set_rhs_of_pde(M,N)
    # Professor's plane wave: constant phase/amplitude along the top boundary.
    # f=0 in the interior does not mean zero excitation.
    fd[0,:]=1.0
    chi=pre.set2zero(pre._set_chi(M,N,x,y),d)
    return d,h,(f,fd,fn,fr,bp,ap,ad,bn,br),chi


class Evaluator:
    """Same discrete equations; assemble fixed terms once and reuse one LU for u/p."""
    def __init__(self, problem):
        self.d,self.h,self.args,_=problem
        d,h=self.d,self.h
        f,fd,fn,fr,bp,ap,ad,bn,br=self.args
        A,b=pro.compute_stiffness_matrix(d,h,f,bp)
        A,b=pro.compute_dirichlet_condition(d,fd,ad,bp,A,b)
        A,b=pro.compute_neumann_condition(d,h,fn,bn,bp,A,b)
        A,b=pro.compute_robin_condition(d,h,fr,np.zeros_like(bp),br,bp,A,b)
        self.active=np.flatnonzero(d.ravel()==_env.NODE_INTERIOR)
        self.A0=A.tocsr()[self.active][:,self.active].tocsc()
        self.b0=b.ravel()[self.active]
        self.mass=h*h*ap.ravel()[self.active]
        lookup=np.full(d.shape,-1,dtype=int)
        lookup.ravel()[self.active]=np.arange(len(self.active))
        i,j=np.where(d==_env.NODE_INTERIOR)
        rows,aa,bb,ii,jj=[],[],[],[],[]
        for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            a,c=i+di,j+dj
            use=d[a,c]==_env.NODE_ROBIN
            rows.extend(lookup[i[use],j[use]])
            aa.extend(a[use]);bb.extend(c[use]);ii.extend(i[use]);jj.extend(j[use])
        self.rows=np.array(rows,dtype=int)
        self.a,self.b=np.array(aa,dtype=int),np.array(bb,dtype=int)
        self.beta=bp[np.array(ii,dtype=int),np.array(jj,dtype=int)]
        self.betaR=br[self.a,self.b]
        self.frob=fr[self.a,self.b]

    def evaluate(self, chi, ks, alphas, gradient=False):
        energies=[]
        grad=np.zeros_like(chi)
        if len(ks)==0 or len(ks)!=len(alphas):
            raise ValueError('Matching nonempty k/Alpha arrays required.')
        if np.max(ks)*self.h>2*np.pi/10:
            raise ValueError('Insufficient grid: need at least 10 points per wavelength.')
        for k,Alpha in zip(ks,alphas):
            ratio=Alpha*chi[self.a,self.b]*self.h/self.beta
            denom=self.betaR-ratio
            diagonal=k*k*self.mass.copy()
            np.add.at(diagonal,self.rows,ratio/denom)
            rhs=self.b0.copy()
            np.add.at(rhs,self.rows,(self.h/denom-self.h/self.betaR)*self.frob)
            lu=splu(self.A0+sparse.diags(diagonal,format='csc'))
            u=lu.solve(rhs)
            energies.append(float(self.h**2*np.vdot(u,u).real))
            if gradient:
                p=lu.solve(self.h**2*u,trans='H')
                dA=Alpha*self.h*self.betaR/(self.beta*denom**2)
                db=self.frob*self.h**2*Alpha/(self.beta*denom**2)
                contribution=2*np.real(np.conj(p[self.rows])*(db-dA*u[self.rows]))
                np.add.at(grad,(self.a,self.b),contribution/len(ks))
        return np.array(energies),grad


def spectrum(evaluator,chi,ks,alphas,label):
    values=[]
    for index,(k,Alpha) in enumerate(zip(ks,alphas)):
        energies,_=evaluator.evaluate(chi,[k],[Alpha])
        values.append(energies[0])
        if index==0 or (index+1)%25==0 or index+1==len(ks):
            print(f'{label} : {index+1}/{len(ks)}, energie={energies[0]:.8g}',flush=True)
    return np.array(values)


def peak_indices(energy,count):
    candidates=np.unique(np.r_[find_peaks(energy)[0],np.argmax(energy)]).astype(int)
    return candidates[np.argsort(energy[candidates])[-count:]]


def optimize(evaluator,chi0,ks,alphas,volume,max_iter=100,tolerance=1e-6,label=''):
    chi=demo.your_project_density(chi0,evaluator.d,volume)
    energies,grad=evaluator.evaluate(chi,ks,alphas,True)
    current=energies.mean()
    history=[current]
    mu,k,reason=5.,0,'limite d iterations'
    while k<max_iter and mu>1e-5:
        ene,accepted=current,False
        while ene>=current and mu>1e-5:
            trial=chi-mu*grad
            trial=demo.your_project_density(trial,evaluator.d,volume)
            tested,_=evaluator.evaluate(trial,ks,alphas)
            ene=float(tested.mean())
            if np.isfinite(ene) and ene<current:
                variation=np.linalg.norm(trial-chi)
                chi,accepted=trial,True
                mu=mu*1.1
            else:
                mu=mu/2
                ene=current
        if not accepted:
            reason='pas sous le seuil, aucun nouvel essai accepte'
            break
        k+=1
        history.append(ene)
        print(f'{label} iteration {k}: E_cibles={ene:.10g}, variation_chi={variation:.3e}, pas={mu:.6g}',flush=True)
        if variation<tolerance:
            reason='stabilisation de chi'
            break
        values,grad=evaluator.evaluate(chi,ks,alphas,True)
        current=float(values.mean())
    if k<max_iter and mu<=1e-5 and reason=='limite d iterations':
        reason='pas sous le seuil'
    return chi,np.array(history),reason


def run_comparison(N=None,level=0,length_m=1.,sound_speed=340.,max_iter=100,
                   number_peaks=3,indices=None,output=None):
    tables={name:material_alpha.load_material(name) for name in material_alpha.MATERIALS}
    omega=tables['ROCK_WOOL'][0]
    np.testing.assert_array_equal(omega,tables['RECYCLED_RUBBER'][0])
    selected=np.arange(len(omega)) if indices is None else np.asarray(indices,dtype=int)
    if len(selected)<3 or len(np.unique(selected))!=len(selected) or np.any(selected<0) or np.any(selected>=len(omega)):
        raise ValueError('Select >=3 distinct valid table indices.')
    selected=np.sort(selected)
    omega=omega[selected]
    frequencies=omega/(2*np.pi)
    ks,_=material_alpha.solver_parameters(omega,np.ones(len(omega)),length_m,sound_speed)
    needed=int(np.ceil(10*np.max(ks)/(2*np.pi)))
    multiple=max(16,4**level)
    N=int(np.ceil(max(needed,4)/multiple)*multiple) if N is None else N
    if N<needed or N%(4**level):
        raise ValueError(f'Use N >= {needed} and divisible by {4**level}.')
    if max_iter<0 or number_peaks<1:
        raise ValueError('Require max_iter >= 0 and number_peaks >= 1.')
    out=Path(output) if output else Path(__file__).resolve().parent/'resultats_comparaison'
    out.mkdir(parents=True,exist_ok=True)
    print(f'COMPARAISON: {len(omega)} pulsations, {frequencies[0]:.3f}-{frequencies[-1]:.3f} Hz; N={N}, niveau={level}, L={length_m} m, c={sound_speed} m/s',flush=True)
    print('Source: onde plane imposee en haut (f_dir=1); f=0 dans le volume.',flush=True)
    problem=make_problem(N,level)
    domain,h,_,chi0=problem
    evaluator=Evaluator(problem)
    mask=domain==_env.NODE_ROBIN
    volume=float(chi0.sum()/mask.sum())
    full=np.zeros_like(chi0);full[mask]=1
    results={}
    targets=[]
    # Initial density, mesh and source are identical for both materials.
    for material,(_,alphas,errors) in tables.items():
        _,a=material_alpha.solver_parameters(omega,alphas[selected],length_m,sound_speed)
        initial=spectrum(evaluator,chi0,ks,a,material+' initial')
        fully=spectrum(evaluator,full,ks,a,material+' tout absorbant (reference)')
        peaks=peak_indices(initial,number_peaks)
        targets.extend(peaks)
        results[material]={'alpha':a,'error':errors[selected],'initial':initial,'fully':fully}
    targets=np.unique(targets).astype(int)
    print('Frequences cibles COMMUNES (union des pics initiaux):',frequencies[targets],flush=True)
    summaries=[]
    for material,data in results.items():
        chi,history,reason=optimize(evaluator,chi0,ks[targets],data['alpha'][targets],volume,max_iter,label=material)
        final=spectrum(evaluator,chi,ks,data['alpha'],material+' apres optimisation')
        data['final']=final
        np.testing.assert_allclose(chi.sum(),chi0.sum(),rtol=0,atol=1e-8)
        initial_max,final_max=float(data['initial'].max()),float(final.max())
        row={'material':material,'energy_initial_max':initial_max,'energy_final_max':final_max,
             'frequency_initial_max_hz':float(frequencies[np.argmax(data['initial'])]),
             'frequency_final_max_hz':float(frequencies[np.argmax(final)]),
             'reduction_max_percent':100*(1-final_max/initial_max),
             'energy_initial_mean':float(data['initial'].mean()),'energy_final_mean':float(final.mean()),
             'target_initial_mean':float(history[0]),'target_final_mean':float(history[-1]),
             'material_quantity':float(chi.sum()),'accepted_iterations':len(history)-1,'stop_reason':reason}
        summaries.append(row)
        np.savez(out/(material+'.npz'),omega_rad_s=omega,frequency_hz=frequencies,k=ks,alpha_solver=data['alpha'],
                 alpha_fitting_error=data['error'],domain=domain,chi_initial=chi0,chi_final=chi,
                 energy_initial=data['initial'],energy_final=final,energy_fully_absorbing=data['fully'],
                 target_indices=targets,history=history)
        with (out/(material+'_spectre.csv')).open('w',newline='',encoding='utf-8') as file:
            writer=csv.writer(file)
            writer.writerow(['omega_rad_s','frequency_hz','initial','optimized','fully_absorbing_reference','initial_local_peak','final_local_peak'])
            p0,p1=set(find_peaks(data['initial'])[0]),set(find_peaks(final)[0])
            for i in range(len(omega)):
                writer.writerow([omega[i],frequencies[i],data['initial'][i],final[i],data['fully'][i],int(i in p0),int(i in p1)])
        print(f"{material}: max initial={initial_max:.9g}, max final={final_max:.9g}, moyenne finale={final.mean():.9g}, quantite={chi.sum():.9g}",flush=True)
    with (out/'comparaison.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=list(summaries[0]))
        writer.writeheader();writer.writerows(summaries)
    configuration={'N':N,'level':level,'length_m':length_m,'sound_speed':sound_speed,
                   'source':'plane wave on upper Dirichlet boundary; zero volume forcing',
                   'table_indices':selected.tolist(),'common_target_frequencies_hz':frequencies[targets].tolist(),
                   'volume_fraction':volume,'max_iter':max_iter,'chi_tolerance':1e-6,
                   'objective':'equal-weight mean energy at common initial peaks, not a minimax',
                   'commit':'817b56c73548e2a6f75d8b4021a228555eead5cd'}
    (out/'configuration.json').write_text(json.dumps(configuration,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for material,data in results.items():
        axes[0].plot(frequencies,data['initial'],label=material+' initial')
        axes[0].plot(frequencies,data['final'],label=material+' optimise')
        axes[1].plot(frequencies,data['final'],label=material)
    for ax in axes:
        ax.set(xlabel='Frequence (Hz)',ylabel='Energie discrete normalisee')
        ax.grid(alpha=.25);ax.legend(fontsize=8)
    axes[0].set_title('Avant / apres, meme quantite')
    axes[1].set_title('Pics restants apres optimisation')
    fig.tight_layout();fig.savefig(out/'comparaison_energies.png',dpi=160);plt.close(fig)
    ranked=sorted(summaries,key=lambda row:row['energy_final_max'])
    print('\nCLASSEMENT SUR LE MAXIMUM ECHANTILLONNE APRES OPTIMISATION:',flush=True)
    for rank,row in enumerate(ranked,1):
        print(f"{rank}. {row['material']}: {row['energy_final_max']:.10g} a {row['frequency_final_max_hz']:.3f} Hz",flush=True)
    print('Classement limite a cette geometrie, cette quantite, cette source et cette grille de frequences.',flush=True)
    print('Resultats enregistres dans :',out.resolve(),flush=True)
    return summaries,configuration


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--N',type=int)
    parser.add_argument('--level',type=int,default=0)
    parser.add_argument('--length-m',type=float,default=1.)
    parser.add_argument('--c',type=float,default=340.)
    parser.add_argument('--max-iter',type=int,default=100)
    parser.add_argument('--number-peaks',type=int,default=3)
    parser.add_argument('--indices',type=int,nargs='+')
    parser.add_argument('--output')
    args=parser.parse_args()
    run_comparison(args.N,args.level,args.length_m,args.c,args.max_iter,args.number_peaks,args.indices,args.output)
