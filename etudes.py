"""Frequency, shape, volume and simultaneous-frequency studies.

Default: dimensionless professor example, NOT a calibrated porous material.
python etudes.py --help
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import _env
import preprocessing as pre
import processing as pro
import demo_control_polycopie2024 as demo
import material_alpha


def make_problem(N, level, source='plane'):
    M, h = 2*N, 1/N
    d, x, y, _, _ = pre._set_geometry_of_domain(M, N, level)
    bp, ap, ad, bn, ar, br = pre._set_coefficients_of_pde(M, N)
    f, fd, fn, fr = pre._set_rhs_of_pde(M, N)
    if source == 'plane':
        fd[0, :] = 1
    elif source == 'point':
        fd[0, N//2] = 10
    else:
        raise ValueError('Unknown source.')
    chi = pre.set2zero(pre._set_chi(M, N, x, y), d)
    return d, h, (f, fd, fn, fr, bp, ap, ad, bn, br), chi


def evaluate(problem, chi, wavenumbers, alphas, weights=None, gradient=False):
    d, h, args, _ = problem
    ks, alphas = np.atleast_1d(wavenumbers), np.atleast_1d(alphas)
    if len(ks) != len(alphas) or len(ks) == 0:
        raise ValueError('One Alpha is required for each wavenumber.')
    if not np.all(np.isfinite(ks)) or np.any(ks <= 0) or np.max(ks)*h > 2*np.pi/10:
        raise ValueError('Require positive k and at least 10 grid intervals per wavelength.')
    if not np.all(np.isfinite(alphas)):
        raise ValueError('Non-finite Alpha.')
    weights = np.ones(len(ks))/len(ks) if weights is None else np.asarray(weights, float)
    if weights.shape != ks.shape or np.any(weights < 0) or not np.all(np.isfinite(weights)) or weights.sum() <= 0:
        raise ValueError('Invalid frequency weights.')
    weights = weights/weights.sum()
    energies, grad = [], np.zeros_like(chi)
    for k, Alpha, weight in zip(ks, alphas, weights):
        u, mat, rhs = pro.solve_helmholtz(d,h,k,*args,Alpha*chi,return_system=True)
        energies.append(demo.your_compute_objective_function(d,u,h))
        if gradient:
            p = demo.your_compute_adjoint(d,u,h,mat)
            grad += weight*demo.your_compute_gradient(d,u,p,h,args[3],args[4],args[-1],Alpha,chi)
    return float(np.dot(weights,energies)), np.array(energies), grad


def optimize_frequencies(problem, chi, ks, alphas, volume, mu=5., max_iter=100, weights=None):
    # Same outer/inner while loops and step updates as the professor's skeleton.
    d = problem[0]
    chi = demo.your_project_density(chi,d,volume)
    history, k = [], 0
    current, _, grad = evaluate(problem,chi,ks,alphas,weights,gradient=True)
    history.append(current)
    while k < max_iter and mu > 1e-5:
        ene, accepted = current, False
        while ene >= current and mu > 1e-5:
            trial = chi-mu*grad
            trial = demo.your_project_density(trial,d,volume)
            ene, _, _ = evaluate(problem,trial,ks,alphas,weights)
            if np.isfinite(ene) and ene < current:
                chi, accepted = trial, True
                mu = mu*1.1
            else:
                mu = mu/2
                ene = current
        if not accepted:
            break
        k += 1
        current, _, grad = evaluate(problem,chi,ks,alphas,weights,gradient=True)
        history.append(current)
    return chi, np.array(history), grad


def material_coefficients(path, frequencies, length):
    # Supply coefficients already derived/calibrated in the professor's sign convention.
    # CSV columns: frequency_hz,alpha_real_per_m,alpha_imag_per_m
    table = np.genfromtxt(path, delimiter=',', names=True)
    table = np.atleast_1d(table)
    f = table['frequency_hz']
    re, im = table['alpha_real_per_m'], table['alpha_imag_per_m']
    if len(f) < 2 or not np.all(np.isfinite([f,re,im])) or np.any(np.diff(f) <= 0):
        raise ValueError('Material table requires >=2 finite rows in strictly increasing frequency order.')
    if min(frequencies) < f[0] or max(frequencies) > f[-1]:
        raise ValueError('Material table does not cover the requested range; extrapolation is disabled.')
    return length*(np.interp(frequencies,f,re)+1j*np.interp(frequencies,f,im))


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    if args.material:
        if args.alpha_csv is not None or args.length_m is None:
            raise ValueError('--material requires --length-m and cannot be combined with --alpha-csv.')
        omega_table, alpha_table, errors = material_alpha.load_material(args.material)
        frequencies = omega_table/(2*np.pi)
        selected = np.ones(len(frequencies), dtype=bool)
        if args.hz:
            selected = (frequencies >= args.hz[0]-1e-9) & (frequencies <= args.hz[1]+1e-9)
        omega_table, alpha_table = omega_table[selected], alpha_table[selected]
        frequencies, errors = frequencies[selected], errors[selected]
        ks, alphas = material_alpha.solver_parameters(omega_table, alpha_table, args.length_m, args.c)
        abscissa, xlabel = frequencies, 'Frequency (Hz)'
        np.savez(out/'material_used.npz',omega_rad_s=omega_table,frequency_hz=frequencies,
                 alpha_supplied=alpha_table,alpha_solver=alphas,k_solver=ks,fitting_error=errors)
        print('Material %s: %d original frequency entries, no interpolation.'%(args.material,len(ks)),flush=True)
    elif args.hz:
        if args.alpha_csv is None or args.length_m is None:
            raise ValueError('--hz requires --length-m and --alpha-csv (no invented material coefficient).')
        if args.length_m <= 0 or args.c <= 0:
            raise ValueError('Positive length and sound speed required.')
        frequencies = np.linspace(*args.hz,args.samples)
        ks = 2*np.pi*frequencies*args.length_m/args.c
        alphas = material_coefficients(args.alpha_csv,frequencies,args.length_m)
        abscissa, xlabel = frequencies, 'Frequency (Hz)'
    else:
        ks = np.linspace(args.k_min,args.k_max,args.samples)
        alphas = np.full(ks.shape,10-10j)
        abscissa, xlabel = ks, 'Dimensionless wavenumber k (example Alpha = 10 - 10j)'
    if len(ks) < 3 or np.any(np.diff(ks) <= 0):
        raise ValueError('Use >=3 increasing frequencies.')
    records, summaries = [], []
    for level in args.levels:
        problem = make_problem(args.N,level,args.source)
        d,h,_,chi0 = problem
        mask = d == _env.NODE_ROBIN
        # Sampling guard is enforced before the expensive sweep.
        if ks[-1]*h > 2*np.pi/10:
            needed = int(np.ceil(10*ks[-1]/(2*np.pi)))
            raise ValueError('Grid too coarse: use N >= %d, divisible by 4**level.' % needed)
        full = np.zeros_like(chi0)
        full[mask] = 1
        _, Efull, _ = evaluate(problem,full,ks,alphas)
        _, Einitial, _ = evaluate(problem,chi0,ks,alphas)
        peaks = find_peaks(Efull)[0]
        # Always include the largest sampled energy, including an endpoint.
        targets = np.unique(np.r_[peaks,np.argmax(Efull)]).astype(int)
        targets = targets[np.argsort(Efull[targets])[-3:]]
        volume0 = chi0.sum()/mask.sum()
        single_index = targets[np.argmax(Efull[targets])]
        single, hist1, _ = optimize_frequencies(problem,chi0,ks[[single_index]],alphas[[single_index]],volume0,max_iter=args.max_iter)
        multi, histm, _ = optimize_frequencies(problem,chi0,ks[targets],alphas[targets],volume0,max_iter=args.max_iter)
        _, Esingle, _ = evaluate(problem,single,ks,alphas)
        _, Emulti, _ = evaluate(problem,multi,ks,alphas)
        for i in range(len(ks)):
            records.append([level,float(abscissa[i]),ks[i],Efull[i],Einitial[i],Esingle[i],Emulti[i],int(i in peaks)])
        plt.figure()
        for curve,label in [(Efull,'Fully absorbing'),(Einitial,'Initial'),(Esingle,'One frequency'),(Emulti,'Several frequencies')]:
            plt.plot(abscissa,curve,label=label)
        plt.xlabel(xlabel); plt.ylabel('Discrete acoustic energy'); plt.legend(); plt.tight_layout()
        plt.savefig(out/('energy_level_%d.png'%level)); plt.close()
        np.savez(out/('fields_level_%d.npz'%level),domain=d,chi_initial=chi0,chi_single=single,chi_multi=multi,history_single=hist1,history_multi=histm,ks=ks,alphas=alphas,target_indices=targets)
        for volume in args.volumes:
            density, history, _ = optimize_frequencies(problem,chi0,ks[targets],alphas[targets],volume,max_iter=args.max_iter)
            _, spectrum, _ = evaluate(problem,density,ks,alphas)
            summaries.append([level,volume,history[0],history[-1],spectrum.mean(),spectrum.max(),len(history)-1])
            np.save(out/('chi_level_%d_volume_%g.npy'%(level,volume)),density)
        print('level=%d completed; targets=%s; initial volume=%g'%(level,abscissa[targets],volume0),flush=True)
    for name,header,rows in [
        ('spectra.csv',['level','frequency_hz_or_k','k','fully_absorbing','initial','optimized_single','optimized_multi','sampled_local_maximum'],records),
        ('volumes.csv',['level','volume_fraction','target_initial_energy','target_final_energy','mean_sampled_energy','max_sampled_energy','accepted_iterations'],summaries)]:
        with (out/name).open('w',newline='',encoding='utf-8') as file:
            writer=csv.writer(file);writer.writerow(header);writer.writerows(rows)
    (out/'configuration.json').write_text(json.dumps(vars(args),indent=2),encoding='utf-8')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--N',type=int,default=16)
    parser.add_argument('--levels',type=int,nargs='+',default=[0,1])
    parser.add_argument('--k-min',type=float,default=1)
    parser.add_argument('--k-max',type=float,default=9)
    parser.add_argument('--samples',type=int,default=25)
    parser.add_argument('--hz',type=float,nargs=2,metavar=('MIN','MAX'))
    parser.add_argument('--length-m',type=float)
    parser.add_argument('--c',type=float,default=340)
    parser.add_argument('--alpha-csv')
    parser.add_argument('--material',choices=material_alpha.MATERIALS)
    parser.add_argument('--source',choices=['plane','point'],default='plane')
    parser.add_argument('--volumes',type=float,nargs='+',default=[.25,.4,.6,1])
    parser.add_argument('--max-iter',type=int,default=100)
    parser.add_argument('--output',default='resultats_etudes')
    run(parser.parse_args())
