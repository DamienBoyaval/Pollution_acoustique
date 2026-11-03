"""Experience : energie initiale / optimisee binaire pour la source train.

Lancer : python test_energie_frequence.py
Chaque frequence a sa propre optimisation, repartant du meme chi initial.
La courbe optimisee n'est donc pas le spectre d'un mur unique fixe.
"""
import argparse
import contextlib
import csv
import json
from pathlib import Path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import _env
import preprocessing as pre
import processing as pro
import material_alpha
import train_source
import demo_control_polycopie2024 as demo


def run(fmin=500., fmax=1000., points=100, N=None, max_iter=100,
        material='ROCK_WOOL', level=0, fraction=.3, speed=300.,
        threshold=.5, output='resultats_energie_frequence_train'):
    omega, alpha, _ = material_alpha.load_material(material)
    frequencies = omega / (2 * np.pi)
    if not (np.isfinite(fmin) and np.isfinite(fmax) and
            frequencies[0]-1e-8 <= fmin < fmax <= frequencies[-1]+1e-8):
        raise ValueError('Choisir 500 <= fmin < fmax <= 4000 Hz.')
    if points < 2 or points > len(omega) or not 0 <= fraction <= 1:
        raise ValueError('Il faut 2 a 1000 points et une fraction entre 0 et 1.')
    if level < 0 or max_iter < 0 or not 0 < threshold <= 1 or not np.isfinite(speed) or speed <= 0:
        raise ValueError('Niveau et iterations >= 0 ; seuil dans ]0,1] ; vitesse > 0.')
    # Selection dans les tables, sans interpolation du coefficient Alpha.
    available = np.flatnonzero((frequencies >= fmin-1e-8) & (frequencies <= fmax+1e-8))
    if len(available) < points:
        raise ValueError('Trop de points demandes pour cette sous-plage des tables.')
    indices = available[np.round(np.linspace(0, len(available)-1, points)).astype(int)]
    needed = int(np.ceil(10 * frequencies[indices[-1]] / 340.))
    multiple = max(16, 4**level)
    if N is None:
        N = int(np.ceil(max(50, needed)/multiple)*multiple)
    if N < max(16, needed) or N % (4**level):
        raise ValueError(f'N doit etre >= {max(16, needed)} et divisible par {4**level}.')
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    M, h = 2*N, 1./N
    d, x, y, _, _ = pre._set_geometry_of_domain(M, N, level)
    bp, ap, ad, bn, _, br = pre._set_coefficients_of_pde(M, N)
    _, fd, fn, fr = pre._set_rhs_of_pde(M, N)
    fd[:] = 0.
    chi0 = demo.your_project_density(pre.set2zero(pre._set_chi(M,N,x,y), d), d, fraction)
    boundary_count = np.count_nonzero(d == _env.NODE_ROBIN)
    config = dict(material=material, N=N, level=level, max_iter=max_iter,
                  fraction=fraction, threshold=threshold, speed_kmh=speed,
                  length_m=1., sound_speed=340., source='train_source defaults; coherent',
                  source_sigma=.03, rolling_center=[.9,.5], aero_center=[.5,.5],
                  rolling_peak_hz=1000., aero_peak_hz=500., width_octaves=1.,
                  reference_speed_kmh=300., gain_rolling=1., gain_aero=1.,
                  frequency_indices=indices.tolist(), frequencies_hz=frequencies[indices].tolist(),
                  protocol='Independent optimization at each frequency; same initial chi, no warm start',
                  energy='h^2 sum(abs(u)^2) on interior; absolute, not J/J0 or dB')
    (out/'configuration.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    rows, continuous_densities, binary_densities = [], [], []
    print(f'{points} frequences ; N={N} ; {material} ; {max_iter} iterations maximum par frequence.', flush=True)
    print('Chaque frequence est optimisee independamment ; chi initial identique.', flush=True)
    with (out/'energies.csv').open('w', newline='', encoding='utf-8') as csv_file, \
         (out/'optimisation.log').open('w', encoding='utf-8') as log:
        writer = None
        for number, index in enumerate(indices, 1):
            frequency = float(frequencies[index])
            k, A = material_alpha.solver_parameters(omega[index], alpha[index], 1., 340.)
            source, amplitudes = train_source.gaussian_train_source(d,h,frequency,speed)
            def solve(chi):
                return pro.solve_helmholtz(d,h,float(k),source,fd,fn,fr,bp,ap,ad,bn,br,A*chi)
            def energy(u):
                return float(demo.your_compute_objective_function(d,u,h))
            initial_energy = energy(solve(chi0))
            with contextlib.redirect_stdout(log):
                print(f'\nFREQUENCE {frequency:.9f} Hz')
                chi, history, u, _ = demo.your_optimization_procedure(
                    d,h,float(k),source,fd,fn,fr,bp,ap,ad,bn,br,A*chi0,
                    complex(A),5.,chi0.copy(),fraction,max_iter=max_iter)
            final_energy = energy(u)  # history contient J/J0, pas l'energie absolue.
            chi_binary = demo.binarize_chi_threshold(chi,d,threshold)
            binary_energy = energy(solve(chi_binary))
            if not np.all(np.isfinite([initial_energy,final_energy,binary_energy])) or initial_energy <= 0:
                raise RuntimeError(f'Energie invalide a {frequency} Hz.')
            np.testing.assert_allclose(final_energy/initial_energy,history[-1,0],rtol=1e-8)
            if final_energy > initial_energy*(1+1e-10):
                raise RuntimeError('L optimisation continue a augmente l energie.')
            row = dict(frequency_hz=frequency, energy_initial=initial_energy,
                       energy_optimized=final_energy, energy_binary=binary_energy,
                       ratio_optimized=final_energy/initial_energy,
                       ratio_binary=binary_energy/initial_energy,
                       fraction_initial=float(chi0.sum()/boundary_count),
                       fraction_optimized=float(chi.sum()/boundary_count),
                       fraction_binary=float(chi_binary.sum()/boundary_count),
                       accepted_iterations=len(history)-1,
                       amplitude_rolling=float(amplitudes[0]), amplitude_aero=float(amplitudes[1]))
            if writer is None:
                writer = csv.DictWriter(csv_file,fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            csv_file.flush()
            log.flush()
            rows.append(row)
            continuous_densities.append(chi.copy())
            binary_densities.append(chi_binary.copy())
            print(f'[{number}/{points}] {frequency:.2f} Hz : initial={initial_energy:.4e}, '
                  f'binaire={binary_energy:.4e}',flush=True)
    np.savez_compressed(out/'densites.npz', domain=d, chi_initial=chi0,
                        frequency_hz=frequencies[indices],
                        chi_optimized=np.array(continuous_densities), chi_binary=np.array(binary_densities))
    plot_results(rows,out,material)
    print(f'Figure et donnees : {out}',flush=True)
    return rows, config


def plot_results(rows,out,material):
    frequency = [r['frequency_hz'] for r in rows]
    fig, axes = plt.subplots(2,1,figsize=(10,8),sharex=True,layout='constrained')
    for key,label,color in [('initial','Frontiere initiale','#687386'),
                            ('binary','Apres seuillage binaire','#be5c20')]:
        axes[0].semilogy(frequency,[r['energy_'+key] for r in rows],'.-',label=label,color=color)
    axes[0].set_ylabel('Energie absolue du champ (unites du modele)')
    axes[0].legend()
    for key,label,color in [('binary','Binaire / initiale','#be5c20')]:
        axes[1].plot(frequency,[r['ratio_'+key] for r in rows],'.-',label=label,color=color)
    axes[1].axhline(1.,color='#687386',linestyle='--',label='Energie initiale')
    axes[1].set(xlabel='Frequence (Hz)',ylabel='Rapport a l energie initiale')
    axes[1].legend()
    for ax in axes:
        ax.grid(True,alpha=.25)
    fig.suptitle(f'Source train - {material}\nOptimisation independante a chaque frequence (chi different)')
    fig.savefig(out/'energie_avant_apres.png',dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fmin',type=float,default=500.)
    parser.add_argument('--fmax',type=float,default=1000.)
    parser.add_argument('--points',type=int,default=100)
    parser.add_argument('--N',type=int,default=None)
    parser.add_argument('--max-iter',type=int,default=100)
    parser.add_argument('--material',choices=material_alpha.MATERIALS,default='ROCK_WOOL')
    parser.add_argument('--level',type=int,default=0)
    parser.add_argument('--fraction',type=float,default=.3)
    parser.add_argument('--speed',type=float,default=300.)
    parser.add_argument('--threshold',type=float,default=.5)
    parser.add_argument('--output',default='resultats_energie_frequence_train')
    run(**vars(parser.parse_args()))
