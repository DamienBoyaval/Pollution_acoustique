"""Experience : energie initiale / optimisee binaire pour la source train.

Lancer : python test_energie_frequence.py
Chaque frequence a sa propre optimisation, repartant du meme chi initial.
La premiere figure compare des chi differents. La seconde evalue sur toute
la bande le chi optimise au maximum d'energie initiale echantillonne.
Par defaut, toutes les frequences disponibles des tables sont utilisees.
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


def run(fmin=train_source.FREQUENCY_MIN_HZ, fmax=train_source.FREQUENCY_MAX_HZ, points=None, N=None, max_iter=30,
        material='ROCK_WOOL', level=0, fraction=.3, speed=300.,
        threshold=.5, output='resultats_energie_frequence_train'):
    omega, alpha, _ = material_alpha.load_material(material)
    frequencies = omega / (2 * np.pi)
    # Selection dans les tables, sans interpolation du coefficient Alpha.
    available = np.flatnonzero((frequencies >= fmin-1e-8) & (frequencies <= fmax+1e-8))
    if len(available) < 2:
        raise ValueError('La bande doit contenir au moins deux frequences des tables.')
    if points is None:
        indices = available
    else:
        if not isinstance(points, (int, np.integer)) or not 2 <= points <= len(available):
            raise ValueError('Choisir entre 2 et le nombre de frequences disponibles.')
        indices = available[np.round(np.linspace(0, len(available)-1, points)).astype(int)]
    points = len(indices)
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

    # Paroi entierement absorbante (chi = 1 sur toute la frontiere de Robin)
    chi_full = np.zeros((M, N), dtype=float)
    chi_full[d == _env.NODE_ROBIN] = 1.0

    config = dict(material=material, N=N, level=level, max_iter=max_iter,
                  fraction=fraction, threshold=threshold, speed_kmh=speed,
                  length_m=1., sound_speed=340., source='train_source defaults; coherent',
                  source_sigma=.03, rolling_center=[.9,.5], aero_center=[.5,.5],
                  rolling_peak_hz=train_source.ROLLING_PEAK_HZ, aero_peak_hz=train_source.AERO_PEAK_HZ, width_octaves=1.,
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
            chi_binary = demo.binarize_chi_threshold(chi,d,threshold,fraction=fraction)
            binary_energy = energy(solve(chi_binary))
            full_energy = energy(solve(chi_full))
            if not np.all(np.isfinite([initial_energy,final_energy,binary_energy,full_energy])) or initial_energy <= 0:
                raise RuntimeError(f'Energie invalide a {frequency} Hz.')
            np.testing.assert_allclose(final_energy/initial_energy,history[-1,0],rtol=1e-8)
            if final_energy > initial_energy*(1+1e-10):
                raise RuntimeError('L optimisation continue a augmente l energie.')
            row = dict(frequency_hz=frequency, energy_initial=initial_energy,
                       energy_optimized=final_energy, energy_binary=binary_energy,
                       energy_full=full_energy,
                       ratio_optimized=final_energy/initial_energy,
                       ratio_binary=binary_energy/initial_energy,
                       ratio_full=full_energy/initial_energy,
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
                  f'optimale={final_energy:.4e}, binaire={binary_energy:.4e}, full={full_energy:.4e}',flush=True)
    np.savez_compressed(out/'densites.npz', domain=d, chi_initial=chi0, chi_full=chi_full,
                        frequency_hz=frequencies[indices],
                        chi_optimized=np.array(continuous_densities), chi_binary=np.array(binary_densities))
    # Reutiliser le chi de la frequence au plus grand pic INITIAL.
    peak_position = int(np.argmax([r['energy_initial'] for r in rows]))
    peak_frequency = rows[peak_position]['frequency_hz']
    fixed_chi = continuous_densities[peak_position].copy()
    fixed_binary = binary_densities[peak_position].copy()
    fixed_rows = []
    print(f'Chi fixe : optimisation au pic initial {peak_frequency:.3f} Hz.', flush=True)
    for number, (index, reference) in enumerate(zip(indices, rows), 1):
        frequency = float(frequencies[index])
        k, A = material_alpha.solver_parameters(omega[index], alpha[index], 1., 340.)
        source, _ = train_source.gaussian_train_source(d, h, frequency, speed)
        energies = []
        for density in (fixed_chi, fixed_binary):
            u = pro.solve_helmholtz(d,h,float(k),source,fd,fn,fr,bp,ap,ad,bn,br,A*density)
            energies.append(float(demo.your_compute_objective_function(d,u,h)))
        if not np.all(np.isfinite(energies)):
            raise RuntimeError(f'Energie du chi fixe invalide a {frequency} Hz.')
        fixed_rows.append(dict(frequency_hz=frequency, target_frequency_hz=peak_frequency,
                               energy_initial=reference['energy_initial'],
                               energy_fixed=energies[0], energy_fixed_binary=energies[1],
                               energy_full=reference['energy_full']))
        if number == 1 or number % 25 == 0 or number == points:
            print(f'Chi fixe [{number}/{points}] {frequency:.2f} Hz', flush=True)
    with (out/'energies_chi_fixe.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(fixed_rows[0]))
        writer.writeheader(); writer.writerows(fixed_rows)
    np.savez_compressed(out/'chi_fixe.npz', domain=d, chi_initial=chi0,
                        chi_fixed=fixed_chi, chi_fixed_binary=fixed_binary,
                        target_frequency_hz=peak_frequency, target_table_index=indices[peak_position])
    config['fixed_chi_target_frequency_hz'] = peak_frequency
    config['fixed_chi_selection'] = 'Maximum sampled initial energy; continuous optimized chi and its thresholded version'
    (out/'configuration.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    plot_results(rows,out,material)
    plot_fixed_results(fixed_rows, out, material)
    print(f'Figure et donnees : {out}',flush=True)
    return rows, config


def plot_results(rows, out, material):
    frequency = [float(r['frequency_hz']) for r in rows]
    fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
    
    for key, label, color in [('initial', 'Frontiere initiale', '#687386'),
                              ('optimized', 'Repartition optimisee par frequence', '#1971c2'),
                              ('binary', 'Apres seuillage binaire', '#be5c20'),
                              ('full', 'Paroi entierement absorbante (100%)', '#2b8a3e')]:
        ax.semilogy(frequency, [float(r['energy_' + key]) for r in rows], '-', label=label, color=color)
        
    ax.set_xlabel('Frequence (Hz)')
    ax.set_ylabel('Energie absolue du champ (unites du modele)')
    ax.legend()
    ax.grid(True, alpha=0.25)
    
    fig.suptitle(f'Source train - {material}\nOptimisation independante a chaque frequence (chi different)')
    fig.savefig(out / 'energie_avant_apres.png', dpi=160)
    plt.close(fig)


def plot_fixed_results(rows, out, material):
    frequency = np.array([float(r['frequency_hz']) for r in rows])
    target = float(rows[0]['target_frequency_hz'])
    fig, ax = plt.subplots(figsize=(11, 5), layout='constrained')
    for key, label, color in [
        ('initial', 'Repartition initiale', '#687386'),
        ('fixed', 'Chi continu du pic, fixe sur toute la bande', '#1971c2'),
        ('fixed_binary', 'Chi binaire du pic, fixe sur toute la bande', '#be5c20'),
        ('full', 'Paroi entierement absorbante (100%)', '#2b8a3e')]:
        values = np.array([float(r['energy_' + key]) for r in rows])
        ax.semilogy(frequency, values, '-', label=label, color=color)
        peak = int(np.argmax(values))
        ax.scatter(frequency[peak], values[peak], color=color, marker='*', s=65, zorder=5)
    ax.axvline(target, color='#555555', linestyle=':', label=f'Cible initiale : {target:.1f} Hz')
    ax.set(xlabel='Frequence (Hz)', ylabel='Energie absolue du champ (unites du modele)')
    ax.grid(True, alpha=.25)
    ax.legend(fontsize=8)
    fig.suptitle(f'Source train - {material} : un seul chi optimise a {target:.1f} Hz\n'
                 'Sans reoptimisation aux autres frequences ; etoiles : maxima echantillonnes')
    fig.savefig(out/'energie_chi_fixe.png', dpi=160)
    plt.close(fig)


def replot(output='resultats_energie_frequence_train', material=None):
    out = Path(output).resolve()
    csv_path = out / 'energies.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f'{csv_path} introuvable.')
    if material is None:
        cfg_path = out / 'configuration.json'
        if cfg_path.exists():
            with cfg_path.open('r', encoding='utf-8') as f:
                material = json.load(f).get('material', 'ROCK_WOOL')
        else:
            material = 'ROCK_WOOL'
    with csv_path.open('r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    plot_results(rows, out, material)
    fixed_path = out/'energies_chi_fixe.csv'
    if fixed_path.exists():
        with fixed_path.open('r', encoding='utf-8') as file:
            plot_fixed_results(list(csv.DictReader(file)), out, material)
    print(f'Graphe mis a jour dans : {out / "energie_avant_apres.png"}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fmin',type=float,default=train_source.FREQUENCY_MIN_HZ)
    parser.add_argument('--fmax',type=float,default=train_source.FREQUENCY_MAX_HZ)
    parser.add_argument('--points',type=int,default=None, help='Nombre de frequences ; par defaut toutes les lignes de la bande (714 pour 500-3000 Hz).')
    parser.add_argument('--N',type=int,default=None)
    parser.add_argument('--max-iter',type=int,default=30)
    parser.add_argument('--material',choices=material_alpha.MATERIALS,default='ROCK_WOOL')
    parser.add_argument('--level',type=int,default=0)
    parser.add_argument('--fraction',type=float,default=.3)
    parser.add_argument('--speed',type=float,default=300.)
    parser.add_argument('--threshold',type=float,default=.5)
    parser.add_argument('--output',default='resultats_energie_frequence_train')
    parser.add_argument('--replot',action='store_true',help='Re-tracer le graphe depuis le CSV existant')
    args = vars(parser.parse_args())
    if args.pop('replot'):
        replot(args['output'], args['material'])
    else:
        run(**args)
