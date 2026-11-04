"""Experience : optimisation independante a chaque frequence pour la source train.

1. Balayage initial : calcul de J(chi_0) et J(chi_full).
2. Detection du pic initial.
3. Optimisation a chaque frequence, en repartant du meme chi initial.
4. Evaluation sur toute la bande du chi optimise au pic initial.

Les deux figures sont affichees sans enregistrer de fichiers.
"""
import time
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import matplotlib.pyplot as plt
import numpy as np

import _env
import preprocessing as pre
import processing as pro
import material_alpha
import train_source
import demo_control_polycopie2024 as demo
from test_energie_pic_fixe_unifie import build_geometry, wall_label


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
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
    return fig


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
    return fig


# ---------------------------------------------------------------------------
# Experience
# ---------------------------------------------------------------------------
def run(fmin=train_source.FREQUENCY_MIN_HZ, fmax=train_source.FREQUENCY_MAX_HZ,
        points=500, N=None, max_iter=30, material='ROCK_WOOL', j=0,
        fraction=.3, speed=300., threshold=.5):
    """Optimise chaque frequence, puis evalue le chi du pic sur toute la bande.

    points : nombre de frequences echantillonnees (None = toutes).
    N      : mailles sur 1 m (None = automatique).
    j      : 0 = plat ; 1, 2, 3, ... = niveau de la fractale de Minkowski.
    """
    label = wall_label(j)
    t_start = time.time()
    omega, alpha, _ = material_alpha.load_material(material)
    frequencies = omega / (2 * np.pi)
    available = np.flatnonzero((frequencies >= fmin - 1e-8) & (frequencies <= fmax + 1e-8))
    if points is None:
        indices = available
    else:
        indices = available[np.round(np.linspace(0, len(available) - 1, points)).astype(int)]
    points = len(indices)

    d, x, y, mesh_cells, M, h = build_geometry(j, N, frequencies[indices[-1]])
    N = d.shape[1]
    bp, ap, ad, bn, _, br = pre._set_coefficients_of_pde(M, N)
    _, fd, fn, fr = pre._set_rhs_of_pde(M, N)
    fd[:] = 0.
    chi0 = demo.your_project_density(pre.set2zero(pre._set_chi(M, N, x, y), d), d, fraction)
    boundary_count = np.count_nonzero(d == _env.NODE_ROBIN)

    chi_full = np.zeros((M, N), dtype=float)
    chi_full[d == _env.NODE_ROBIN] = 1.0

    print('=' * 65)
    print(f'OPTIMISATION A CHAQUE FREQUENCE ({material}) - {label}')
    print(f'Grille : {points} frequences ; N={mesh_cells} mailles ; max_iter={max_iter}')
    print('=' * 65)

    def solve_eval(k, source, A, chi):
        u = pro.solve_helmholtz(d, h, float(k), source, fd, fn, fr, bp, ap, ad, bn, br, A * chi)
        return float(demo.your_compute_objective_function(d, u, h))

    # -------------------------------------------------------------
    # ETAPE 1 : Balayage initial de reference
    # -------------------------------------------------------------
    print(f'>>> ETAPE 1/4 : Balayage initial ({points} frequences)...', flush=True)
    initial_energies = []
    full_energies = []
    for number, index in enumerate(indices, 1):
        frequency = float(frequencies[index])
        k, A = material_alpha.solver_parameters(omega[index], alpha[index], 1., 340.)
        source, _ = train_source.gaussian_train_source(d, h, frequency, speed)
        initial_energies.append(solve_eval(k, source, A, chi0))
        full_energies.append(solve_eval(k, source, A, chi_full))
        if number == 1 or number % 50 == 0 or number == points:
            print(f'   [Etape 1] {number}/{points} : {frequency:.2f} Hz', flush=True)

    # -------------------------------------------------------------
    # ETAPE 2 : Detection du pic d'energie initiale
    # -------------------------------------------------------------
    peak_idx = int(np.argmax(initial_energies))
    peak_frequency = float(frequencies[indices[peak_idx]])
    print(f'>>> ETAPE 2/4 : Pic initial a {peak_frequency:.2f} Hz', flush=True)

    # -------------------------------------------------------------
    # ETAPE 3 : Optimisation independante a chaque frequence
    # -------------------------------------------------------------
    print('>>> ETAPE 3/4 : Optimisation par frequence (meme chi initial)...', flush=True)
    rows = []
    for position, index in enumerate(indices):
        frequency = float(frequencies[index])
        k, A = material_alpha.solver_parameters(omega[index], alpha[index], 1., 340.)
        source, amplitudes = train_source.gaussian_train_source(d, h, frequency, speed)
        chi_opt, history, u_opt, _ = demo.your_optimization_procedure(
            d, h, float(k), source, fd, fn, fr, bp, ap, ad, bn, br,
            A * chi0, complex(A), 5., chi0.copy(), fraction, max_iter=max_iter)
        chi_binary = demo.binarize_chi_threshold(chi_opt, d, threshold, fraction=fraction)
        e_opt = float(demo.your_compute_objective_function(d, u_opt, h))
        e_bin = solve_eval(k, source, A, chi_binary)
        e_init, e_full = initial_energies[position], full_energies[position]
        rows.append(dict(
            frequency_hz=frequency, energy_initial=e_init,
            energy_optimized=e_opt, energy_binary=e_bin, energy_full=e_full,
            ratio_optimized=e_opt / e_init, ratio_binary=e_bin / e_init,
            ratio_full=e_full / e_init,
            fraction_initial=float(chi0.sum() / boundary_count),
            fraction_optimized=float(chi_opt.sum() / boundary_count),
            fraction_binary=float(chi_binary.sum() / boundary_count),
            accepted_iterations=len(history) - 1,
            amplitude_rolling=float(amplitudes[0]), amplitude_aero=float(amplitudes[1])))
        if position == peak_idx:
            fixed_chi = chi_opt.copy()
            fixed_binary = chi_binary.copy()
        print(f'   [Etape 3] {position + 1}/{points} : {frequency:.2f} Hz -> '
              f'init={e_init:.3e}, opt={e_opt:.3e}, bin={e_bin:.3e}', flush=True)

    # -------------------------------------------------------------
    # ETAPE 4 : Evaluation spectrale du mur fixe optimise au pic
    # -------------------------------------------------------------
    print(f'>>> ETAPE 4/4 : Evaluation du chi fixe de {peak_frequency:.2f} Hz...', flush=True)
    fixed_rows = []
    for number, (index, reference) in enumerate(zip(indices, rows), 1):
        frequency = float(frequencies[index])
        k, A = material_alpha.solver_parameters(omega[index], alpha[index], 1., 340.)
        source, _ = train_source.gaussian_train_source(d, h, frequency, speed)
        fixed_rows.append(dict(
            frequency_hz=frequency, target_frequency_hz=peak_frequency,
            energy_initial=reference['energy_initial'],
            energy_fixed=solve_eval(k, source, A, fixed_chi),
            energy_fixed_binary=solve_eval(k, source, A, fixed_binary),
            energy_full=reference['energy_full']))
        if number == 1 or number % 50 == 0 or number == points:
            print(f'   [Etape 4] {number}/{points} : {frequency:.2f} Hz', flush=True)

    config = dict(
        material=material, N=mesh_cells, grid_shape=list(d.shape), spacestep=h,
        geometry='plate' if j == 0 else 'fractale', level=int(j), max_iter=max_iter,
        fraction=fraction, threshold=threshold, speed_kmh=speed,
        length_m=1., sound_speed=340.,
        fixed_chi_target_frequency_hz=peak_frequency,
        fixed_chi_selection='Maximum sampled initial energy',
        protocol='Independent optimization at each frequency; same initial chi, no warm start',
        points=points, frequency_indices=indices.tolist(), frequencies_hz=frequencies[indices].tolist(),
        total_time_seconds=float(time.time() - t_start))

    plot_results(rows, None, material)
    plot_fixed_results(fixed_rows, None, material)
    print(f'Temps total d execution : {time.time() - t_start:.2f} s')
    plt.show()
    return rows, config


if __name__ == '__main__':
    # Modifier ici les parametres de l'experience.
    # j : 0 = surface plane ; 1, 2, 3, ... = niveau de la fractale.
    rows, config = run(
        fmin=train_source.FREQUENCY_MIN_HZ,
        fmax=train_source.FREQUENCY_MAX_HZ,
        points=500,           # None = toutes les frequences des tables
        N=None,               # None = automatique
        max_iter=30,
        material='ROCK_WOOL',  # ou 'RECYCLED_RUBBER'
        j=0,
        fraction=0.3,
        speed=300.,           # km/h
        threshold=0.5,
    )
