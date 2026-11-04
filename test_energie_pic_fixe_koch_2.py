"""Paroi de von Koch quadratique (geometrie.py), niveau 2 par defaut.

Experience rapide : balayage initial, optimisation ciblée au pic et evaluation du mur fixe.

Protocole en 4 etapes :
1. Balayage initial sur toutes les frequences : calcul de J(chi_0) et J(chi_full).
2. Detection du pic initial : identification automatique de la frequence de resonance maximale.
3. Optimisation ciblee : optimisation de chi uniquement a cette frequence de pic (max_iter iterations).
4. Evaluation spectrale du mur fixe : evaluation de chi fixe (continu et binaire) sur toute la bande.

Les figures sont affichees a la fin du calcul, sans enregistrer de fichiers.

Lancer :
    python test_energie_pic_fixe_koch_2.py
    python test_energie_pic_fixe_koch_2.py --points 50 --max-iter 25
"""
import argparse
import csv
import json
import os
from pathlib import Path
import time
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import matplotlib.pyplot as plt
import numpy as np

import _env
import geometrie
import preprocessing as pre
import processing as pro
import material_alpha
import train_source
import demo_control_polycopie2024 as demo


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
    fig.suptitle(f'Source train - {material} - Paroi de Koch niveau 2 : un seul chi optimise au pic de {target:.1f} Hz\n'
                 'Sans reoptimisation aux autres frequences ; etoiles : maxima echantillonnes')
    return fig


def plot_binary_comparison(rows, out, material):
    """Deuxieme figure : comparaison exclusive entre la repartition initiale et la paroi binaire fixe."""
    frequency = np.array([float(r['frequency_hz']) for r in rows])
    target = float(rows[0]['target_frequency_hz'])
    fig, ax = plt.subplots(figsize=(11, 5), layout='constrained')
    for key, label, color in [
        ('initial', 'Repartition initiale (reference 30%)', '#687386'),
        ('fixed_binary', 'Paroi binaire fixe (optimisee au pic)', '#be5c20')]:
        values = np.array([float(r['energy_' + key]) for r in rows])
        ax.semilogy(frequency, values, '-', label=label, color=color, linewidth=1.5)
        peak = int(np.argmax(values))
        ax.scatter(frequency[peak], values[peak], color=color, marker='*', s=80, zorder=5,
                   label=f'Max {label.split()[0]} : {values[peak]:.2e} a {frequency[peak]:.1f} Hz')
    ax.axvline(target, color='#2b8a3e', linestyle='--', linewidth=1.2, label=f'Pic cible optimise : {target:.1f} Hz')
    ax.set(xlabel='Frequence (Hz)', ylabel='Energie absolue du champ (unites du modele)')
    ax.grid(True, alpha=.25)
    ax.legend(fontsize=8.5, loc='upper right')
    fig.suptitle(f'Source train - {material} - Paroi de Koch niveau 2 : Comparaison Initiale vs Paroi Binaire Fixe\n'
                 f'Mur optimise a {target:.1f} Hz puis binarise (30% exact de materiau) ; etoiles : maxima echantillonnes')
    return fig


def plot_total_energy_barchart(rows, out, material):
    """Troisieme figure : bar chart comparant l'energie totale spectrale (Initiale, Binaire fixe et 100% absorbante)."""
    e_init = np.array([float(r['energy_initial']) for r in rows])
    e_bin = np.array([float(r['energy_fixed_binary']) for r in rows])

    total_init = float(np.sum(e_init))
    total_bin = float(np.sum(e_bin))
    total_full = float(np.sum([float(r['energy_full']) for r in rows]))

    target = float(rows[0]['target_frequency_hz'])
    n_pts = len(rows)

    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')

    labels = ['Paroi initiale\n(avant optimisation)', 'Paroi binaire fixe\n(optimisée au pic)',
              'Paroi entièrement absorbante\n(100% de matériau)']
    totals = [total_init, total_bin, total_full]
    colors = ['#687386', '#be5c20', '#2b8a3e']

    bars = ax.bar(labels, totals, color=colors, width=0.48, edgecolor='#333333', linewidth=1.2, zorder=3)

    # Affichage des valeurs numeriques precises au-dessus de chaque barre
    for bar, val in zip(bars, totals):
        height = bar.get_height()
        rel_str = ' (100%)' if val == total_init else f' ({100*val/total_init:.1f}%)'
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height * 1.02,
            f'{val:.3e}{rel_str}',
            ha='center', va='bottom', fontsize=10.5, fontweight='bold', color='#222222'
        )

    # Espace superieur pour les valeurs au-dessus des barres
    y_top = max(totals) * 1.25
    ax.set_ylim(0, y_top)

    ax.set_ylabel('Énergie acoustique totale cumulée (unités du modèle)', fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.35, zorder=0)
    fig.suptitle(f'Source train - {material} - Paroi de Koch niveau 2 : Bilan Énergétique Global ({n_pts} fréquences)\n'
                 f'Mur fixé optimisé à {target:.1f} Hz ; référence entièrement absorbante à 100%',
                 fontsize=11.5, fontweight='bold')

    return fig


def replot(output='resultats_energie_pic_fixe_koch_2', material=None):
    out = Path(output).resolve()
    csv_path = out / 'energies_pic_fixe.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f'Fichier introuvable : {csv_path}')
    if material is None:
        cfg = out / 'configuration.json'
        if cfg.exists():
            material = json.loads(cfg.read_text(encoding='utf-8')).get('material', 'ROCK_WOOL')
        else:
            material = 'ROCK_WOOL'
    with csv_path.open('r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    plot_fixed_results(rows, out, material)
    plot_binary_comparison(rows, out, material)
    plot_total_energy_barchart(rows, out, material)
    plt.show()


def run(fmin=train_source.FREQUENCY_MIN_HZ, fmax=train_source.FREQUENCY_MAX_HZ, points=None, N=None,
        max_iter=30, material='ROCK_WOOL', level=2, fraction=.3, speed=300.,
        threshold=.5, output='resultats_energie_pic_fixe_koch_2', solver='standard', quantity=None):
    if solver != 'standard':
        raise ValueError('Seul le solveur du code d origine (standard) est disponible.')
    t_start = time.time()
    omega, alpha, _ = material_alpha.load_material(material)
    frequencies = omega / (2 * np.pi)
    available = np.flatnonzero((frequencies >= fmin - 1e-8) & (frequencies <= fmax + 1e-8))
    if len(available) < 2:
        raise ValueError('La bande doit contenir au moins deux frequences des tables.')
    if points is None:
        indices = available
    else:
        if not isinstance(points, (int, np.integer)) or not 2 <= points <= len(available):
            raise ValueError('Choisir entre 2 et le nombre de frequences disponibles.')
        indices = available[np.round(np.linspace(0, len(available) - 1, points)).astype(int)]
    points = len(indices)

    if not isinstance(level, (int, np.integer)) or level != 2:
        raise ValueError('Ce programme utilise le niveau de Koch 2.')
    # N reste le nombre de mailles sur une largeur de 1 m.
    # Quatre motifs de Koch, avec au moins deux mailles par petit segment.
    needed = int(np.ceil(10 * frequencies[indices[-1]] / 340.))
    multiple = geometrie.N_MORCEAUX * 3**level
    minimum = max(50, needed, geometrie.CELLULES_MIN_MOTIF * multiple)
    if N is None:
        N = int(np.ceil(minimum / multiple) * multiple)
    if not isinstance(N, (int, np.integer)) or N < minimum or N % multiple:
        raise ValueError(f'N doit etre >= {minimum} et divisible par {multiple}.')

    out = Path(output).resolve()
    mesh_cells = N
    h = 1. / mesh_cells
    n_piece = mesh_cells // geometrie.N_MORCEAUX
    floor = int(np.ceil(n_piece / 2)) + geometrie.MARGE
    M = floor + mesh_cells + 1
    cells = geometrie.carres_entre((0, floor), (mesh_cells, floor + mesh_cells))
    nodes = []
    for piece in range(geometrie.N_MORCEAUX):
        line = geometrie.koch_sur_segment(
            cells, (piece * n_piece, floor), ((piece + 1) * n_piece, floor), level)
        nodes.extend(np.array([[px], [M - 1 - py]]) for px, py in line)
    d = geometrie.carres_vers_domaine(cells, mesh_cells + 1, M)
    # Reprendre la repartition initiale sur 20 a 50 % du parcours de la paroi.
    # Le dernier point lateral est exclu par _set_chi, comme dans le test plat.
    x, y = pre.create_fractal_coordinates(nodes, np.empty((M, mesh_cells)))
    N = d.shape[1]
    bp, ap, ad, bn, _, br = pre._set_coefficients_of_pde(M, N)
    _, fd, fn, fr = pre._set_rhs_of_pde(M, N)
    fd[:] = 0.
    chi0 = demo.your_project_density(pre.set2zero(pre._set_chi(M, N, x, y), d), d, fraction)
    boundary_count = np.count_nonzero(d == _env.NODE_ROBIN)
    if quantity is not None:
        if not np.isfinite(quantity) or not 0 < quantity <= h * boundary_count:
            raise ValueError('quantity doit etre positive et ne pas depasser la longueur discrete du bord.')
        fraction = float(quantity / (h * boundary_count))
        chi0 = demo.your_project_density(
            pre.set2zero(pre._set_chi(M, N, x, y), d), d, fraction)

    chi_full = np.zeros((M, N), dtype=float)
    chi_full[d == _env.NODE_ROBIN] = 1.0

    print('=' * 65)
    print(f'SIMULATION OPTIMISATION AU PIC INITIAL ({material})')
    print(f'Grille : {points} frequences ({frequencies[indices[0]]:.1f} a {frequencies[indices[-1]]:.1f} Hz)')
    print(f'Maillage : N={mesh_cells} mailles (grille {M}x{N} = {M*N} noeuds) ; max_iter={max_iter}')
    print(f'Paroi : von Koch quadratique, niveau {level} ; pas h={h:.6g} m')
    print(f'Mode solveur : {solver}')
    print('=' * 65)

    def solve_eval(k, s, A, chi, return_field=False):
        u = pro.solve_helmholtz(d, h, float(k), s, fd, fn, fr, bp, ap, ad, bn, br, A * chi)
        ene = float(demo.your_compute_objective_function(d, u, h))
        if return_field:
            return ene, u
        return ene

    # -------------------------------------------------------------
    # ETAPE 1 : Balayage initial sur toutes les frequences (chi0 et chi_full)
    # -------------------------------------------------------------
    print(f'>>> ETAPE 1/4 : Balayage initial de reference ({points} frequences)...', flush=True)
    t1 = time.time()
    initial_energies = []
    full_energies = []
    sources = []
    k_list = []
    A_list = []

    for idx_num, index in enumerate(indices, 1):
        frequency = float(frequencies[index])
        k, A = material_alpha.solver_parameters(omega[index], alpha[index], 1., 340.)
        source, _ = train_source.gaussian_train_source(d, h, frequency, speed)
        sources.append(source)
        k_list.append(k)
        A_list.append(A)

        e_init = solve_eval(k, source, A, chi0)
        e_full = solve_eval(k, source, A, chi_full)
        initial_energies.append(e_init)
        full_energies.append(e_full)

        if idx_num == 1 or idx_num % 50 == 0 or idx_num == points:
            print(f'   [Etape 1] {idx_num}/{points} : {frequency:.2f} Hz -> E_init={e_init:.4e}, E_full={e_full:.4e}', flush=True)

    print(f'   Etape 1 terminee en {time.time() - t1:.2f} s.\n', flush=True)

    # -------------------------------------------------------------
    # ETAPE 2 : Detection du pic d'energie initiale
    # -------------------------------------------------------------
    peak_idx = int(np.argmax(initial_energies))
    peak_frequency = float(frequencies[indices[peak_idx]])
    peak_initial_energy = initial_energies[peak_idx]

    print('>>> ETAPE 2/4 : Detection du pic de resonance initial')
    print(f'   -> Frequence la plus defavorable identifiee : {peak_frequency:.2f} Hz')
    print(f'   -> Energie initiale maximale a cette frequence : {peak_initial_energy:.4e}\n', flush=True)

    # -------------------------------------------------------------
    # ETAPE 3 : Optimisation ciblee au pic
    # -------------------------------------------------------------
    print(f'>>> ETAPE 3/4 : Optimisation de chi uniquement a {peak_frequency:.2f} Hz ({max_iter} iterations max)...', flush=True)
    t3 = time.time()
    k_peak = k_list[peak_idx]
    A_peak = A_list[peak_idx]
    source_peak = sources[peak_idx]

    chi_opt, history, u_opt, _ = demo.your_optimization_procedure(
        d, h, float(k_peak), source_peak, fd, fn, fr, bp, ap, ad, bn, br,
        A_peak * chi0, complex(A_peak), 5., chi0.copy(), fraction, max_iter=max_iter)

    chi_binary = demo.binarize_chi_threshold(chi_opt, d, threshold, fraction=fraction)
    e_opt_peak = float(demo.your_compute_objective_function(d, u_opt, h))
    e_bin_peak = solve_eval(k_peak, source_peak, A_peak, chi_binary)

    gain_opt_peak = 100. * (1. - e_opt_peak / peak_initial_energy)
    gain_bin_peak = 100. * (1. - e_bin_peak / peak_initial_energy)
    print(f'   Iterations effectuees : {len(history) - 1}')
    print(f'   Energie initiale       : {peak_initial_energy:.4e}')
    print(f'   Energie continue opt.  : {e_opt_peak:.4e} (attenuation : -{gain_opt_peak:.2f} %)')
    print(f'   Energie binaire        : {e_bin_peak:.4e} (attenuation : -{gain_bin_peak:.2f} %)')
    print(f'   Etape 3 terminee en {time.time() - t3:.2f} s.\n', flush=True)

    # -------------------------------------------------------------
    # ETAPE 4 : Evaluation spectrale du mur fixe sur toute la bande
    # -------------------------------------------------------------
    print(f'>>> ETAPE 4/4 : Evaluation de ce mur unique fixe sur les {points} frequences...', flush=True)
    t4 = time.time()
    rows = []

    for idx_num, (index, k, A, source, e_init, e_full) in enumerate(
            zip(indices, k_list, A_list, sources, initial_energies, full_energies), 1):
        frequency = float(frequencies[index])
        e_fixed = solve_eval(k, source, A, chi_opt)
        e_fixed_bin = solve_eval(k, source, A, chi_binary)

        row = dict(
            frequency_hz=frequency,
            target_frequency_hz=peak_frequency,
            energy_initial=e_init,
            energy_fixed=e_fixed,
            energy_fixed_binary=e_fixed_bin,
            energy_full=e_full,
            ratio_fixed=e_fixed / e_init,
            ratio_fixed_binary=e_fixed_bin / e_init,
            ratio_full=e_full / e_init,
        )
        rows.append(row)

        if idx_num == 1 or idx_num % 50 == 0 or idx_num == points:
            print(f'   [Etape 4] {idx_num}/{points} : {frequency:.2f} Hz -> init={e_init:.3e}, fixe={e_fixed:.3e}, bin={e_fixed_bin:.3e}', flush=True)

    print(f'   Etape 4 terminee en {time.time() - t4:.2f} s.\n', flush=True)

    config = dict(
        material=material, N=mesh_cells, grid_shape=list(d.shape), spacestep=h,
        geometry='koch', pieces=geometrie.N_MORCEAUX, level=level, max_iter=max_iter,
        fraction=fraction, threshold=threshold, speed_kmh=speed,
        material_quantity=float(h * chi0.sum()),
        binary_quantity=float(h * chi_binary.sum()),
        length_m=1., sound_speed=340., solver=solver,
        target_frequency_hz=peak_frequency, target_table_index=int(indices[peak_idx]),
        peak_initial_energy=float(peak_initial_energy),
        peak_optimized_energy=float(e_opt_peak),
        peak_binary_energy=float(e_bin_peak),
        total_time_seconds=float(time.time() - t_start),
        source='train_source defaults; coherent',
        points=points, frequencies_hz=frequencies[indices].tolist()
    )

    plot_fixed_results(rows, out, material)
    plot_binary_comparison(rows, out, material)
    plot_total_energy_barchart(rows, out, material)

    # -------------------------------------------------------------
    # Bilan final affiche
    # -------------------------------------------------------------
    max_init = np.max([r['energy_initial'] for r in rows])
    max_fixed = np.max([r['energy_fixed'] for r in rows])
    max_fixed_bin = np.max([r['energy_fixed_binary'] for r in rows])
    max_full = np.max([r['energy_full'] for r in rows])

    f_max_init = rows[int(np.argmax([r['energy_initial'] for r in rows]))]['frequency_hz']
    f_max_fixed = rows[int(np.argmax([r['energy_fixed'] for r in rows]))]['frequency_hz']
    f_max_bin = rows[int(np.argmax([r['energy_fixed_binary'] for r in rows]))]['frequency_hz']

    print('=' * 65)
    print('BILAN GLOBAL SUR TOUTE LA BANDE (500 - 3000 Hz) :')
    print(f'1. Pic initial max        : {max_init:.4e} (a {f_max_init:.1f} Hz)')
    print(f'2. Pic continu fixe max   : {max_fixed:.4e} (a {f_max_fixed:.1f} Hz) -> Atténuation globale : -{100*(1-max_fixed/max_init):.1f} %')
    print(f'3. Pic binaire fixe max   : {max_fixed_bin:.4e} (a {f_max_bin:.1f} Hz) -> Atténuation globale : -{100*(1-max_fixed_bin/max_init):.1f} %')
    print(f'4. Reference 100% absorb. : {max_full:.4e}')
    print(f'Temps total d execution   : {time.time() - t_start:.2f} s')
    print('=' * 65)
    plt.show()
    return rows, config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fmin', type=float, default=train_source.FREQUENCY_MIN_HZ)
    parser.add_argument('--fmax', type=float, default=train_source.FREQUENCY_MAX_HZ)
    parser.add_argument('--points', type=int, default=None, help='Nombre de frequences a echantillonner (par defaut toutes, 714 pour 500-3000 Hz).')
    parser.add_argument('--N', type=int, default=None, help='Mailles sur la largeur ; multiple de 4 * 3**level. Automatique par defaut.')
    parser.add_argument('--max-iter', type=int, default=30, help='Iterations max pour l optimisation unique au pic (defaut : 30).')
    parser.add_argument('--material', choices=material_alpha.MATERIALS, default='ROCK_WOOL')
    parser.add_argument('--level', type=int, choices=[2], default=2, help='Programme dedie au niveau 2.')
    parser.add_argument('--fraction', type=float, default=.3)
    parser.add_argument('--quantity', type=float, help='Quantite Q=h*sum(chi), prioritaire sur --fraction (L=1 m).')
    parser.add_argument('--speed', type=float, default=300.)
    parser.add_argument('--threshold', type=float, default=.5)
    parser.add_argument('--output', default='resultats_energie_pic_fixe_koch_2', help='Dossier des anciens resultats pour --replot uniquement ; aucun fichier ecrit.')
    parser.add_argument('--solver', choices=['standard'], default='standard', help='Solveur du code d origine : processing.solve_helmholtz.')
    parser.add_argument('--replot', action='store_true', help='Re-tracer le graphe depuis le CSV existant sans recalculer.')
    args = vars(parser.parse_args())
    if args.pop('replot'):
        replot(args['output'], args['material'])
    else:
        run(**args)

