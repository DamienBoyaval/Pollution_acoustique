"""Experience rapide unifiee : mur plat ou paroi de von Koch quadratique.

Le parametre j choisit la geometrie du mur :
    j = 0 : surface plane
    j = 1 : von Koch niveau 1
    j = 2 : von Koch niveau 2
    ...

Protocole en 4 etapes :
1. Balayage initial sur toutes les frequences : calcul de J(chi_0) et J(chi_full).
2. Detection du pic initial : identification automatique de la frequence de resonance maximale.
3. Optimisation ciblee : optimisation de chi uniquement a cette frequence de pic (max_iter iterations).
4. Evaluation spectrale du mur fixe : evaluation de chi fixe (continu et binaire) sur toute la bande.

La contrainte est sum(chi) = fraction * nombre_de_points_Robin (ou quantity si
elle est donnee). La binarisation arrondit le nombre de points absorbants a
l'entier le plus proche. Pour j >= 1, la quantite totale de materiau augmente
donc avec la longueur discrete du bord (utiliser quantity pour comparer les
geometries a quantite de materiau egale).
"""

import time
import matplotlib.pyplot as plt
import numpy as np

import _env
import preprocessing as pre
import processing as pro
import material_alpha
import train_source
import demo_control_polycopie2024 as demo


# ---------------------------------------------------------------------------
# Utilitaires lies a j
# ---------------------------------------------------------------------------
def wall_label(j):
    return 'Paroi plate' if j == 0 else f'Paroi de Koch niveau {j}'

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_fixed_results(rows, material, label):
    frequency = np.array([float(r['frequency_hz']) for r in rows])
    target = float(rows[0]['target_frequency_hz'])
    fig, ax = plt.subplots(figsize=(11, 5), layout='constrained')
    for key, name, color in [
        ('initial', 'Repartition initiale', '#687386'),
        ('fixed', 'Chi continu du pic, fixe sur toute la bande', '#1971c2'),
        ('fixed_binary', 'Chi binaire du pic, fixe sur toute la bande', '#be5c20'),
        ('full', 'Paroi entierement absorbante (100%)', '#2b8a3e')]:
        values = np.array([float(r['energy_' + key]) for r in rows])
        ax.semilogy(frequency, values, '-', label=name, color=color)
        peak = int(np.argmax(values))
        ax.scatter(frequency[peak], values[peak], color=color, marker='*', s=65, zorder=5)
    ax.axvline(target, color='#555555', linestyle=':', label=f'Cible initiale : {target:.1f} Hz')
    ax.set(xlabel='Frequence (Hz)', ylabel='Energie absolue du champ (unites du modele)')
    ax.grid(True, alpha=.25)
    ax.legend(fontsize=8)
    fig.suptitle(f'Source train - {material} - {label} : un seul chi optimise au pic de {target:.1f} Hz\n'
                 'Sans reoptimisation aux autres frequences ; etoiles : maxima echantillonnes')
    return fig


def plot_binary_comparison(rows, material, label):
    """Deuxieme figure : comparaison exclusive entre la repartition initiale et la paroi binaire fixe."""
    frequency = np.array([float(r['frequency_hz']) for r in rows])
    target = float(rows[0]['target_frequency_hz'])
    fig, ax = plt.subplots(figsize=(11, 5), layout='constrained')
    for key, name, color in [
        ('initial', 'Repartition initiale (reference 30%)', '#687386'),
        ('fixed_binary', 'Paroi binaire fixe (optimisee au pic)', '#be5c20')]:
        values = np.array([float(r['energy_' + key]) for r in rows])
        ax.semilogy(frequency, values, '-', label=name, color=color, linewidth=1.5)
        peak = int(np.argmax(values))
        ax.scatter(frequency[peak], values[peak], color=color, marker='*', s=80, zorder=5,
                   label=f'Max {name.split()[0]} : {values[peak]:.2e} a {frequency[peak]:.1f} Hz')
    ax.axvline(target, color='#2b8a3e', linestyle='--', linewidth=1.2, label=f'Pic cible optimise : {target:.1f} Hz')
    ax.set(xlabel='Frequence (Hz)', ylabel='Energie absolue du champ (unites du modele)')
    ax.grid(True, alpha=.25)
    ax.legend(fontsize=8.5, loc='upper right')
    fig.suptitle(f'Source train - {material} - {label} : Comparaison Initiale vs Paroi Binaire Fixe\n'
                 f'Mur optimise a {target:.1f} Hz puis binarise (30% exact de materiau) ; etoiles : maxima echantillonnes')
    return fig


def plot_total_energy_barchart(rows, material, label):
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
    for n, (bar, val) in enumerate(zip(bars, totals)):
        height = bar.get_height()
        rel_str = ' (100%)' if n == 0 else f' ({100*val/total_init:.1f}%)'
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
    fig.suptitle(f'Source train - {material} - {label} : Bilan Énergétique Global ({n_pts} fréquences)\n'
                 f'Mur fixé optimisé à {target:.1f} Hz ; référence entièrement absorbante à 100%',
                 fontsize=11.5, fontweight='bold')

    return fig


# ---------------------------------------------------------------------------
# Geometrie (entierement fournie par preprocessing.py)
# ---------------------------------------------------------------------------
# Nombre minimal de mailles par plus petit segment de Koch (longueur N / 4**j).
CELLULES_MIN_SEGMENT = 2


def build_geometry(j, N, f_max):
    """Retourne d (domaine), x, y (chemin de la paroi), mesh_cells, M, h.

    Tout passe par preprocessing._set_geometry_of_domain(M, N, j), avec M = 2N :
    j = 0 : mur plat (ligne N) ; j >= 1 : von Koch quadratique de niveau j
    (motif de rapport 1/4, donc N doit etre divisible par 4**j).
    """
    needed = int(np.ceil(10 * f_max / 340.))
    minimum = max(16, needed, CELLULES_MIN_SEGMENT * 4**j)
    if N is None:
        multiple = max(16, 4**j)
        N = int(np.ceil(max(50, minimum) / multiple) * multiple)
    M, h = 2 * N, 1. / N
    d, x, y, _, _ = pre._set_geometry_of_domain(M, N, j)
    return d, x, y, N, M, h

# ---------------------------------------------------------------------------
# Experience
# ---------------------------------------------------------------------------
def run(fmin=train_source.FREQUENCY_MIN_HZ, fmax=train_source.FREQUENCY_MAX_HZ,
        points=None, N=None, max_iter=30, material='ROCK_WOOL', j=0,
        fraction=.3, quantity=None, speed=300., threshold=0.5):
    """Optimise un seul mur au pic initial, puis l'evalue sur toute la bande.

    fmin, fmax : bande de frequences (Hz), bornes incluses dans les tables du materiau.
    points     : nombre de frequences echantillonnees (None = toutes).
    N          : mailles sur 1 m (None = automatique) ; doit etre divisible par 4**j,
                 avec au moins CELLULES_MIN_SEGMENT * 4**j mailles. M = 2N.
    max_iter   : iterations max de l'optimisation au pic.
    material   : un element de material_alpha.MATERIALS.
    j          : 0 = plat, 1 = Koch 1, 2 = Koch 2, 3 = Koch 3, ...
    fraction   : fraction de points Robin absorbants (contrainte sum(chi) = fraction * S).
    quantity   : Q = h * sum(chi) en metres ; prioritaire sur fraction (L = 1 m).
    speed      : vitesse du train (km/h).
    threshold  : conserve pour compatibilite (la binarisation se fait par rang).
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
    boundary_count = np.count_nonzero(d == _env.NODE_ROBIN)
    chi0 = demo.your_project_density(pre.set2zero(pre._set_chi(M, N, x, y), d), d, fraction)

    chi_full = np.zeros((M, N), dtype=float)
    chi_full[d == _env.NODE_ROBIN] = 1.0

    print('=' * 65)
    print(f'SIMULATION OPTIMISATION AU PIC INITIAL ({material})')
    print(f'Grille : {points} frequences ({frequencies[indices[0]]:.1f} a {frequencies[indices[-1]]:.1f} Hz)')
    print(f'Maillage : N={mesh_cells} mailles (grille {M}x{N} = {M*N} noeuds) ; max_iter={max_iter}')
    if j == 0:
        print(f'Paroi : plate ; pas h={h:.6g} m')
    else:
        print(f'Paroi : von Koch quadratique, niveau {j} ; pas h={h:.6g} m')
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
        geometry='plate' if j == 0 else 'koch', level=int(j), max_iter=max_iter,
        fraction=fraction, threshold=threshold, speed_kmh=speed,
        material_quantity=float(h * chi0.sum()),
        binary_quantity=float(h * chi_binary.sum()),
        length_m=1., sound_speed=340.,
        target_frequency_hz=peak_frequency, target_table_index=int(indices[peak_idx]),
        peak_initial_energy=float(peak_initial_energy),
        peak_optimized_energy=float(e_opt_peak),
        peak_binary_energy=float(e_bin_peak),
        total_time_seconds=float(time.time() - t_start),
        source='train_source defaults; coherent',
        points=points, frequencies_hz=frequencies[indices].tolist()
    )

    plot_fixed_results(rows, material, label)
    plot_binary_comparison(rows, material, label)
    plot_total_energy_barchart(rows, material, label)

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
    print(f'BILAN GLOBAL SUR TOUTE LA BANDE ({frequencies[indices[0]]:.0f} - {frequencies[indices[-1]]:.0f} Hz) - {label} :')
    print(f'1. Pic initial max        : {max_init:.4e} (a {f_max_init:.1f} Hz)')
    print(f'2. Pic continu fixe max   : {max_fixed:.4e} (a {f_max_fixed:.1f} Hz) -> Atténuation globale : -{100*(1-max_fixed/max_init):.1f} %')
    print(f'3. Pic binaire fixe max   : {max_fixed_bin:.4e} (a {f_max_bin:.1f} Hz) -> Atténuation globale : -{100*(1-max_fixed_bin/max_init):.1f} %')
    print(f'4. Reference 100% absorb. : {max_full:.4e}')
    print(f'Temps total d execution   : {time.time() - t_start:.2f} s')
    print('=' * 65)
    plt.show()
    return rows, config


if __name__ == '__main__':
    # ----------------------------------------------------------------------
    # Modifier ici les parametres de l'experience.
    # j : 0 = surface plane ; 1 = Koch niveau 1 ; 2 = Koch niveau 2 ; 3 = Koch niveau 3
    # ----------------------------------------------------------------------
    rows, config = run(
        fmin=train_source.FREQUENCY_MIN_HZ,   # Hz
        fmax=train_source.FREQUENCY_MAX_HZ,   # Hz
        points=None,          # None = toutes les frequences des tables ; ex. 50
        N=None,               # None = automatique
        max_iter=30,
        material='ROCK_WOOL', # ou 'RECYCLED_RUBBER' (voir material_alpha.MATERIALS)
        j=0,
        fraction=0.3,
        quantity=None,        # Q = h*sum(chi), prioritaire sur fraction
        speed=300.,           # km/h
        threshold=0.5,
    )
