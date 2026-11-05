"""Optimiser successivement le pic du spectre de la paroi binaire courante.

Un cycle : maximum d'energie -> optimisation continue -> binarisation
-> nouveau spectre. Le chi binaire est le depart du cycle suivant.
La frequence cible est celle du maximum d'energie, pas la plus haute frequence.

La premiere optimisation utilise fraction. Les suivantes conservent la
quantite du chi binaire precedent pour ne pas modifier ce point de depart.
Le seuil chi_binarize_1 peut donc faire evoluer la quantite de materiau.
Cette strategie ne garantit pas une diminution du maximum sur toute la bande.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import csv
import json
from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np

import _env
import preprocessing as pre
import processing as pro
import material_alpha
import train_source
import demo_control_polycopie2024 as demo
from test_energie_pic_fixe_unifie import build_geometry, wall_label


def plot_boundary(domain, h, states, labels, title):
    """Zoom sur les noeuds de la paroi : gris = chi nul, rouge = chi egal a 1."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('materiau', ['#d9d9d9', '#c62828'])
    rows, cols = np.where(domain == _env.NODE_ROBIN)
    fig, axes = plt.subplots(1, len(states), figsize=(6 * len(states), 4),
                             squeeze=False, layout='constrained')
    for ax, chi, label in zip(axes[0], states, labels):
        points = ax.scatter(cols*h, rows*h, c=chi[rows, cols], cmap=cmap,
                            vmin=0, vmax=1, marker='s', s=16)
        ax.set(xlabel='x (m)', ylabel='y depuis le haut (m)',
               title=f'{label}\nMateriau : {100*chi[rows, cols].mean():.2f} %')
        ax.set_xlim(-2*h, (domain.shape[1]+1)*h)
        ax.set_ylim(rows.max()*h + .06, rows.min()*h - .06)
        ax.set_aspect('equal')
        ax.grid(alpha=.15)
    fig.colorbar(points, ax=list(axes[0]), label='Chi : 0 = sans materiau ; 1 = avec materiau',
                 shrink=.75)
    fig.suptitle(title + '\nParoi de Robin : chaque carre represente un noeud')
    return fig


def plot_results(frequencies, spectra, material, j, best_cycle):
    fig, ax = plt.subplots(figsize=(11, 5), layout='constrained')
    for number, energy in enumerate(spectra):
        label = 'Repartition initiale' if number == 0 else f'Binaire apres cycle {number}'
        line, = ax.semilogy(frequencies, energy, label=label)
        peak = int(np.argmax(energy))
        ax.scatter(frequencies[peak], energy[peak], color=line.get_color(), marker='*', s=70)
    ax.set(xlabel='Frequence (Hz)', ylabel='Energie absolue du champ')
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.suptitle(f'{material} - {wall_label(j)} - chi_binarize_1\n'
                 'Une paroi fixe par courbe ; etoiles : maxima echantillonnes')

    fig_max, ax_max = plt.subplots(figsize=(9, 4), layout='constrained')
    maxima = [float(np.max(energy)) for energy in spectra]
    ax_max.plot(range(len(spectra)), maxima, 'o-')
    for number, energy in enumerate(spectra):
        peak = int(np.argmax(energy))
        ax_max.annotate(f'{frequencies[peak]:.1f} Hz', (number, maxima[number]),
                        xytext=(0, 8), textcoords='offset points', ha='center', fontsize=8)
    ax_max.set(xlabel='Cycle (0 = repartition initiale)', ylabel='Maximum d energie sur la bande',
               title='Evolution du pire pic apres chaque binarisation')
    ax_max.set_xticks(range(len(spectra)))
    ax_max.margins(y=.2)
    ax_max.grid(alpha=.25)
    totals = [float(np.sum(energy)) for energy in spectra]
    fig_sum, ax_sum = plt.subplots(figsize=(10, 5), layout='constrained')
    labels = ['Initiale'] + [f'Cycle {i}' for i in range(1, len(spectra))]
    colors = ['#687386'] + ['#1971c2'] * (len(spectra) - 1)
    colors[best_cycle] = '#2b8a3e'
    bars = ax_sum.bar(labels, totals, color=colors)
    for bar, total in zip(bars, totals):
        ax_sum.annotate(f'{total:.4e}\n({100*total/totals[0]:.1f} % de l initiale)',
                        (bar.get_x()+bar.get_width()/2, total),
                        xytext=(0, 5), textcoords='offset points', ha='center', fontsize=9)
    ax_sum.set_ylim(0, max(totals)*1.25)
    ax_sum.set(ylabel='Somme des energies sur les frequences echantillonnees',
               title=f'Meilleure repartition binaire parmi les cycles : cycle {best_cycle} (vert)')
    ax_sum.grid(axis='y', alpha=.25)
    fig_compare, ax_compare = plt.subplots(figsize=(11, 5), layout='constrained')
    ax_compare.semilogy(frequencies, spectra[0], ':', color='#687386',
                        label='Repartition initiale')
    ax_compare.semilogy(frequencies, spectra[best_cycle], '-', color='#2b8a3e',
                        label=f'Meilleur chi binaire - cycle {best_cycle}')
    ax_compare.set(xlabel='Frequence (Hz)', ylabel='Energie absolue du champ',
                   title='Initiale et meilleur chi binaire selon la somme sur la bande')
    ax_compare.grid(alpha=.25)
    ax_compare.legend()

    fig_compare_sum, ax_compare_sum = plt.subplots(figsize=(8, 5), layout='constrained')
    selected_totals = [totals[0], totals[best_cycle]]
    bars = ax_compare_sum.bar(['Repartition initiale', f'Meilleur chi binaire\nCycle {best_cycle}'],
                              selected_totals, color=['#687386', '#2b8a3e'])
    for bar, total in zip(bars, selected_totals):
        ax_compare_sum.annotate(f'{total:.4e}\n({100*total/totals[0]:.1f} % de l initiale)',
                                (bar.get_x()+bar.get_width()/2, total),
                                xytext=(0, 5), textcoords='offset points', ha='center')
    ax_compare_sum.set_ylim(0, max(selected_totals)*1.25)
    ax_compare_sum.set(ylabel='Somme des energies sur les frequences echantillonnees',
                       title='Bilan : repartition initiale et meilleur chi binaire')
    ax_compare_sum.grid(axis='y', alpha=.25)
    return fig, fig_max, fig_sum, fig_compare, fig_compare_sum


def run(fmin=500., fmax=3000., points=None, N=None, j=2,
        material='ROCK_WOOL', fraction=.3, speed=300.,
        cycles=3, max_iter=30, threshold=.5,
        output_dir='resultats_optimisation_pics_successifs', show=True):
    """Binarisation par chi_binarize_1 uniquement. points=None : toute la table.

    cycles fixe le nombre d'optimisations successives. Les cycles sont effectues
    meme si le maximum augmente ou si le chi binaire reste identique.
    output_dir=None desactive l'enregistrement ; show=False masque les figures.
    """
    start = time.time()
    omega, alpha, _ = material_alpha.load_material(material)
    all_frequencies = omega / (2 * np.pi)
    indices = np.flatnonzero((all_frequencies >= fmin - 1e-8)
                             & (all_frequencies <= fmax + 1e-8))
    if points is not None:
        positions = np.round(np.linspace(0, len(indices) - 1, points)).astype(int)
        indices = indices[positions]
    frequencies = all_frequencies[indices]
    d, x, y, mesh_cells, M, h = build_geometry(j, N, frequencies[-1])
    N = d.shape[1]
    bp, ap, ad, bn, _, br = pre._set_coefficients_of_pde(M, N)
    _, fd, fn, fr = pre._set_rhs_of_pde(M, N)
    S = np.count_nonzero(d == _env.NODE_ROBIN)
    chi = demo.your_project_density(pre.set2zero(pre._set_chi(M, N, x, y), d), d, fraction)
    ks, alphas = material_alpha.solver_parameters(omega[indices], alpha[indices], 1., 340.)

    def source_at(position):
        source, _ = train_source.gaussian_train_source(d, h, frequencies[position], speed)
        return source

    def spectrum(density):
        energy = []
        for position, frequency in enumerate(frequencies):
            u = pro.solve_helmholtz(d, h, float(ks[position]), source_at(position),
                                   fd, fn, fr, bp, ap, ad, bn, br, alphas[position] * density)
            energy.append(float(demo.your_compute_objective_function(d, u, h)))
            if position == 0 or (position + 1) % 50 == 0 or position + 1 == len(frequencies):
                print(f'  Spectre {position + 1}/{len(frequencies)} : {frequency:.2f} Hz', flush=True)
        return np.array(energy)

    print(f'{wall_label(j)} ; N={mesh_cells} ; {len(frequencies)} frequences ; '
          f'{cycles} cycles ; chi_binarize_1', flush=True)
    def display_boundary(states, labels, title, filename):
        fig = plot_boundary(d, h, states, labels, title)
        if output_dir is not None:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            fig.savefig(out/f'{filename}.png', dpi=150)
        if show:
            plt.show(block=False)
            plt.pause(.05)
        else:
            plt.close(fig)

    display_boundary([chi], ['Repartition initiale'],
                     f'{wall_label(j)} - etat initial', 'frontiere_initiale')
    print('Calcul du spectre initial', flush=True)
    spectra = [spectrum(chi)]
    densities = [chi.copy()]
    continuous_densities = []
    history = []

    for cycle in range(1, cycles + 1):
        peak = int(np.argmax(spectra[-1]))
        fraction_start = float(chi.sum() / S)
        print(f'Cycle {cycle}/{cycles} : cible {frequencies[peak]:.2f} Hz ; '
              f'energie={spectra[-1][peak]:.6e} ; materiau={100*fraction_start:.2f} %', flush=True)
        chi_cont, optim_history, u_cont, _ = demo.your_optimization_procedure(
            d, h, float(ks[peak]), source_at(peak), fd, fn, fr, bp, ap, ad, bn, br,
            alphas[peak] * chi, complex(alphas[peak]), 5., chi.copy(),
            fraction_start, max_iter=max_iter)
        chi_binary = demo.chi_binarize_1(chi_cont, d, threshold)
        display_boundary(
            [chi, chi_cont, chi_binary],
            ['Depart du cycle', 'Apres optimisation continue', 'Apres seuil (depart suivant)'],
            f'Cycle {cycle} - optimisation a {frequencies[peak]:.2f} Hz',
            f'frontiere_cycle_{cycle:02d}')

        new_energy = spectrum(chi_binary)
        new_peak = int(np.argmax(new_energy))
        history.append(dict(
            cycle=cycle, target_frequency_hz=float(frequencies[peak]),
            energy_before=float(spectra[-1][peak]),
            energy_continuous_at_target=float(demo.your_compute_objective_function(d, u_cont, h)),
            energy_binary_at_target=float(new_energy[peak]),
            next_peak_frequency_hz=float(frequencies[new_peak]),
            max_energy_after=float(new_energy[new_peak]),
            fraction_before=fraction_start, fraction_after=float(chi_binary.sum()/S),
            accepted_iterations=len(optim_history)-1))
        print(f'  Apres binarisation : maximum={new_energy[new_peak]:.6e} '
              f'a {frequencies[new_peak]:.2f} Hz', flush=True)
        # Le spectre et le point de depart suivants correspondent au MEME chi binaire.
        chi = chi_binary.copy()
        spectra.append(new_energy)
        densities.append(chi.copy())
        continuous_densities.append(chi_cont.copy())

    # Comparer les parois binaires produites ; l'etat initial reste une reference.
    totals = np.sum(np.array(spectra), axis=1)
    best_cycle = int(np.argmin(totals[1:])) + 1 if history else 0
    best_chi = densities[best_cycle].copy()
    best_result = dict(cycle=best_cycle, criterion='sum of sampled energies',
                       energy_sum=float(totals[best_cycle]), initial_energy_sum=float(totals[0]),
                       reduction_percent=float(100*(1-totals[best_cycle]/totals[0])),
                       material_fraction=float(best_chi.sum()/S),
                       best_including_initial_cycle=int(np.argmin(totals)))
    display_boundary(
        [best_chi], [f'Chi binaire retenu - cycle {best_cycle}'],
        f'Plus faible somme parmi les cycles : {totals[best_cycle]:.4e}', 'chi_meilleur')
    print(f'Meilleur cycle binaire : {best_cycle} ; somme={totals[best_cycle]:.6e} ; '
          f'reduction par rapport a l initiale={best_result["reduction_percent"]:.2f} %', flush=True)
    if totals[0] < totals[best_cycle]:
        print('La repartition initiale reste meilleure que tous les cycles binaires.', flush=True)

    config = dict(material=material, level=j, N=mesh_cells, spacestep=h,
                  initial_fraction=fraction, speed_kmh=speed, cycles=cycles,
                  max_iter=max_iter, binarisation=1, threshold=threshold,
                  points=len(frequencies), elapsed_seconds=time.time()-start,
                  volume_policy='Preserve the current binary quantity at the next optimization')
    figures = plot_results(frequencies, spectra, material, j, best_cycle)
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out/'bilan.json').write_text(json.dumps(dict(configuration=config, cycles=history,
                                                    best_result=best_result), indent=2),
                                    encoding='utf-8')
        np.savez(out/'densites_et_spectres.npz', frequencies_hz=frequencies,
                 spectra=np.array(spectra), densities=np.array(densities),
                 continuous_densities=np.array(continuous_densities), domain=d,
                 best_chi=best_chi, best_cycle=best_cycle, energy_sums=totals)
        with (out/'energies.csv').open('w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['frequency_hz'] + [f'energy_cycle_{i}' for i in range(len(spectra))])
            writer.writerows(zip(frequencies, *spectra))
        for name, fig in zip(['spectres', 'maxima', 'sommes_energies',
                              'comparaison_initial_meilleur_spectre',
                              'comparaison_initial_meilleur_barres'], figures):
            fig.savefig(out/f'{name}.png', dpi=150)
    print(f'Calcul termine : {len(history)} cycles effectues. '
          f'Temps de calcul : {time.time()-start:.1f} s', flush=True)
    if show:
        print('Les calculs sont finis. Fermer les figures pour terminer le programme.', flush=True)
        plt.show()
    return frequencies, spectra, densities, history


if __name__ == '__main__':
    run(
        fmin=500., fmax=3000.,
        points=None,       # Toutes les frequences ; 10 pour un essai rapide.
        N=None,
        j=2,
        material='ROCK_WOOL',
        fraction=.3,      # Quantite initiale ; ensuite, quantite du binaire precedent.
        speed=300.,
        cycles=4,         # Nombre de pics optimises successivement.
        max_iter=50,
        threshold=.5,
    )

