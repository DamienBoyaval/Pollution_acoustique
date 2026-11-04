"""Compare plusieurs quantites de chi sur le mur plat, Koch 1 et Koch 2."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import test_energie_pic_fixe as plat
import test_energie_pic_fixe_koch as koch1
import test_energie_pic_fixe_koch_2 as koch2
import material_alpha
import geometrie


FORMS = {'plat': plat, 'koch1': koch1, 'koch2': koch2}
LABELS = {'plat': 'Mur plat', 'koch1': 'Koch niveau 1', 'koch2': 'Koch niveau 2'}


def save_plot(ax, path, title):
    ax.set(xlabel='Frequence (Hz)', ylabel='Energie discrete', title=title)
    ax.set_yscale('log')
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=160)
    plt.close(ax.figure)


def replot(output):
    """Regenere tous les graphes de comparaison sans resoudre Helmholtz."""
    out = Path(output)
    config = json.loads((out / 'configuration.json').read_text(encoding='utf-8'))
    records = []
    for case in config['cases']:
        folder = out / case['folder']
        data = np.genfromtxt(folder / 'energies_pic_fixe.csv', delimiter=',', names=True)
        saved = json.loads((folder / 'configuration.json').read_text(encoding='utf-8'))
        records.append(dict(case, data=data, saved=saved))
    summaries = []
    for kind, column in [('binaire', 'energy_fixed_binary'), ('continu', 'energy_fixed')]:
        for form in FORMS:
            _, ax = plt.subplots(figsize=(11, 5))
            subset = [r for r in records if r['form'] == form]
            for r in subset:
                ax.plot(r['data']['frequency_hz'], r['data'][column], label=f"Q={r['quantity']:.4g}")
            ax.plot(subset[0]['data']['frequency_hz'], subset[0]['data']['energy_full'],
                    '--', color='black', alpha=.6, label='100 % absorbant (quantite differente)')
            save_plot(ax, out / f'{form}_plusieurs_chi_{kind}.png',
                      f'{LABELS[form]} : plusieurs chi {kind}s fixes')

        # Une comparaison equitable des formes pour CHAQUE budget de materiau.
        for qi, quantity in enumerate(config['quantities']):
            _, ax = plt.subplots(figsize=(11, 5))
            for r in records:
                if r['quantity_index'] == qi:
                    ax.plot(r['data']['frequency_hz'], r['data'][column], label=LABELS[r['form']])
            save_plot(ax, out / f'comparaison_Q{qi:02d}_{kind}.png',
                      f'Meme quantite Q={quantity:.4g} : chi {kind}s fixes')

        # Meilleur budget par forme, parmi les seuls candidats calcules.
        # Les quantites peuvent differer : ne pas appeler cela une comparaison a budget egal.
        _, ax = plt.subplots(figsize=(11, 5))
        best = []
        for form in FORMS:
            subset = [r for r in records if r['form'] == form]
            winner = min(subset, key=lambda r: float(r['data'][column].max()))
            best.append(winner)
            ax.plot(winner['data']['frequency_hz'], winner['data'][column],
                    label=f"{LABELS[form]} ; Q={winner['quantity']:.4g}")
            summaries.append(dict(form=form, kind=kind, quantity=winner['quantity'],
                                  max_energy=float(winner['data'][column].max()),
                                  folder=winner['folder']))
        save_plot(ax, out / f'meilleurs_chi_{kind}.png',
                  f'Meilleur chi {kind} par forme : minimum du pic sur la bande\n'
                  'Quantites potentiellement differentes, indiquees dans la legende')

        fig, axes = plt.subplots(1, 3, figsize=(13, 5))
        for ax, r in zip(axes, best):
            with np.load(out / r['folder'] / 'chi_fixe.npz') as archive:
                domain = archive['domain']
                chi = archive['chi_fixed_binary' if kind == 'binaire' else 'chi_fixed']
                import _env
                yy, xx = np.where(domain == _env.NODE_ROBIN)
                artist = ax.scatter(xx / config['N'], yy / config['N'], c=chi[yy, xx],
                                    vmin=0, vmax=1, cmap='viridis', s=9)
                ax.set(title=f"{LABELS[r['form']]} ; Q={r['quantity']:.4g}",
                       xlabel='x (m)', ylabel='y (m)')
                ax.invert_yaxis()
                ax.set_aspect('equal')
        fig.colorbar(artist, ax=list(axes), label='chi', shrink=.7)
        fig.suptitle(f'Repartition des meilleurs chi {kind}s (parmi les candidats)')
        fig.savefig(out / f'repartition_meilleurs_chi_{kind}.png', dpi=160, bbox_inches='tight')
        plt.close(fig)

    with (out / 'meilleurs_chi.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    with (out / 'bilan_tous_chi.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['form', 'quantity', 'fraction', 'binary_quantity', 'target_hz',
                         'max_initial', 'max_continuous', 'max_binary', 'max_full'])
        for r in records:
            writer.writerow([r['form'], r['quantity'], r['saved']['fraction'],
                             r['saved']['binary_quantity'], r['saved']['target_frequency_hz'],
                             *[float(r['data'][key].max()) for key in
                               ['energy_initial', 'energy_fixed', 'energy_fixed_binary', 'energy_full']]])
    return summaries


def run(quantities=(.1, .2, .3, .4, .5), fmin=500., fmax=3000., points=None,
        N=None, max_iter=30, material='ROCK_WOOL', speed=300.,
        output='resultats_comparaison_chi_formes'):
    if not np.isfinite([fmin, fmax]).all() or not 0 < fmin < fmax:
        raise ValueError('Exiger 0 < fmin < fmax.')
    quantities = np.asarray(quantities, dtype=float)
    if quantities.ndim != 1 or not quantities.size or not np.isfinite(quantities).all() or np.any(quantities <= 0):
        raise ValueError('Les quantites doivent etre positives et finies.')
    if max_iter < 0:
        raise ValueError('max_iter doit etre positif ou nul.')
    multiple = geometrie.N_MORCEAUX * 3**2
    minimum = max(50, int(np.ceil(10 * fmax / 340.)), geometrie.CELLULES_MIN_MOTIF * multiple)
    N = int(np.ceil(minimum / multiple) * multiple) if N is None else N
    if N < minimum or N % multiple:
        raise ValueError(f'N doit etre >= {minimum} et multiple de {multiple}.')
    # Arrondi commun au nombre entier de noeuds absorbants : meme Q en binaire aussi.
    counts = np.rint(quantities * N).astype(int)
    if np.any(counts <= 0) or np.any(counts > N) or len(np.unique(counts)) != len(counts):
        raise ValueError('Choisir des quantites distinctes, representables et <= 1 m sur le mur plat.')
    effective = counts / N
    out = Path(output).resolve()
    if (out / 'configuration.json').exists():
        raise FileExistsError('Dossier deja utilise : choisir --output ou utiliser --replot.')
    out.mkdir(parents=True, exist_ok=True)
    config = dict(N=N, requested_quantities=quantities.tolist(), quantities=effective.tolist(),
                  fmin=fmin, fmax=fmax, points=points, max_iter=max_iter, material=material,
                  speed_kmh=speed, quantity_definition='Q=h*sum(chi), L=1 m',
                  selection='minimum of sampled maximum energy across tested quantities', cases=[])
    print(f'Maillage commun N={N}. Quantites effectives : {effective}', flush=True)
    for qi, quantity in enumerate(effective):
        for form, module in FORMS.items():
            folder = f'{form}_Q{qi:02d}'
            module.run(fmin=fmin, fmax=fmax, points=points, N=N, max_iter=max_iter,
                       material=material, speed=speed, quantity=float(quantity), output=str(out / folder))
            config['cases'].append(dict(form=form, quantity_index=qi, quantity=float(quantity), folder=folder))
    (out / 'configuration.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    result = replot(out)
    print(f'Graphes et bilans : {out}', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quantities', type=float, nargs='+', default=[.1, .2, .3, .4, .5])
    parser.add_argument('--fmin', type=float, default=500.)
    parser.add_argument('--fmax', type=float, default=3000.)
    parser.add_argument('--points', type=int, default=None)
    parser.add_argument('--N', type=int)
    parser.add_argument('--max-iter', type=int, default=30)
    parser.add_argument('--material', choices=material_alpha.MATERIALS, default='ROCK_WOOL')
    parser.add_argument('--speed', type=float, default=300.)
    parser.add_argument('--output', default='resultats_comparaison_chi_formes')
    parser.add_argument('--replot', action='store_true')
    args = vars(parser.parse_args())
    if args.pop('replot'):
        replot(args['output'])
    else:
        run(**args)
