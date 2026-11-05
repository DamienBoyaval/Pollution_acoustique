from pathlib import Path
import csv, json, shutil

root = Path(__file__).resolve().parents[1]
out = root / 'figures_presentation'
for level in (1, 2):
    for num in (1, 2, 3):
        shutil.copy2(Path('C:/Users/damie/Downloads/resultats') / f'j{level}' / f'Figure_{num}.png', out / f'j{level}_Figure_{num}.png')
rows = list(csv.DictReader((root / 'resultats_optimisation_pics_successifs/energies.csv').open()))
summary = []
for cycle in range(7):
    e = [float(r[f'energy_cycle_{cycle}']) for r in rows]
    k = max(range(len(e)), key=e.__getitem__)
    summary.append(dict(cycle=cycle, energy_sum=sum(e), maximum=max(e), frequency_hz=float(rows[k]['frequency_hz'])))
assert len(rows) == 714
assert min(summary[1:], key=lambda r:r['energy_sum'])['cycle'] == 4
assert min(summary[1:], key=lambda r:r['maximum'])['cycle'] == 3
archived = json.loads((root / 'resultats_optimisation_pics_successifs/bilan.json').read_text())
assert abs(summary[4]['energy_sum'] - archived['best_result']['energy_sum']) < 1e-14
def fr(x, digits=4):
    return f'{x:.{digits}f}'.replace('.', ',')
lines = [r'\begin{tabular}{rrrr}', r'\toprule Cycle & Somme ($10^{-4}$) & Maximum ($10^{-5}$) & Fréquence du max. (Hz)\\', r'\midrule']
for r in summary:
    lines.append(f"{r['cycle']} & {fr(r['energy_sum']*1e4)} & {fr(r['maximum']*1e5)} & {fr(r['frequency_hz'],1)}" + r'\\')
lines += [r'\bottomrule', r'\end{tabular}']
(out/'table_cycles.tex').write_text('\n'.join(lines), encoding='utf-8')
(out/'verification_cycles.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
