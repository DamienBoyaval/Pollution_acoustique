"""Pilotage temporaire : le programme d'origine reste inchangé."""
from pathlib import Path
import csv, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import test_energie_pic_fixe_unifie as test

out = Path('resultats_pic_fixe_seuil_j2')
out.mkdir(exist_ok=True)

# Le test d'origine calcule aussi le tri. Pendant cette seule exécution, le
# second appel renvoie le même seuil ; aucun résultat de tri n'est tracé.
test.demo.chi_binarize_2 = lambda chi, domain, threshold=.5, fraction=.3: test.demo.chi_binarize_1(chi, domain, threshold)
test.plot_fixed_results = lambda *args, **kwargs: None
test.plot_binary_comparison = lambda *args, **kwargs: None
test.plot_total_energy_barchart = lambda *args, **kwargs: None
test.plt.show = lambda: None

rows, config = test.run(fmin=500, fmax=3000, points=None, N=128,
                        max_iter=30, material='ROCK_WOOL', j=2,
                        fraction=.3, speed=300., threshold=.5)
assert all(np.isclose(r['energy_fixed_binary_1'],r['energy_fixed_binary_2']) for r in rows)
config['displayed_binarization'] = 'chi_binarize_1 (seuil 0.5); label chi_binarize'
config['runner_note'] = 'Temporary in-memory alias of chi_binarize_2 to chi_binarize_1; original source unchanged.'
with (out/'energies.csv').open('w',newline='',encoding='utf-8') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
(out/'configuration.json').write_text(json.dumps(config,indent=2),encoding='utf-8')

f=np.array([r['frequency_hz'] for r in rows]); target=rows[0]['target_frequency_hz']
series=[('energy_initial','Répartition initiale','#687386',':'),
        ('energy_fixed','Optimisation continue au pic','#1971c2','-'),
        ('energy_fixed_binary_1','chi_binarize (seuil 0,5)','#9333a8','--'),
        ('energy_full','Paroi couverte à 100 %','#2b8a3e','-')]
fig,ax=plt.subplots(figsize=(12,5.5),layout='constrained')
for key,label,color,style in series:
    y=np.array([r[key] for r in rows]); ax.semilogy(f,y,style,color=color,label=label)
    k=int(y.argmax());ax.scatter(f[k],y[k],color=color,marker='*',s=80,zorder=5)
ax.axvline(target,color='#555',ls=':',label=f'Pic ciblé : {target:.1f} Hz')
ax.set(xlabel='Fréquence (Hz)',ylabel='Énergie du champ (unités du modèle)',
       title='Source train - ROCK_WOOL - fractale niveau 2 : un seul mur fixé sur la bande')
ax.grid(alpha=.25);ax.legend(fontsize=8)
fig.savefig(out/'spectre_chi_binarize.png',dpi=180);plt.close(fig)

totals=[sum(r[key] for r in rows) for key,_,_,_ in series]
fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
bars=ax.bar(['Initiale','Continu','chi_binarize','100 %'],totals,
            color=[x[2] for x in series],edgecolor='#333')
for b,v in zip(bars,totals):ax.text(b.get_x()+b.get_width()/2,v*1.02,f'{v:.3e}',ha='center',fontsize=9)
ax.set_ylim(0,max(totals)*1.18);ax.set(ylabel='Somme des énergies échantillonnées (unités du modèle)',
    title=f'Mur fixé au pic de {target:.1f} Hz - {len(rows)} fréquences')
ax.grid(axis='y',alpha=.25)
fig.savefig(out/'somme_chi_binarize.png',dpi=180);plt.close(fig)
print('Résultats enregistrés dans',out.resolve(),flush=True)
