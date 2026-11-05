# Calculs complémentaires au sujet EI

La présentation existante est conservée et complétée par `complements_sujet_EI.tex`. Les six figures jointes sont copiées dans `figures_presentation`. Aucun nouveau calcul acoustique long n'a été lancé pour cette synthèse : les chiffres des cycles sont recalculés à partir du CSV existant.

## Résultats à retenir

- Figures jointes, niveau 1 : somme réduite de 21,0 % par seuil et de 18,8 % par tri, mais maximum augmenté d'environ 7,3 % et 13,1 %.
- Niveau 2 : somme réduite de 42,2 % par seuil et de 32,1 % par tri ; maximum réduit d'environ 66,4 % et 53,2 %.
- Six cycles archivés : le cycle 4 minimise la somme (baisse de 41,33 %) ; le cycle 3 minimise le maximum (environ 86,2 % sous le maximum initial). Les critères ne sélectionnent pas la même paroi.
- La quantité après seuil des figures jointes n'est pas connue. Un classement énergétique ne suffit pas pour conclure à budget de matériau identique.
- Le JSON des cycles indique binarisation 1/seuil, tandis que le code actuel appelle `chi_binarize_2`. Des légendes du code parlent encore de seuil. Les résultats archivés doivent être identifiés par leur configuration historique, et non par le code actuel. Ne pas écraser ce dossier lors d'une relance.

## Priorité 1 : optimiser simultanément les pics du train (développement)

Objectif : une seule densité pour 706,7067, 1043,0430 et 1719,2192 Hz, puis ajout des pics restants, notamment autour de 1715,7 Hz. Ces valeurs sont des lignes des tables, pas des fréquences propres exactes.

Réutiliser les mécanismes multifréquences de `comparaison_materiaux.py` ou `etudes.py`, mais avec `train_source.py` et les conditions homogènes du scénario train. Ne pas comparer directement les énergies d'une source au bord avec celles du train.

Pour chaque itération : résoudre les états et adjoints, sommer les gradients avec les mêmes poids que l'objectif, projeter sur le budget, accepter selon l'objectif total. Comparer deux objectifs : moyenne quadraturée sur la bande et approximation du maximum. Après binarisation, rebalayer toute la bande avec le même chi. Conserver le meilleur mur selon le critère retenu, pas seulement le dernier cycle.

Livrables utiles : spectres initiaux/continus/binaires/100 %, chi, somme, intégrale quadraturée, maximum, fréquence du maximum, quantité, gradient projeté et historique. Les poids spectraux du train sont déjà dans la source : ne pas les appliquer une deuxième fois sans changer explicitement le modèle.

## Priorité 2 : vérifier maillage et résolution fréquentielle

Pour la paroi du cycle 3 et celle du cycle 4, comparer N=128 et N=256 en conservant la même répartition physique et la même quantité. Le transfert de chi vers le nouveau maillage doit être défini ; réoptimiser séparément sur chaque grille mélange erreur de discrétisation et différence d'optimisation.

Raffiner les fréquences autour de chaque pic, notamment 700–715, 1035–1050 et 1705–1730 Hz. Le pas actuel d'environ 3,5035 Hz peut rater le sommet. Le minimum de dix points par longueur d'onde ne suffit pas pour valider le maximum ou les détails de la paroi.

Les tables actuelles ne permettent pas un raffinement arbitraire sans interpolation ou nouveau calcul d'alpha. Choisir et documenter l'une des méthodes, vérifier sa précision et la passivité (partie imaginaire de signe attendu). Ne pas relancer sans adaptation `references_mauri/compute_alpha.py` : sa grille par défaut ne reproduit pas les tables archivées.

Critère proposé : comparer les valeurs ET fréquences des pics avec une tolérance fixée avant le calcul (par exemple 5 % sur les valeurs), puis raffiner davantage si elle n'est pas atteinte.

## Priorité 3 : quantité et comparaison des formes

Le sujet demande quelle quantité est optimale : ce point n'est pas répondu par un seul essai à 30 %. Faire varier un budget total Q commun, par exemple Q=0,1 ; 0,2 ; 0,3 ; 0,4 ; 0,6 m dans la convention 2D du code, si ces valeurs sont admissibles pour toutes les parois. Il s'agit d'une longueur couverte équivalente à épaisseur fixée, pas d'une masse en kg.

Comparer j=0,1,2 à Q fixé, avec plusieurs initialisations et une binarisation par tri. Pour comparer les matériaux, garder également source, géométrie, fréquences, maillage et budget identiques. Tracer le front de compromis quantité / maximum / intégrale fréquentielle. Sans pénalisation du coût ou seuil de performance, « quantité optimale » ne désigne pas un choix industriel unique.

Le script unifié peut déjà tester un pic et sa paroi fixe. Exemple à exécuter depuis le dossier du projet dans un environnement Python disposant des dépendances du projet :

```python
import json
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import test_energie_pic_fixe_unifie as test

# Une configuration par dossier ; ne pas écraser les archives.
out = Path('resultats_nouveaux/j2_Q03_N128')
out.mkdir(parents=True, exist_ok=False)
rows, config = test.run(fmin=500, fmax=3000, points=None, N=128,
                       max_iter=80, material='ROCK_WOOL', j=2,
                       fraction=0.3, quantity=0.3, speed=300., threshold=0.5)
with (out/'energies.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(out/'configuration.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
for index, number in enumerate(plt.get_fignums(), 1):
    plt.figure(number).savefig(out/f'figure_{index}.png', dpi=180)
plt.close('all')
```

Cet exemple exploite l'interface existante ; il n'a pas été exécuté ici. Il exporte les résultats retournés, mais le script unifié ne retourne pas les densités : ajouter leur export NPZ pour rendre la comparaison en maillage reproductible. Vérifier aussi que la fraction réellement transmise au tri correspond à Q, et non à une valeur par défaut de 30 %.

Pour relancer seulement des cycles avec le code ACTUEL et un nouveau dossier :

```python
from test_optimisation_pics_successifs import run
run(j=2, N=128, points=None, cycles=6, max_iter=80,
    output_dir='resultats_cycles_tri_nouvel_essai', show=False)
```

Ce n'est pas une reproduction garantie de l'archive marquée « binarisation 1 ». Corriger au préalable les métadonnées et les légendes de la méthode utilisée. Dans le code actuel, l'appel au tri utilise sa fraction par défaut de 0,3 : pour une étude de quantité, transmettre explicitement la fraction cible ou `fraction=None` selon la politique choisie.

## Priorité 4 : ouverture et sources indépendantes (développement)

- Garder la source volumique et remplacer le bord supérieur de Dirichlet par Sommerfeld `du/dn - i*k*u = 0`, en vérifiant les signes avec la convention temporelle. Comparer J, pertes sur Robin et flux sortant à même source et même chi. Une PML ou un domaine agrandi permettrait ensuite de tester l'erreur du bord absorbant.
- Résoudre séparément roulement et aérodynamique, puis sommer les énergies et gradients pour un modèle incohérent. Comparer à l'hypothèse actuelle de phase commune. Tester quelques vitesses et déplacements des centres avant d'optimiser une paroi robuste.
- Contrôler l'adjoint par différences finies directionnelles avec plusieurs tailles de perturbation et une direction préservant la quantité. Le test doit porter sur l'opérateur réellement assemblé, y compris l'élimination Robin.

## Questions du sujet encore incomplètes

1. État de l'art externe et positionnement concurrentiel : les références locales donnent un point de départ, pas une revue exhaustive.
2. Startup, organisation de l'équipe et coûts : à compléter par les auteurs.
3. Gradient de forme complet comparant bord ouvert/réfléchissant : bilan énergétique et formule de transport présentés, mais conditions aux limites différentiées et algorithme de forme non développés.
4. Paroi plane entièrement couverte sous la source train et comparaison homogène de toutes les formes : à consolider.
5. Optimisation simultanée des pics du train, quantité optimale, convergence des pics et validation expérimentale : à faire.

## Compilation et présentation

Compiler deux fois `presentation_optimisation_binarisation.tex` avec pdflatex depuis le dossier du projet. Il inclut `complements_sujet_EI.tex` et `figures_presentation/table_cycles.tex` ; conserver aussi les images du dossier des cycles. Les annexes donnent les dérivations et les six figures jointes. Pour 20 minutes, sélectionner environ 18–22 diapositives principales et garder les autres pour les questions.
