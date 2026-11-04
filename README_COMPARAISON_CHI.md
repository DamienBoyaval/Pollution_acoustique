# Comparer plusieurs chi : mur plat, Koch 1 et Koch 2

Depuis le dossier du projet :

```powershell
python comparaison_chi_formes.py
```

Le calcul utilise la source train, la laine de roche et toutes les frequences
disponibles entre 500 et 3000 Hz. Il teste cinq quantites de materiau
Q = 0.1, 0.2, 0.3, 0.4 et 0.5 sur chacune des trois formes (15 optimisations).
Chaque chi est optimise au pic de son spectre initial, puis conserve FIXE
pour son evaluation sur toute la bande. Ce n'est pas une optimisation
simultanee de toute la bande, ni une recherche de l'optimum global.

## Choisir les quantites et le cout du calcul

```powershell
python comparaison_chi_formes.py --quantities 0.15 0.3 0.45 --points 50 --max-iter 30 --output resultats_comparaison_chi_rapide
```

Sans `--points`, les 714 frequences sont utilisees. Une grille de 50 points
est plus rapide mais peut manquer des pics et changer le classement.
`--material RECYCLED_RUBBER` permet une autre campagne.

Q est la longueur discrete de materiau Q = h * somme(chi), pour L = 1 m.
Ce n'est PAS une fraction identique de chaque paroi : la fraction est
Q / (h * nombre_de_points_Robin), donc elle diminue lorsque le bord s'allonge.
Le maillage N est commun aux trois formes (108 par defaut sur 500-3000 Hz).
Pour rendre la quantite identique aussi pour les chi binaires, le nombre de
points absorbants est arrondi a round(Q*N). Les quantites demandees et
effectivement utilisees sont sauvegardees dans configuration.json.
Il s'agit de la mesure discrete du projet, pas d'un volume 3D de materiau.

## Graphes dans resultats_comparaison_chi_formes

- `plat_plusieurs_chi_binaire.png`, `koch1_plusieurs_chi_binaire.png`,
  `koch2_plusieurs_chi_binaire.png` : une courbe par quantite, et la reference
  entierement absorbante. Cette reference utilise une quantite differente.
- `comparaison_Q00_binaire.png`, etc. : les trois formes au MEME budget.
- `meilleurs_chi_binaire.png` : meilleur candidat de chaque forme, choisi
  par le plus petit maximum d'energie echantillonne sur la bande.
  Les quantites de ces trois gagnants peuvent differer et sont indiquees.
- `repartition_meilleurs_chi_binaire.png` : positions du materiau pour les
  gagnants, avec chi=0 reflechissant et chi=1 absorbant.
- Les memes figures suffixees `continu` pour les chi relaxes entre 0 et 1.
- `bilan_tous_chi.csv` et `meilleurs_chi.csv` : valeurs de comparaison.
- Un sous-dossier par forme et quantite conserve les courbes detaillees,
  la configuration et les chi dans `chi_fixe.npz`.

Les energies sont absolues dans le modele normalise, pas des decibels.
Les classements portent sur les candidats et les frequences testes.

Pour retracer les figures sans refaire les calculs :

```powershell
python comparaison_chi_formes.py --replot
```

Pour refaire une campagne, donner un nouveau `--output` ; le script refuse
de remplacer une campagne ayant deja son fichier de configuration.

Les trois scripts individuels acceptent aussi `--quantity`, prioritaire
sur `--fraction`. Dans ce mode individuel, l'arrondi binaire peut differer
si les maillages ne sont pas communs : utiliser le script de comparaison
pour le protocole a quantite egale.
