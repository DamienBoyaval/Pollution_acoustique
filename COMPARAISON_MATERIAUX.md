# Comparaison automatique : laine de roche et caoutchouc recycle

Lancer `demo_control_polycopie2024.py` comme avant : son mode par defaut est
maintenant `mode = 'comparaison'`. Il appelle `comparaison_materiaux.py`.
Pour revenir au test d'un seul materiau, choisir `mode = 'un_materiau'`.

Les tables sont celles du commit 817b56c73548e2a6f75d8b4021a228555eead5cd
de la branche mauri : 1000 pulsations de 500 a 4000 Hz, pas 3.5035035 Hz.
Elles remplacent les anciennes tables de 100 points de 250 a 4000 Hz.
La reference `compute_alpha.py` et les informations du commit sont conservees
dans `references_mauri/`. Ne pas la lancer pour regenerer automatiquement les
tables : son parametrage publie utilise encore 10 points de 250 a 4000 Hz et
ne reproduit donc pas la grille des tables du commit.

Le coefficient est celui du modele du collegue, avec mu_0=1 et c_0=340 m/s.
Le code utilise donc c=340 m/s, k=omega*L/c et Alpha_num=Alpha_fourni*L.
L=1 m est l'echelle physique du domaine, pas une epaisseur du materiau.
La porosite, resistivite et tortuosite publiees sont respectivement
(0.96,32000,1.02) pour la laine et (0.42,18000,1.95) pour le caoutchouc.
Les coefficients sont repris sans recalibrage independant.

## Comparaison

- Meme geometrie, meme grille, meme onde plane imposee en haut (f_dir=1).
  f=0 dans le volume est conserve, conformement au choix de l'utilisateur.
- Meme densite initiale, meme quantite et memes parametres de descente.
- Balayage initial des deux materiaux ; mur entierement absorbant ajoute
  uniquement comme reference (sa quantite est differente).
- Union des trois principaux pics initiaux de chaque materiau : memes
  frequences cibles pour les deux optimisations, avec poids egaux.
- Deux boucles while, projection, pas multiplie par 1.1 si accepte et divise
  par 2 si refuse, 100 iterations maximales, stabilisation chi sous 1e-6.
- Nouveau balayage de toute la grille pour detecter les pics restants,
  y compris ceux qui n'etaient pas dans la liste initiale des cibles.
- Classement selon le maximum d'energie FINAL echantillonne, et indication
  de l'energie moyenne. Un pourcentage de baisse seul ne departage pas les
  materiaux. L'objectif optimise est une moyenne sur les frequences cibles,
  pas le maximum spectral ; il ne garantit pas que chaque pic diminue.

Sorties : `resultats_comparaison/comparaison.csv`, les deux spectres CSV,
`comparaison_energies.png`, les densites et historiques NPZ et la configuration
JSON. Le classement est limite a cette source, cette geometrie, cette
quantite et cette grille de frequences ; aucun optimum global n'est affirme.

Le maillage est choisi automatiquement pour conserver au moins dix points
par longueur d'onde a la frequence maximale. Avec L=1 m et 4000 Hz, N=240.
Le balayage complet de 1000 frequences et les optimisations prennent plus
de temps que le precedent test a une seule frequence. La progression et
les valeurs numeriques sont affichees pendant les calculs.

Les termes fixes sont assembles une seule fois ; les points non interieurs
elimines ont des valeurs nulles dans le solveur d'origine. La meme
factorisation creuse est utilisee pour le champ direct et l'adjoint. Les
equations discretes et la derivee du professeur sont conservees et comparees
au solveur d'origine dans les tests. Dans `partition_domain`, un ensemble
auxiliaire accelere seulement les tests d'appartenance des points visites ;
la liste et l'ordre de parcours du professeur sont conserves.

Test rapide (sous-plage uniquement, pas un classement global 500-4000 Hz) :

```text
python comparaison_materiaux.py --N 16 --indices 0 1 2 --max-iter 5 --output test_comparaison
python -m unittest -v test_comparaison_materiaux.py
```
