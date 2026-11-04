# Koch 1, 2 et 3 : 30 % de chaque paroi

Lancer le niveau souhaite depuis le dossier du projet :

```powershell
python test_energie_pic_fixe_koch.py
python test_energie_pic_fixe_koch_2.py
python test_energie_pic_fixe_koch_3.py
```

Les trois scripts utilisent par defaut une fraction absorbante de 0.3 :
`sum(chi) = 0.3 * nombre_de_points_Robin`.
Pour le chi binaire, le nombre de points absorbants est arrondi a l'entier
le plus proche de 30 % des points de la paroi. La quantite totale utilisee
augmente avec la longueur de la paroi.

Le script niveau 3 fixe ce niveau et cette fraction. Il reprend le solveur,
l'optimisation et les graphes du script Koch commun, avec un maillage
automatique de 216 mailles sur la largeur pour la bande par defaut.

Chaque script optimise au pic initial puis evalue le meme mur continu
et binaire sur toute la bande 500-3000 Hz (714 frequences disponibles),
avec la laine de roche et 30 iterations maximum par defaut.
Les fichiers du niveau 3 sont dans `resultats_energie_pic_fixe_koch_3`.

Essai court :

```powershell
python test_energie_pic_fixe_koch_3.py --points 3 --max-iter 1 --output tmp/essai_koch3
```

Pour retracer les figures existantes sans relancer le calcul :

```powershell
python test_energie_pic_fixe_koch_3.py --replot
```

Le programme `comparaison_chi_formes.py` reste un protocole distinct a
quantites Q communes. Utiliser les trois commandes ci-dessus pour le
protocole a 30 % de chaque paroi ; ne pas ajouter `--quantity` aux scripts
Koch 1 et 2, car cette option remplace la contrainte de fraction.
