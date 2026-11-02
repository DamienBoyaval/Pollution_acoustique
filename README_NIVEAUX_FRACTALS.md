# Essais des niveaux de fractale

Lancer `essai_niveaux_fractals.py` pour comparer les niveaux 0, 1 et 2
du generateur du professeur, les deux materiaux et les deux excitations :
onde plane au bord, puis onde plane au bord avec une gaussienne ajoutee
dans le volume. Les donnees aux bords restent celles du professeur pour
chaque geometrie. Le bord Robin change de forme selon le niveau.

Par defaut, on etudie la premiere frequence des tables : 500 Hz, avec N=64,
L=1 m, c=340 m/s, gaussienne d'amplitude 1, de centre normalise (0.5,0.25)
et sigma 0.08. Les 100 iterations maximales et le seuil de variation de chi
1e-6 sont conserves. Il ne s'agit pas d'un balayage de toute la bande.

La comparaison conserve la meme quantite totale pour tous les niveaux,
definie par Q = L*h*sum(chi) = 0.4. Ce comptage nodal suit la convention
discrete du professeur, sans nouvelle quadrature de longueur du bord.
Cela correspond a des fractions de couverture differentes : environ 0.4,
0.2 et 0.1 aux niveaux 0, 1 et 2. Conserver la meme fraction aurait ajoute
du materiau en augmentant le niveau et biaise l'interpretation.
Le motif initial du professeur est projete sur cette contrainte commune.

Pour chaque cas, on calcule l'energie avant et apres optimisation de chi.
Les frequences utilisees sont identiques pour tous les cas. S'il y en a
plusieurs, une seule densite est optimisee pour leur energie moyenne.
La gaussienne represente une excitation supplementaire, pas une amelioration
du materiau : une baisse par interference n'est pas un rendement d'absorption.

Sorties dans `resultats_niveaux_fractals/` : CSV, bilan JSON, graphiques des
energies et des geometries, densites et historiques NPZ. L'aire discrete et
l'energie rapportee a cette aire sont indiquees pour ne pas confondre un
changement de taille du domaine avec un changement de performance.
Le motif a une aire theorique constante, mais les aires de grille different
legerement avec le nombre de points de bord. Le classement peut dependre
du maillage, de la frequence et du nombre d'iterations ; il ne garantit pas
un minimum global ou un resultat valable sur toute la plage 500-4000 Hz.

Exemples :

```text
python essai_niveaux_fractals.py
python essai_niveaux_fractals.py --N 128 --output resultats_raffinement_N128
python essai_niveaux_fractals.py --indices 0 143 428 714 999
python -m unittest -v test_niveaux_fractals.py
```

Le maillage automatique satisfait les dix points par longueur d'onde et
prevoit au moins quatre cellules par plus petit segment du motif. Avec des
frequences montant a 4000 Hz et un niveau maximal 2, il choisit N=240.
L'exemple multifrequence est donc plus couteux que le test a 500 Hz.

Reutilisation : `comparaison_materiaux.make_problem`, son evaluateur et sa
descente, `essai_source_gaussienne.add_gaussian`, la projection et le
generateur fractal du professeur. Les fonctions d'elements finis des anciens
projets utilisent d'autres bords ; leurs conditions ne sont pas transposees
dans ces essais. Les mesures d'aire et de geometrie sont conservees comme
controles de comparaison, comme dans l'ancien projet.
