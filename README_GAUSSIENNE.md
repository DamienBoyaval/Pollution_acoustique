# Essai separe de source gaussienne

Lancer `essai_source_gaussienne.py` pour comparer a repartition de materiau
fixee : le cas actuel avec onde plane au bord, puis ce meme cas auquel on
ajoute une source gaussienne dans le volume. Les types et valeurs des
conditions aux bords restent identiques : notamment f_dir[0,:]=1.
Il ne s'agit donc pas d'une gaussienne seule ni d'une modification de la
condition de Dirichlet. Seul le second membre interieur f change.

La source ajoutee est

```text
f_g(x,y) = amplitude * exp(-((x-x0)^2+(y-y0)^2)/(2*sigma^2))
```

Elle vaut zero hors des noeuds interieurs. Les coordonnees sont celles du
domaine normalise du professeur. Par defaut : amplitude 1, centre (0.5,0.25),
sigma 0.08, N=50, niveau 0, indice de frequence 0 (500 Hz dans les nouvelles
tables), L=1 m et c=340 m/s. L'amplitude est un second membre numerique,
sans calibration de puissance physique. Les energies ne sont pas en dB.

```text
python essai_source_gaussienne.py
python essai_source_gaussienne.py --amplitude 10 --sigma 0.1 --center 0.5 0.25
python -m unittest -v test_gaussienne.py
```

Les deux materiaux sont testes. Les energies et differences de champ sont
affichees, puis sauvegardees avec les figures et tableaux dans
`resultats_source_gaussienne/`. La densite chi reste fixe : ce test n'effectue
pas d'optimisation et ne remplace pas la comparaison spectrale complete.
Une modification de l'energie apres ajout de source est un changement
d'excitation, pas une amelioration des performances du materiau.

Tests executes : conservation exacte des tableaux de bord et de chi,
gaussienne uniquement dans le volume, residu du systeme lineaire et
superposition (difference entre les deux champs egale a la reponse de la
gaussienne seule avec conditions homogenes). Le programme principal et les
conditions aux bords du projet ne sont pas modifies par cet ajout.
