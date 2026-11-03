# Fractale + resonateur de Helmholtz

Lancer `essai_fractal_resonateur.py`. Quatre geometries sont comparees :
bord plat, fractale du professeur, resonateur sur bord plat, et fractale
avec le meme resonateur lateral. Par defaut : niveau fractal 1, N=128,
deux materiaux, onde plane au bord puis ajout de gaussienne dans le volume,
quantite Q=L*h*sum(chi)=0.4 et 100 iterations maximales.

## Construction et comparaison

La construction cavite+col et l'estimation de frequence sont reprises de
`generate_q1_mesh` / `design_resonator` de l'ancien projet. Leur geometrie
est adaptee a une branche sur le mur lateral droit, afin de garder le motif
fractal du professeur en bas. Le col ouvre une portion du mur rigide vers
la cavite ajoutee ; les nouvelles parois sont en Neumann homogene.
L'onde plane reste imposee en haut ; la region Robin et la quantite de
materiau sont conservees entre les cas avec et sans resonateur.

Le resonateur de l'ancien projet visait environ 60 Hz, hors des nouvelles
tables Alpha (500-4000 Hz). Les dimensions sont donc parametrees dans
`geometrie_hybride.build_problem` et choisies ici plus petites :

- col de longueur 0.0625 L et de largeur 0.03125 L ;
- cavite carree 0.0625 L par 0.0625 L ;
- centre a y/L=0.375, a l'ecart de la source et du bord fractal.

Avec L=1 m et c=340 m/s, la formule de l'ancien projet
f_H=c/(2*pi)*sqrt(largeur_col/(aire_cavite*longueur_col)) donne environ
612.215 Hz. Cette estimation ignore les corrections d'extremites, les
effets de forme et le couplage au domaine principal. Elle ne mesure pas
la resonance reelle de l'ensemble. Aucun reglage optimal n'est affirme.
La largeur du col est representee par 4 intervalles a N=128 et 8 a N=256.
Un maillage trop grossier ou non aligne est refuse, sans elargir le col
silencieusement (ce qui changerait sa frequence).

Par defaut, les frequences utilisees sont 500 Hz et la ligne de la table
la plus proche de l'estimation (environ 612.112 Hz). Les coefficients
Alpha du collegue sont pris aux memes lignes pour tous les cas.
L=1 m et c=340 m/s, k=omega*L/c, Alpha_num=Alpha_fourni*L.

La descente reste celle utilisee dans les essais precedents : deux while,
projection, pas *1.1 ou /2, seuil sur la variation de chi 1e-6.
Une seule densite est optimisee pour l'energie moyenne du domaine complet
sur les frequences choisies. Les deux materiaux utilisent exactement
les memes frequences, source, parametres et quantite.

## Energie dans la zone principale

Ajouter une cavite ajoute aussi un volume d'air. Le programme indique
separement l'energie totale, l'energie dans la region principale d'origine,
et celle de l'air ajoute (col, ouverture et cavite). Les masques de la
region principale sont identiques pour chaque paire avec/sans resonateur.
Une diminution dans la zone principale accompagnee d'une augmentation
dans la cavite est une redistribution d'energie, pas une dissipation
demontree. Le resonateur est ici rigide et sans pertes internes : seuls
les termes Robin modelisent le materiau absorbant. Les pertes visqueuses
et thermiques du col ne sont pas ajoutees arbitrairement.

## Sorties et commandes

`resultats_fractal_resonateur/` contient les geometries avec zoom sur le col,
les comparaisons d'energies totale/principale, les CSV et JSON, les densites,
historiques et champs NPZ. Les graphiques et tableaux concernent les
frequences indiquees dans la configuration, pas toute la bande 500-4000 Hz.
Apres une optimisation, le lancement en ligne de commande effectue aussi
un balayage local de 500 a 900 Hz (115 frequences exactes disponibles), avec
les densites initiale et optimisee maintenues fixes. Les maxima locaux
echantillonnes et le maximum echantillonne sont marques sur les courbes,
et les valeurs sont exportees dans un tableau. Cela ne mesure pas avec
certitude une resonance propre du resonateur, car les modes du domaine
principal sont couples a ceux de la cavite.

Le balayage supplementaire est dans `resultats_fractal_resonateur/pics_spectraux/`.
`pics_main.png` montre avant/apres, et `pics_principaux_optimises.png` compare
les pics restants sans les grandes valeurs initiales qui peuvent masquer
les differences. Utiliser `--no-spectral-plot` pour ne pas refaire ce balayage.
Le diagnostic spectral deja execute se trouve aussi dans
`resultats_pics_hybrides/`. Il concerne l'onde plane seule ; une option permet
de refaire le meme graphique pour l'essai gaussien.

```text
python essai_fractal_resonateur.py
python essai_fractal_resonateur.py --level 2
python essai_fractal_resonateur.py --indices 0 29 32 57 86 114
python essai_fractal_resonateur.py --N 256 --max-iter 0 --output verification_maillage_N256_sans_optimisation
python -m unittest -v test_hybride.py
python plot_pics_hybrides.py
python plot_pics_hybrides.py --source plane_plus_gaussian
```

L'essai N=256 avec max-iter=0 verifie les champs INITIAUX sur un maillage
plus fin, pas la convergence des optimisations. Une validation de la
frequence de resonance et une optimisation de la taille du col/cavite
demanderaient des essais supplementaires. L'optimalite globale n'est pas
garantie par les descentes locales.
