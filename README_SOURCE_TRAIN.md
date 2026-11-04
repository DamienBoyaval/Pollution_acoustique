# Source gaussienne dependante de la frequence

Dans `demo_control_polycopie2024.py`, mode `un_materiau`, le second membre
volumique est maintenant `f(x,y;nu) = A_roul(nu,V) G_roul(x,y) + A_aero(nu,V) G_aero(x,y)`.
`nu = omega_physical/(2*pi)` est la frequence en Hz choisie par `frequency_index`.
Le coefficient du materiau et le nombre d'onde utilisent la meme frequence.
La binarisation de chi reste appliquee apres l'optimisation.

## Appui bibliographique et limites des donnees

- **Benjamin Cotte (2008)**, *Propagation acoustique en milieu exterieur complexe :
  problemes specifiques au ferroviaire dans le contexte des trains a grande vitesse*,
  Ecole Centrale de Lyon, section 3.1.3 et annexe D.1, pp. 189-190.
  [These](https://acoustique.ec-lyon.fr/publi/cotte_thesis.pdf),
  [notice HAL](https://theses.hal.science/tel-00660494).
  Le modele distingue les familles de sources et associe un spectre de reference
  et un exposant de vitesse a chaque famille. L'equation D.1 donne
  `L(V) = L(Vref) + 10*n*log10(V/Vref)`.
  Les valeurs spectrales detaillees sont renvoyees au rapport Charbonnel (2002) :
  elles n'ont pas ete recuperees et ne sont pas pretendument reproduites ici.
- **M. Priour (2005)**, *Influence des effets couples de la meteorologie et du
  relief sur la propagation acoustique : utilisation d'une methode d'elements
  finis de frontiere et validation experimentale*, Ecole Centrale de Lyon,
  chapitre 1, p. 28.
  [These](https://acoustique.ec-lyon.fr/publi/priour_thesis.pdf).
  Les bandes d'octave 500, 1000 et 2000 Hz sont soulignees pour le bruit
  des transports. Cela motive la zone d'etude, sans fournir le spectre exact
  de chacune des deux sources du present programme.
- Complement aux theses : **FRA (2005)**, *High-Speed Ground Transportation
  Noise and Vibration Impact Assessment*, section 2.2.2, p. 2-12.
  [Rapport](https://railroads.dot.gov/sites/fra.dot.gov/files/fra_net/1218/final_nv.pdf).
  Les lois typiques en vitesse sont de l'ordre de 30 log10(V) pour le roulement
  et de 60 a 70 log10(V) pour l'aerodynamique. On conserve ici les exposants
  idealises 3 et 6 deja utilises par le projet.

## Adaptation choisie pour ce code

Faute de spectres mesures, on choisit une enveloppe lisse de poids quadratique :

`S(nu) = exp(-0.5 * (log2(nu/nu_c)/b)**2)`

Puis l'amplitude du forcage est :

`A(nu,V) = gain * (V/Vref)**(n/2) * sqrt(S(nu))`.

La racine carree est essentielle : multiplier une amplitude par A multiplie
le carre de la reponse par A**2, a frequence et geometrie fixees pour une
source isolee. On ne multiplie pas directement l'amplitude par un poids
d'energie, et on n'ajoute pas de sinus temporel dans ce solveur harmonique.

**Les enveloppes log-gaussiennes, leurs centres et largeurs sont nos hypotheses
de scenario ; ce ne sont ni des equations tirees des theses, ni des donnees
mesurees, ni un ajustement valide sur un TGV particulier.**

| Parametre | Roulement | Aerodynamique |
|---|---:|---:|
| Centre de l'enveloppe (Hz) | 2250 | 1425 |
| Largeur b en octaves | 1 | 1 |
| Exposant quadratique n | 3 | 6 |
| Gain au centre, a Vref | 1 | 1 |

Les deux centres distincts permettent un essai de sensibilite du placement de
chi a la repartition du bruit entre les deux positions. Les centres sont les milieux arithmetiques des bandes 2000-2500 Hz (roues)
et 1250-1600 Hz (contribution aerodynamique de la motrice) observees sur TGV Duplex :
Mellet et al. (2006), section 2.1, https://doi.org/10.1016/j.jsv.2005.08.069.
Ce choix ne calibre ni les amplitudes ni les largeurs. Le premier pic de roulement
(voie, 800-1000 Hz) n'est pas represente par une enveloppe distincte.
Vref=300 km/h est une normalisation, pas une vitesse prouvee d'egalite des bruits.
Les centres ne se deplacent pas avec V dans cette approximation.

Les gaussiennes spatiales ont les centres normalises (x,y)=(0.9,0.5) et
(0.5,0.5). Leur sigma=0.03 est fixe, independamment de la frequence et du
maillage. Pour N=50, cela reproduit les positions et la largeur precedentes.
Le forcage est nul hors de l'interieur et les donnees aux bords sont conservees.

Ce sont des amplitudes relatives du second membre, pas des pressions calibrees
en Pa ni des niveaux en dB a 1 metre. Une source volumique 2D ne reproduit pas
directement un spectre de pression de source ponctuelle 3D. Les poids S ne sont
pas des niveaux par tiers d'octave ni une densite spectrale en Pa**2/Hz.
La propagation et les resonances peuvent produire des pics de champ ailleurs
qu'aux centres des enveloppes.

La somme des deux forcages reste coherente, a phase commune, comme dans le
programme initial. Ce n'est pas une somme statistique de bruits independants.
Pour celle-ci, il faudrait resoudre les deux sources separement et sommer
leurs energies et gradients. Ni le mouvement, ni le Doppler, ni le rayonnement
dipolaire du pantographe ne sont modelises ici.

## Utilisation et verification

Changer `frequency_index` pour choisir une des 1000 frequences du materiau ;
modifier `spectral_parameters` pour essayer d'autres centres, largeurs ou gains.
Le controle des dix points par longueur d'onde reste actif : augmenter N
pour les hautes frequences (par exemple N=240 pour atteindre 4000 Hz a L=1 m).
Le mode `comparaison` et les autres experiences gardent leurs sources propres.

Tests : `python -m unittest -v test_train_source.py`.
Les anciens resultats du dossier n'ont pas ete recalcules avec cette source.

## Comparaison de l'energie avant/apres suivant la frequence

`python test_energie_frequence.py` selectionne 500 frequences parmi les lignes des tables entre 500 et
3000 Hz, sans interpolation d'Alpha, avec 30 iterations maximum par frequence, la laine de roche et une
fraction initiale de 30 %. Le maillage est choisi pour toute la plage.
Le programme recalcule les sources du train et Alpha a chaque frequence.

Test court :
`python test_energie_frequence.py --fmax 600 --points 3 --N 50 --max-iter 8`

Les figures s'affichent ensemble dans des fenetres Matplotlib a la fin du calcul.
Aucun dossier ni fichier de resultats (CSV, NPZ, JSON, journal ou image) n'est cree.
Ce comportement s'applique aussi a `test_energie_pic_fixe.py`,
`test_energie_pic_fixe_koch.py`, `test_energie_pic_fixe_koch_2.py` et
`test_energie_pic_fixe_koch_3.py`. Fermer les figures termine l'affichage.
Les anciens resultats restent en place. `--replot --output DOSSIER` permet
encore de les lire et d'afficher leurs graphiques, sans enregistrer de fichiers.

Echantillonnage reduit pour un calcul plus court :
`python test_energie_frequence.py --fmax 3000 --points 40`

Chaque point repart du meme chi initial et optimise son propre chi ; il ne
s'agit pas du spectre d'une seule frontiere optimisee puis fixee. La reference
est la distribution initiale projetee a 30 %, pas un mur entierement rigide.
La premiere figure affiche les energies initiale, continue optimisee, binaire
et entierement absorbante. Chaque frequence a son propre chi optimise. L'historique normalise J/J0
n'est pas confondu avec l'energie absolue. La quantite binaire peut changer.

Les points traces sont les frequences exactes selectionnees dans les tables ;
les lignes entre ces points ne prouvent pas l'absence de pics intermediaires.

Les bornes par defaut et centres spectraux sont centralises dans train_source.py.
La selection conserve les frequences exactes des tables : la derniere frequence
inferieure a 3000 Hz est environ 2997,998 Hz. Le maillage automatique est N=96
pour cette bande au niveau 0. Les anciens CSV/NPZ ne sont pas recalcules :
leurs configurations historiques restent celles du calcul d'origine.

## Performance sur toute la bande du chi choisi au pic initial

Apres les optimisations independantes, le programme selectionne la frequence
ou l'energie INITIALE est maximale sur la grille exploree. Il reprend le chi
continu optimise a cette frequence et sa version binaire. Ces deux repartitions
restent ensuite fixes : seul le nombre d'onde, Alpha et la source changent avec
la frequence. Aucune nouvelle optimisation n'est faite dans ce second balayage.

La seconde figure compare ce chi fixe avec la repartition initiale et la paroi
entierement absorbante ; cible en pointilles et maxima echantillonnes en etoiles.
Elle est affichee sans enregistrer les donnees ni l'image.

`--replot` retrace aussi ce graphique si son CSV existe. `--points 5` permet un
essai court ; sans cette option, 500 frequences des tables sont utilisees.
Le calcul dense conserve jusqu'a 30 iterations pour chacune des 500 frequences
et peut etre long. Les traits relient les calculs, sans lissage artificiel ; des
resonances plus etroites que le pas peuvent encore echapper a l'echantillonnage.
Le seuillage binaire conserve sa limitation : il peut changer la quantite de
materiau et augmenter l'energie a certaines frequences.
