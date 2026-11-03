# Resultats du mix fractale + resonateur

Les quatre geometries sont effectivement construites et testees : bord plat,
fractale de niveau 1, bord plat avec resonateur lateral, et fractale avec ce
meme resonateur. Les deux materiaux et les deux sources sont compares a
quantite Q=L*h*sum(chi)=0.4 identique. Les nouvelles parois de cavite sont
rigides ; seul le bord Robin porte le materiau absorbant.

## Optimisation sur deux frequences

Les densites sont optimisees pour la moyenne de l'energie TOTALE aux deux
frequences 500 et 612.112112 Hz, avec N=128, L=1 m et c=340 m/s. Le
resonateur a une estimation cavite-col de 612.215 Hz, sans correction
d'extremites ; ce n'est pas une frequence couplee mesuree.

Energie moyenne finale dans la zone principale, onde plane au bord :

| Geometrie | Laine : energie principale | Caoutchouc : energie principale |
|---|---:|---:|
| Plat | 0.791968 | 2.157511 |
| Fractale niveau 1 | 0.754885 | 1.831846 |
| Plat + resonateur | 0.762754 | 2.315551 |
| Fractale + resonateur | 0.730616 | 1.881485 |

Le mix en laine donne la plus petite valeur sur ces deux frequences. Le
gain sur ce seul objectif ne suffit pas pour conclure sur les pics de bruit
dans toute une plage. Tous ces essais atteignent la limite des 100 pas ;
aucun optimum global n'est affirme. L'essai gaussien reste tres proche dans
cette configuration ; sa source n'est pas calibree sur un train reel.

## Verification des pics restants

Un second calcul utilise les memes densites initiales et finales FIXEES et
les 115 frequences exactes disponibles de 500 a 899.399399 Hz. Il ne refait
pas une optimisation independante a chaque frequence. Les pics locaux et
le maximum echantillonnes sont indiques sur les courbes et dans `pics.csv`.

Maximum final de l'energie principale sur cette grille locale :

| Geometrie | Pic laine | Pic caoutchouc |
|---|---:|---:|
| Plat | 2.885681 | 21.124192 |
| Fractale niveau 1 | 3.612575 | 33.276292 |
| Plat + resonateur | 2.542759 | 22.131106 |
| Fractale + resonateur | 3.921305 | 18.062098 |

La laine donne des pics plus bas que le caoutchouc pour les quatre formes
dans CE diagnostic local. Le meilleur pic obtenu est celui du bord plat
avec resonateur en laine, environ 11.88 % plus bas que le bord plat en
laine. Le mix fractal ne bat pas ces references sur le critere du pic restant,
malgre son meilleur resultat sur les deux frequences d'optimisation.

C'est pourquoi il ne faut pas annoncer que le mix est le meilleur produit.
Les choix de frequences cibles doivent etre reexamines a partir des vrais
pics observes. Le diagnostic local ne remplace pas le balayage 500-4000 Hz.

## Maillage et limites

Le col comporte quatre intervalles a N=128 et huit a N=256. Une verification
des etats INITIAUX, sans optimisation, a ete faite a N=256 sur les deux
frequences. L'ecart relatif maximal sur l'energie principale initiale est
de 7.40 %. Ce controle ne valide pas la convergence des pics
spectraux ou des optimisations. Il faut encore raffiner le maillage et
les frequences pres des pics avant une conclusion definitive.

L'energie dans la cavite est separee de celle dans la zone principale.
Un stockage d'energie dans la cavite ne doit pas etre presente comme une
dissipation. Les pertes visqueuses/thermiques du col ne sont pas modelisees.
Les energies sont des integrales de pression normalisee au carre, pas des
niveaux en dB mesures chez un recepteur.

Les CSV/JSON et figures sont dans `resultats_fractal_resonateur/`,
`verification_maillage_N256_sans_optimisation/` et `resultats_pics_hybrides/`.
Les NPZ locaux conservent les champs et densites pour refaire les calculs.
