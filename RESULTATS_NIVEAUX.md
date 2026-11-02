# Essais de fractales a 500 Hz

Trois niveaux du motif du professeur (0, 1 et 2), deux materiaux et deux
excitations ont ete testes. Chaque optimisation part du motif initial du
professeur projete sur une quantite commune Q=L*h*sum(chi)=0.4. Le volume
de materiau n'augmente donc pas avec la longueur du bord fractal.

Les sources sont l'onde plane imposee au bord superieur, puis cette meme
onde plane avec une source gaussienne volumique supplementaire. Les donnees
aux bords restent identiques entre les deux essais d'une meme geometrie.

Parametres : 500 Hz (premiere ligne des nouvelles tables de mauri), L=1 m,
c=340 m/s, gaussienne d'amplitude 1, centre (0.5,0.25), sigma=0.08.
Le niveau 0 est plat. Les autres niveaux reutilisent exactement le motif du
professeur, sans remplacer la geometrie par celle d'un autre projet.

## Resultats avec N=128

Energies obtenues apres au plus 100 iterations, onde plane seule :

| Niveau | Laine de roche | Caoutchouc recycle |
|---|---:|---:|
| 0 | 0.658125 | 0.497943 |
| 1 | 0.529185 | 0.682221 |
| 2 | 0.553116 | 2.024062 |

Avec onde plane + gaussienne :

| Niveau | Laine de roche | Caoutchouc recycle |
|---|---:|---:|
| 0 | 0.654999 | 0.495803 |
| 1 | 0.528442 | 0.682873 |
| 2 | 0.552051 | 2.025577 |

Le caoutchouc est meilleur que la laine pour le bord plat dans ces essais ;
la laine est meilleure que le caoutchouc pour les deux bords fractals.
Le meilleur cas obtenu parmi ces essais est le bord plat en caoutchouc,
pour chaque excitation prise separement. Le niveau 1 est le meilleur des
trois niveaux testes pour la laine. Augmenter la fractalite n'ameliore
donc pas automatiquement le resultat.

L'ajout de cette gaussienne change peu les energies. Avec deux sources
coherentes, des interferences peuvent produire une hausse ou une baisse
de l'energie ; cela ne constitue pas une mesure d'absorption du materiau.
L'amplitude de la gaussienne n'est pas calibree sur une source ferroviaire.

## Limites de la comparaison

Les deux maillages N=64 et N=128 donnent le meme classement par materiau a
chaque niveau, mais leurs energies finales changent d'environ 2 a 19 %.
La plus forte difference concerne le caoutchouc au niveau 2. On ne peut
donc pas annoncer une convergence en maillage ni des valeurs physiques
definitives. Les aires discretes et energies par unite d'aire sont aussi
enregistrees dans les CSV pour rendre visible l'effet de la discretisation.

Presque tous les essais atteignent la limite de 100 iterations. Au niveau 2,
avec N=128, le caoutchouc satisfait le seuil de variation de chi apres 76
iterations (onde plane) et 88 (avec gaussienne). Cela indique une stabilisation
de la densite, pas une preuve d'optimum global. Les differences entre les
deux maillages incluent aussi l'effet du parcours de l'optimisation.

Ces essais ne portent que sur 500 Hz. Ils ne remplacent pas le balayage de
500 a 4000 Hz et ne permettent pas de choisir definitivement un produit.

Les valeurs completes, conditions, quantites, criteres d'arret et densites
sont dans les dossiers `resultats_niveaux_fractals/` (N=64) et
`resultats_raffinement_N128/` (N=128). Les archives NPZ sont generees
localement par le script ; seuls les bilans legers et les figures sont
necessaires pour consulter les resultats dans Git.
