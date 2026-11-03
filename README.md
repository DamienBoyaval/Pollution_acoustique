# EI Pollution Acoustique : Helmholtz et optimisation

Le code reprend le squelette et le motif fractal du professeur : deux boucles
while, gradient adjoint, projection sur les contraintes, augmentation du pas
par 1.1 et reduction par 2. L'arret sur stabilisation de chi utilise 1e-6.

## Programmes a lancer

- `demo_control_polycopie2024.py` : comparaison des deux materiaux sur les
  1000 pulsations fournies, entre 500 et 4000 Hz. Mode par defaut :
  `comparaison`. Pour retrouver un seul materiau/frequence, choisir
  `mode = 'un_materiau'` dans ce fichier.
- `essai_source_gaussienne.py` : essai a 500 Hz, onde plane au bord puis
  ajout d'une gaussienne dans le volume, sans changer les conditions aux bords.
- `essai_niveaux_fractals.py` : niveaux 0, 1 et 2, les deux materiaux et
  les deux sources, avant/apres optimisation, a quantite totale identique.
  Par defaut : 500 Hz, N=64. Voir RESULTATS_NIVEAUX.md pour les essais N=128.
- `essai_fractal_resonateur.py` : quatre geometries (plat, fractale, resonateur, mix), deux frequences cibles puis diagnostic des pics 500-900 Hz.
- `plot_pics_hybrides.py` : retracer le balayage spectral local a partir des densites sauvegardees.
- `etudes.py` : etudes parametrables historiques, voir son aide `--help`.

La source par defaut est l'onde plane imposee en haut (`f_dir[0,:]=1`).
`f=0` dans le volume ne signifie pas que le probleme n'est pas excite.
Seul l'essai gaussien ajoute un second membre volumique non nul.

## Donnees et conventions

`data_alpha/` contient les tables du commit
817b56c73548e2a6f75d8b4021a228555eead5cd de la branche mauri : 1000 couples
(omega,Alpha) par materiau, de 500 a 4000 Hz, avec erreurs d'ajustement.
Les anciennes tables de 100 points 250-4000 Hz ne sont plus la reference.
La porosite/resistivite/tortuosite sont dans `references_mauri/compute_alpha.py`.
Ce fichier de reference ne doit pas etre relance pour ecraser les tables :
ses parametres de generation publies ne correspondent pas a leur grille.

Avec l'echelle de longueur L du domaine normalise :
k_solveur=omega*L/c et Alpha_solveur=Alpha_fourni*L. Valeurs par defaut :
L=1 m et c=340 m/s, comme pour le calcul des coefficients du collegue.
Le maillage est controle pour conserver au moins dix points par longueur
d'onde. Le calcul complet de 1000 frequences est plus long qu'un essai a 500 Hz.

## Documents et resultats

- COMPARAISON_MATERIAUX.md : methode de comparaison spectrale et limites.
- README_GAUSSIENNE.md : configuration de la source gaussienne.
- README_NIVEAUX_FRACTALS.md : essais des geometries et quantite commune.
- README_HYBRIDE.md et RESULTATS_HYBRIDE.md : protocole du mix et valeurs observees.
- CONCLUSIONS_PROVISOIRES.md : choix candidats et limites.
- RESULTATS_NIVEAUX.md : resultats effectifs N=64 et N=128 et interpretation.

Les programmes regenerent leurs figures, CSV, JSON et densites NPZ dans
leurs dossiers de resultats. Ces sorties sont locales et ignorees par Git ;
relancer les scripts correspondants pour les obtenir. Les tables de materiaux
dans data_alpha/ et les documents de reference sont conserves dans le depot.

## Verification

```text
python -m unittest -v test_validation.py test_material_alpha.py test_comparaison_materiaux.py test_gaussienne.py test_niveaux_fractals.py test_hybride.py
```

Ces tests verifient notamment le gradient adjoint contre differences finies,
la projection, les quantites, l'identite avec le solveur du professeur,
la superposition des sources et la conservation des donnees aux bords.
La convergence en maillage et l'optimalite globale ne sont pas affirmees.
