"""Koch niveau 3, avec 30 % de la paroi absorbante.

Les figures sont affichees a la fin du calcul, sans enregistrer de fichiers.

Lancer : python test_energie_pic_fixe_koch_3.py
Essai court : python test_energie_pic_fixe_koch_3.py --points 3 --max-iter 1

La contrainte est sum(chi)=0.3*nombre_de_points_Robin, comme pour les
scripts Koch 1 et 2 par defaut. La binarisation arrondit le nombre de
points absorbants a l'entier le plus proche. La quantite totale de
materiau augmente donc avec la longueur discrete du bord.
"""
import argparse

import material_alpha
import train_source
import test_energie_pic_fixe_koch as koch


OUTPUT = 'resultats_energie_pic_fixe_koch_3'


def run(fmin=train_source.FREQUENCY_MIN_HZ, fmax=train_source.FREQUENCY_MAX_HZ,
        points=None, N=None, max_iter=30, material='ROCK_WOOL', speed=300.,
        output=OUTPUT, solver='standard'):
    """Optimise un seul mur Koch 3, a fraction absorbante fixee a 30 %."""
    return koch.run(fmin=fmin, fmax=fmax, points=points, N=N,
                    max_iter=max_iter, material=material, level=3,
                    fraction=.3, quantity=None, speed=speed, output=output, solver=solver)


def replot(output=OUTPUT, material=None):
    return koch.replot(output, material)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fmin', type=float, default=train_source.FREQUENCY_MIN_HZ)
    parser.add_argument('--fmax', type=float, default=train_source.FREQUENCY_MAX_HZ)
    parser.add_argument('--points', type=int, help='Toutes les frequences par defaut.')
    parser.add_argument('--N', type=int, help='Multiple de 108, au moins 216 ; automatique par defaut.')
    parser.add_argument('--max-iter', type=int, default=30)
    parser.add_argument('--material', choices=material_alpha.MATERIALS, default='ROCK_WOOL')
    parser.add_argument('--speed', type=float, default=300.)
    parser.add_argument('--output', default=OUTPUT, help='Dossier des anciens resultats pour --replot uniquement ; aucun fichier ecrit.')
    parser.add_argument('--solver', choices=['standard'], default='standard')
    parser.add_argument('--replot', action='store_true')
    args = vars(parser.parse_args())
    if args.pop('replot'):
        replot(args['output'], args['material'])
    else:
        run(**args)
