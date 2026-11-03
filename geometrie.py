# -*- coding: utf-8 -*-
"""
Géométrie du domaine : forme de la paroi du bas (bord de Robin).

Types de paroi :
    'fractale'           : préfractale de Minkowski (8 segments de 1/4, aire conservée)
    'koch'               : courbe de Koch quadratique (5 segments de 1/3, une bosse vers l'extérieur)
    'resonateur'         : résonateur de Helmholtz (col + cavité carrée) sous le sol
    'resonateur_fractal' : le même résonateur, parois de la cavité en préfractale de Minkowski

Comme dans projet_v6, le domaine est d'abord un ensemble de petits carrés (on ajoute /
retire des carrés), puis on le transforme en grille de noeuds pour les différences finies :
    noeud dont les 4 carrés voisins sont fluides       -> NODE_INTERIOR
    noeud dont 1 à 3 carrés voisins sont fluides        -> bord (Dirichlet en haut,
                                                           Neumann sur les côtés, Robin ailleurs)
    noeud sans carré fluide autour                      -> NODE_COMPLEMENTARY

Convention : domain_omega[i, j], i = ligne (0 en haut, Dirichlet), j = colonne.
"""


# Python packages
import matplotlib.colors
import matplotlib.pyplot
import numpy


# MRG packages
import _env


# =====================================================================
# PARAMÈTRES
# =====================================================================

C_SON = 340.0                   # vitesse du son (m/s)
LARGEUR = 1.0                   # largeur de la pièce (m)
HAUTEUR = 1.0                   # hauteur de la pièce, au-dessus de la paroi (m)
POINTS_PAR_LONGUEUR_ONDE = 10   # résolution minimale demandée
N_MIN = 100                     # nombre minimal de points sur la largeur (précision)

# --- fractale / koch
N_MORCEAUX = 4                  # la paroi est coupée en 4 morceaux de fractale
CELLULES_MIN_MOTIF = 2          # plus petit segment de la fractale, en nombre de mailles

# --- résonateur de Helmholtz
W_COL = 0.04                    # largeur du col (m)
L_COL = 0.08                    # longueur du col (m)
CELLULES_COL = 4                # nombre minimal de mailles dans la largeur du col

MARGE = 2                       # mailles vides sous la géométrie
TYPES_BORD = ('fractale', 'koch', 'resonateur', 'resonateur_fractal')


# =====================================================================
# RÉSOLUTION : AU MOINS 10 POINTS PAR LONGUEUR D'ONDE
# =====================================================================

def longueur_onde(frequence):
    return C_SON / frequence


def points_par_longueur_onde(spacestep, frequence):
    return longueur_onde(frequence) / spacestep


def verifier_resolution(spacestep, frequence):
    """Erreur si le maillage a moins de POINTS_PAR_LONGUEUR_ONDE points par longueur d'onde."""
    n = points_par_longueur_onde(spacestep, frequence)
    if n < POINTS_PAR_LONGUEUR_ONDE:
        raise ValueError(f"{frequence:.0f} Hz : {n:.1f} points par longueur d'onde "
                         f"(il en faut au moins {POINTS_PAR_LONGUEUR_ONDE}), il faut raffiner le maillage")
    return n


def pas_maximal(frequence):
    """Plus grand pas d'espace qui respecte la résolution demandée."""
    return min(longueur_onde(frequence) / POINTS_PAR_LONGUEUR_ONDE, LARGEUR / (N_MIN - 1))


# =====================================================================
# CONSTRUCTION AVEC DES CARRÉS (coordonnées en nombre de mailles)
# =====================================================================
# Un carré (i, j) occupe [i, i+1] x [j, j+1], x vers la droite, y vers le haut.

def carres_entre(C1, C2):
    """Tous les carrés du rectangle de coins opposés C1 et C2."""
    i1, i2 = sorted([int(round(C1[0])), int(round(C2[0]))])
    j1, j2 = sorted([int(round(C1[1])), int(round(C2[1]))])
    return {(i, j) for i in range(i1, i2) for j in range(j1, j2)}


def minkowski_sur_segment(carres, A, B, niveau):
    """
    Remplace le côté [A, B] par une préfractale de Minkowski (le domaine est à gauche
    de A -> B). Chaque segment P -> Q devient 8 segments de longueur |PQ| / 4 :
    on AJOUTE le carré de la bosse qui sort et on RETIRE celui de la bosse qui rentre
    (même taille -> aire conservée).
    """
    ligne = [numpy.array(A, dtype=float), numpy.array(B, dtype=float)]
    for etape in range(niveau):
        nouvelle_ligne = [ligne[0]]
        for k in range(len(ligne) - 1):
            P = ligne[k]
            Q = ligne[k + 1]
            d = (Q - P) / 4
            n = numpy.array([d[1], -d[0]])        # normale extérieure (à droite de P -> Q)
            carres |= carres_entre(P + d, P + 2 * d + n)       # bosse qui sort : on ajoute
            carres -= carres_entre(P + 2 * d, P + 3 * d - n)   # bosse qui rentre : on retire
            nouvelle_ligne += [P + d, P + d + n, P + 2 * d + n, P + 2 * d,
                               P + 2 * d - n, P + 3 * d - n, P + 3 * d, Q]
        ligne = nouvelle_ligne
    return numpy.array(ligne)


def koch_sur_segment(carres, A, B, niveau):
    """
    Remplace le côté [A, B] par une courbe de Koch quadratique (le domaine est à gauche
    de A -> B). Chaque segment P -> Q devient 5 segments de longueur |PQ| / 3 :
    avant, extérieur, avant, retour, avant. On AJOUTE le carré de la bosse (l'aire augmente).
    """
    ligne = [numpy.array(A, dtype=float), numpy.array(B, dtype=float)]
    for etape in range(niveau):
        nouvelle_ligne = [ligne[0]]
        for k in range(len(ligne) - 1):
            P = ligne[k]
            Q = ligne[k + 1]
            d = (Q - P) / 3
            n = numpy.array([d[1], -d[0]])        # normale extérieure
            carres |= carres_entre(P + d, P + 2 * d + n)
            nouvelle_ligne += [P + d, P + d + n, P + 2 * d + n, P + 2 * d, Q]
        ligne = nouvelle_ligne
    return numpy.array(ligne)


def carres_vers_domaine(carres, nx, ny):
    """Ensemble de carrés -> matrice domain_omega (ny lignes, nx colonnes) des types de noeuds."""
    # fluide[i + 1, j + 1] = 1 si le carré (i, j) est fluide (une case vide autour)
    fluide = numpy.zeros((nx + 1, ny + 1), dtype=numpy.int64)
    for (i, j) in carres:
        if not (0 <= i < nx - 1 and 0 <= j < ny - 1):
            raise ValueError("la géométrie sort de la grille")
        fluide[i + 1, j + 1] = 1
    # nombre de carrés fluides autour de chaque noeud (ix, jy)
    nb = fluide[:-1, :-1] + fluide[1:, :-1] + fluide[:-1, 1:] + fluide[1:, 1:]
    ix, jy = numpy.meshgrid(numpy.arange(nx), numpy.arange(ny), indexing='ij')

    types = numpy.full((nx, ny), _env.NODE_COMPLEMENTARY, dtype=numpy.int64)
    types[nb == 4] = _env.NODE_INTERIOR
    bord = (nb > 0) & (nb < 4)
    types[bord] = _env.NODE_ROBIN
    types[bord & ((ix == 0) | (ix == nx - 1))] = _env.NODE_NEUMANN    # murs gauche et droit
    types[bord & (jy == ny - 1)] = _env.NODE_DIRICHLET                # haut (la source)

    # (ix, jy) -> (ligne, colonne), la ligne 0 est en haut
    return numpy.ascontiguousarray(types.T[::-1, :])


# =====================================================================
# RÉSONATEUR DE HELMHOLTZ
# =====================================================================

def frequence_helmholtz(a, b, w=W_COL, L=L_COL):
    """f0 = c / (2 pi) * sqrt( w / (A (L + 2 w)) ),  A = a b  (2D, correction de longueur 2 w)."""
    return C_SON / (2 * numpy.pi) * numpy.sqrt(w / (a * b * (L + 2 * w)))


def cote_cavite(frequence_resonateur, w=W_COL, L=L_COL):
    """Côté a de la cavité carrée qui donne f0 (formule inversée avec a = b)."""
    k0 = 2 * numpy.pi * frequence_resonateur / C_SON
    return numpy.sqrt(w / (k0 ** 2 * (L + 2 * w)))


# =====================================================================
# FONCTION PRINCIPALE
# =====================================================================

def creer_geometrie(type_bord, niveau, frequence_source, frequence_resonateur=None):
    """
    Construit le domaine : une pièce LARGEUR x HAUTEUR, source (Dirichlet) en haut, murs
    rigides (Neumann) à gauche et à droite, et en bas la paroi de Robin de forme type_bord.

    Paramètres :
        type_bord            : 'fractale', 'koch', 'resonateur' ou 'resonateur_fractal'
        niveau               : niveau de la fractale (ignoré pour 'resonateur')
        frequence_source     : fréquence de la source (Hz), la plus haute utilisée ; le
                               maillage a au moins POINTS_PAR_LONGUEUR_ONDE points par
                               longueur d'onde à cette fréquence
        frequence_resonateur : fréquence visée par le résonateur (Hz), seulement pour les
                               résonateurs

    Le pas est choisi pour que les segments de la fractale (ou le côté de la cavité)
    tombent sur un nombre entier de mailles.

    Renvoie :
        domain_omega : matrice (M, N) des types de noeuds (_env.NODE_...)
        M, N         : nombre de points selon y (lignes) et selon x (colonnes)
        spacestep    : pas d'espace (m)
        zone_paroi   : matrice (M, N) de booléens, noeuds intérieurs de la zone de la paroi
                       (fractale : bande de hauteur max des dents autour de la paroi ;
                       résonateur : col + cavité). Le reste des noeuds intérieurs = la pièce.
    """
    if type_bord not in TYPES_BORD:
        raise ValueError(f"type_bord doit être parmi {TYPES_BORD}")
    h_max = pas_maximal(frequence_source)

    if type_bord in ('fractale', 'koch'):
        # chaque morceau de paroi doit contenir (base ** niveau) * CELLULES_MIN_MOTIF mailles
        base = 4 if type_bord == 'fractale' else 3
        q = base ** niveau * CELLULES_MIN_MOTIF
        n_morceau = q * int(numpy.ceil(LARGEUR / N_MORCEAUX / (q * h_max)))
        nx_mailles = N_MORCEAUX * n_morceau
        spacestep = LARGEUR / nx_mailles
        # minkowski : les dents sortent de moins de 1/3 du morceau, koch : de moins de 1/2
        profondeur = int(numpy.ceil(n_morceau / 3 if type_bord == 'fractale' else n_morceau / 2))
        bande = int(numpy.ceil(n_morceau / 3))       # zone de la paroi au-dessus de la ligne du sol
    else:
        if frequence_resonateur is None:
            raise ValueError("il faut donner frequence_resonateur pour un résonateur")
        niveau_parois = niveau if type_bord == 'resonateur_fractal' else 0
        a = cote_cavite(frequence_resonateur)
        h_max = min(h_max, W_COL / CELLULES_COL)
        # le côté de la cavité doit contenir (4 ** niveau) * CELLULES_MIN_MOTIF mailles
        q = 4 ** niveau_parois * CELLULES_MIN_MOTIF
        n_a = q * int(numpy.ceil(a / (q * h_max)))
        spacestep = a / n_a
        nx_mailles = int(round(LARGEUR / spacestep))
        n_w = int(round(W_COL / spacestep))
        n_L = int(round(L_COL / spacestep))
        # le col est arrondi au maillage : on recalcule la cavité pour garder f0
        a = cote_cavite(frequence_resonateur, n_w * spacestep, n_L * spacestep)
        n_a = q * max(1, int(round(a / (q * spacestep))))
        excursion = int(numpy.ceil(n_a / 3)) if niveau_parois > 0 else 0
        if n_a + 2 * excursion >= nx_mailles:
            raise ValueError("la cavité est trop grande pour la pièce, augmenter frequence_resonateur")
        profondeur = n_L + n_a + excursion
        bande = 0

    y0 = profondeur + MARGE                          # ligne du sol (en mailles)
    nh = int(round(HAUTEUR / spacestep))
    nx = nx_mailles + 1
    ny = y0 + nh + 1

    # --- la pièce, puis on modifie la paroi du bas
    carres = carres_entre((0, y0), (nx_mailles, y0 + nh))
    if type_bord in ('fractale', 'koch'):
        for k in range(N_MORCEAUX):
            A = (k * n_morceau, y0)
            B = ((k + 1) * n_morceau, y0)
            if type_bord == 'fractale':
                minkowski_sur_segment(carres, A, B, niveau)
            else:
                koch_sur_segment(carres, A, B, niveau)
    else:
        xc = (nx_mailles - n_a) // 2                 # cavité centrée
        xn = xc + (n_a - n_w) // 2                   # col au milieu de la cavité
        yc = y0 - n_L - n_a                          # fond de la cavité
        cavite = carres_entre((xc, yc), (xc + n_a, yc + n_a))
        if niveau_parois > 0:
            nb_avant = len(cavite)
            minkowski_sur_segment(cavite, (xc, yc + n_a), (xc, yc), niveau_parois)                 # gauche
            minkowski_sur_segment(cavite, (xc, yc), (xc + n_a, yc), niveau_parois)                 # fond
            minkowski_sur_segment(cavite, (xc + n_a, yc), (xc + n_a, yc + n_a), niveau_parois)     # droite
            if len(cavite) != nb_avant:
                print("  attention : la fractale n'a pas conservé l'aire de la cavité")
        carres |= cavite | carres_entre((xn, y0 - n_L), (xn + n_w, y0))

    domain_omega = carres_vers_domaine(carres, nx, ny)
    M, N = ny, nx

    # --- zone de la paroi (en lignes de la matrice : la ligne i correspond à jy = M - 1 - i)
    jy = (M - 1 - numpy.arange(M))[:, None] * numpy.ones((1, N), dtype=numpy.int64)
    interieur = domain_omega == _env.NODE_INTERIOR
    if type_bord in ('fractale', 'koch'):
        zone_paroi = interieur & (jy <= y0 + bande)
    else:
        zone_paroi = interieur & (jy < y0)

    # --- vérification après construction
    n_pts = verifier_resolution(spacestep, frequence_source)

    print(f"Géométrie '{type_bord}' niveau {niveau} : M = {M}, N = {N} ({M * N} noeuds), "
          f"h = {spacestep * 1000:.2f} mm, {n_pts:.1f} points par longueur d'onde à {frequence_source:.0f} Hz")
    if type_bord in ('resonateur', 'resonateur_fractal'):
        a_r, w_r, L_r = n_a * spacestep, n_w * spacestep, n_L * spacestep
        print(f"  résonateur : col {w_r * 100:.1f} x {L_r * 100:.1f} cm, cavité {a_r * 100:.1f} x {a_r * 100:.1f} cm, "
              f"f0 visée {frequence_resonateur:.1f} Hz, formule avec les dimensions du maillage "
              f"{frequence_helmholtz(a_r, a_r, w_r, L_r):.1f} Hz")
    if M * N > 100000:
        print("  attention : grand maillage, le calcul sera long (baisser le niveau ?)")

    return domain_omega, M, N, spacestep, zone_paroi


def afficher_geometrie(domain_omega, zone_paroi, titre=''):
    """Types de noeuds : gris = hors domaine, blanc = intérieur, rouge = Dirichlet,
    noir = Neumann, vert = Robin ; pointillés bleus = limite de la zone de la paroi."""
    ordre = [_env.NODE_COMPLEMENTARY, _env.NODE_INTERIOR, _env.NODE_DIRICHLET,
             _env.NODE_NEUMANN, _env.NODE_ROBIN]
    image = numpy.zeros(domain_omega.shape)
    for k, valeur in enumerate(ordre):
        image[domain_omega == valeur] = k
    cmap = matplotlib.colors.ListedColormap(['lightgray', 'white', 'red', 'black', 'green'])
    matplotlib.pyplot.figure(figsize=(6, 6))
    matplotlib.pyplot.imshow(image, cmap=cmap, vmin=-0.5, vmax=4.5, interpolation='nearest')
    if zone_paroi.any():
        matplotlib.pyplot.contour(zone_paroi.astype(float), levels=[0.5], colors='blue',
                                  linestyles='dashed', linewidths=0.8)
    matplotlib.pyplot.title(titre)
    matplotlib.pyplot.axis('off')
    matplotlib.pyplot.tight_layout()
