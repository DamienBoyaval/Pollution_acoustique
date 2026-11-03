# -*- coding: utf-8 -*-
"""
Construction du domaine et de la paroi absorbante (frontière de Robin).

Repère de la grille : x = colonne (0..N), y = ligne (0..M), y vers le bas.
La pièce occupe les lignes 0 à N (onde imposée en haut, ligne 0). La paroi part
de (0, N) et arrive en (N, N) ; ce qui est sous la paroi et hors des cavités
est retiré du domaine.

Tous les segments de la paroi sont horizontaux ou verticaux et tous les sommets
ont des coordonnées entières, pour que la paroi tombe sur les points de la grille.

Formes disponibles (paramètre forme de creer_domaine) :
    'plat'        : paroi plate
    'koch'        : fractale du sujet sur toute la paroi
    'minkowski'   : paroi coupée en morceaux de fractale
    'creneaux'    : petites cavités rectangulaires
    'resonateurs' : un résonateur de Helmholtz par nombre d'onde de k_resonateurs,
                    parois de la cavité fractales si niveau > 0

Fractale : chaque segment A -> B est remplacé par 8 segments de longueur |AB|/4
(motif de preprocessing.create_motif_koch) : d'abord une bosse vers l'extérieur
du domaine, puis une vers l'intérieur, de même aire. Au niveau l, la longueur du
segment doit donc être un multiple de 4**l mailles.
"""

import numpy

import _env
import preprocessing


def creer_domaine(M, N, forme='koch', niveau=0, k_resonateurs=()):
    """
    Renvoie domain_omega (M x N) et les coordonnées x, y des points de la paroi.

    forme         : forme de la paroi (voir en tête du fichier)
    niveau        : niveau de la fractale (koch, minkowski, parois des résonateurs)
    k_resonateurs : nombres d'onde visés par les résonateurs, même unité que le
                    wavenumber du solveur (domaine de largeur 1)
    """
    if M < 2 * N:
        raise ValueError("Il faut M >= 2 N pour contenir la paroi et ce qui est dessous.")
    if niveau < 0 or int(niveau) != niveau:
        raise ValueError("Le niveau doit être un entier >= 0.")

    domain_omega = numpy.zeros((M, N), dtype=numpy.int64)
    domain_omega[:, :] = _env.NODE_INTERIOR
    domain_omega[0, :] = _env.NODE_DIRICHLET  # nord : onde imposée
    domain_omega[M - 1, :] = _env.NODE_NEUMANN  # sud
    domain_omega[:, 0] = _env.NODE_NEUMANN  # ouest
    domain_omega[:, N - 1] = _env.NODE_NEUMANN  # est

    sommets = sommets_paroi(M, N, forme, niveau, k_resonateurs)
    x, y = preprocessing.create_fractal_coordinates(sommets, domain_omega)
    for k in range(len(x) - 1):
        domain_omega[int(y[k]), int(x[k])] = _env.NODE_ROBIN
    # tout ce qui est relié au coin bas droit (sous la paroi) sort du domaine
    domain_omega = preprocessing.partition_domain(domain_omega, [M - 2, N - 2])

    return domain_omega, x, y


def sommets_paroi(M, N, forme, niveau=0, k_resonateurs=()):
    """Liste des sommets de la paroi, de (0, N) à (N, N), pour la forme choisie."""
    if forme == 'plat':
        return paroi_plate(N)
    if forme == 'koch':
        return paroi_koch(N, niveau)
    if forme == 'minkowski':
        return paroi_minkowski(N, niveau)
    if forme == 'creneaux':
        return paroi_creneaux(N)
    if forme == 'resonateurs':
        return paroi_resonateurs(M, N, niveau, k_resonateurs)
    raise ValueError("forme inconnue : %s" % forme)


def sommet(x, y):
    """Sommet au format de preprocessing.create_fractal_nodes : numpy.array([[x], [y]])."""
    return numpy.array([[x], [y]], dtype=numpy.float64)


# ---------------------------------------------------------------------------
# Formes de paroi
# ---------------------------------------------------------------------------

def paroi_plate(N):
    return [sommet(0, N), sommet(N, N)]


def paroi_koch(N, niveau):
    """Fractale du sujet sur toute la largeur."""
    if N % 4**niveau:
        raise ValueError("forme 'koch' : N doit être un multiple de 4**niveau = %d" % 4**niveau)
    return preprocessing.create_fractal_nodes([sommet(0, N), sommet(N, N)], niveau)


def paroi_minkowski(N, niveau):
    """Paroi coupée en morceaux de longueur P (multiple de 4**niveau), centrés."""
    P = (N // 3) // 4**niveau * 4**niveau
    if P == 0:
        raise ValueError("forme 'minkowski' : niveau %d trop grand, il faut N >= %d"
                         % (niveau, 3 * 4**niveau))
    n_morceaux = N // P
    x0 = (N - n_morceaux * P) // 2
    sommets = [sommet(0, N)]
    for k in range(n_morceaux):
        A = sommet(x0 + k * P, N)
        B = sommet(x0 + (k + 1) * P, N)
        sommets += preprocessing.create_fractal_nodes([A, B], niveau)
    return sommets + [sommet(N, N)]


def paroi_creneaux(N):
    """Cavités carrées de côté p, espacées de p."""
    p = max(2, N // 10)
    sommets = [sommet(0, N)]
    x0 = p
    while x0 + p <= N - p:
        sommets += [sommet(x0, N), sommet(x0, N + p), sommet(x0 + p, N + p), sommet(x0 + p, N)]
        x0 += 2 * p
    return sommets + [sommet(N, N)]


def paroi_resonateurs(M, N, niveau, k_resonateurs):
    """
    Un résonateur de Helmholtz par nombre d'onde de k_resonateurs. La paroi est
    partagée en bandes égales, un résonateur centré dans chaque bande.
    """
    if len(k_resonateurs) == 0:
        raise ValueError("forme 'resonateurs' : donner au moins un nombre d'onde dans k_resonateurs")
    n = len(k_resonateurs)
    w = max(3, round(0.03 * N))  # largeur du col
    S_max = (N // n - 4) * 3 // 5  # cavité + dents (S/3 de chaque côté) dans la bande
    sommets = [sommet(0, N)]
    for i, k in enumerate(k_resonateurs):
        S, L_col = dimensions_resonateur(k, M, N, w, S_max, niveau)
        print("résonateur k = %.2f : cavité %d x %d mailles (%.1f cm), col %d x %d mailles, "
              "k formule = %.2f" % (k, S, S, 100 * S / N, w, L_col, k_helmholtz(N, w, S, L_col)))
        centre = (2 * i + 1) * N // (2 * n)
        sommets += sommets_resonateur(N, centre - S // 2, S, w, L_col, niveau)
    return sommets + [sommet(N, N)]


# ---------------------------------------------------------------------------
# Résonateur de Helmholtz
# ---------------------------------------------------------------------------

def sommets_resonateur(N, xc, S, w, L_col, niveau):
    """
    Résonateur sous la paroi y = N : col de largeur w et de longueur L_col, puis
    cavité carrée de côté S et de bord gauche xc. Les parois gauche, fond et
    droite de la cavité sont fractales si niveau > 0. Va de (xn, N) à (xn + w, N),
    xn étant le bord gauche du col.
    """
    xn = xc + (S - w) // 2  # col au milieu de la cavité
    yc = N + L_col  # haut de la cavité
    yb = yc + S  # fond de la cavité
    gauche = preprocessing.create_fractal_nodes([sommet(xc, yc), sommet(xc, yb)], niveau)
    fond = preprocessing.create_fractal_nodes([sommet(xc, yb), sommet(xc + S, yb)], niveau)
    droite = preprocessing.create_fractal_nodes([sommet(xc + S, yb), sommet(xc + S, yc)], niveau)
    return ([sommet(xn, N), sommet(xn, yc)]
            + gauche + fond[1:] + droite[1:]
            + [sommet(xn + w, yc), sommet(xn + w, N)])


def k_helmholtz(N, w, S, L_col):
    """
    Nombre d'onde de résonance (formule de Helmholtz 2D, correction de longueur 2 w) :
        k0 = sqrt( w / (A (L + 2 w)) ),  A = S²
    w, S, L_col en mailles ; le domaine fait 1 de large donc une maille = 1 / N.
    """
    return N * numpy.sqrt(w / (S**2 * (L_col + 2 * w)))


def dimensions_resonateur(k, M, N, w, S_max, niveau):
    """
    Côté S de la cavité (multiple de 4**niveau) et longueur L_col du col pour que
    le résonateur résonne en k. On prend la plus grande cavité qui rentre, et on
    règle la longueur du col avec la formule inversée : L = w / (k² A) - 2 w.
    """
    p = 4**niveau
    for S in range(S_max // p * p, w, -p):
        L_col = round(w * N**2 / (k**2 * S**2) - 2 * w)
        if L_col >= 2 and N + L_col + S + S // 3 <= M - 4:
            return S, L_col
    S = S_max // p * p
    k_min = k_helmholtz(N, w, S, M - 4 - N - S - S // 3)
    k_max = k_helmholtz(N, w, (w // p + 1) * p, 2)  # plus petite cavité plus large que le col
    raise ValueError("résonateur k = %.2f impossible au niveau %d avec N = %d : "
                     "k doit être entre %.2f et %.2f (augmente N pour monter en k)"
                     % (k, niveau, N, k_min, k_max))
