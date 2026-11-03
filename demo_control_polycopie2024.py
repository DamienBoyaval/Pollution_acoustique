# -*- coding: utf-8 -*-


# Python packages
import matplotlib.colors
import matplotlib.pyplot
import numpy
import os
import time


# MRG packages
import _env
import preprocessing
import processing
import postprocessing
import geometrie
import train_source

#import solutions
import material_alpha

def your_optimization_procedure(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                                beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                                Alpha, mu, chi, V_obj, max_iter=100):
    """Descente de gradient projete (pulsation omega fixee, volume V_obj fixe).

    On minimise l'energie RELATIVE J_rel(chi) = J(chi) / J(chi_0), de sorte que
    J_rel et son gradient soient en O(1) (J seul vaut ~1e-6 ici, ce qui rendait
    le pas mu = 5 et l'increment 0.001 completement inadaptes).

    chi_{n+1} = P(chi_n - zeta_n * grad J_rel(chi_n))
    zeta_{n+1} = zeta_n + 0.001 si l'energie J_rel diminue,
    sinon zeta_n / 2, zeta_n / 4, ... jusqu'a ce que J_rel(chi_{n+1}) < J_rel(chi_n).
    """

    numb_iter = max_iter
    tolerance_chi = 1e-6      # seuil du critere d'arret ||chi_{n+1} - chi_n|| < delta
    mu_min = 1e-5             # plus petit pas accepte par la recherche lineaire
    mu_increase = 0.001       # increment du pas : zeta_{n+1} = zeta_n + 0.001 si succes

    (M, N) = numpy.shape(domain_omega)
    energy = numpy.zeros((numb_iter + 1, 1), dtype=numpy.float64)  # energie relative J/J0

    def solve(chi_loc):
        return processing.solve_helmholtz(
            domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
            beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob,
            Alpha * chi_loc, return_system=True)

    # initialisation : projection de chi_0 dans l'ensemble admissible U_ad(V_obj)
    chi = your_project_density(chi, domain_omega, V_obj)
    u, mat, _ = solve(chi)

    # normalisation : J0 = energie initiale (absolue)
    J0 = your_compute_objective_function(domain_omega, u, spacestep)
    energy[0, 0] = 1.0


    variation_chi = numpy.inf
    k = 0
    while k < numb_iter and mu > mu_min:
        p = your_compute_adjoint(domain_omega, u, spacestep, mat)
        grad = your_compute_gradient(domain_omega, u, p, spacestep,
                                     f_rob, beta_pde, beta_rob, Alpha, chi) / J0

        # Recherche lineaire : division du pas par 2 tant que l'energie ne diminue pas
        ene = energy[k, 0]
        while ene >= energy[k, 0] and mu > mu_min:
            chi_test = your_project_density(chi - mu * grad, domain_omega, V_obj)
            u_test, mat_test, _ = solve(chi_test)
            ene = your_compute_objective_function(domain_omega, u_test, spacestep) / J0

            if ene < energy[k, 0]:
                variation_chi = numpy.linalg.norm(chi_test - chi)
                chi, u, mat = chi_test, u_test, mat_test
                mu = mu + mu_increase
            else:
                mu = mu / 2

        # Arret si le pas est devenu trop petit sans ameliorer l'energie
        if ene >= energy[k, 0]:
            break

        k += 1
        energy[k, 0] = ene

        if variation_chi < tolerance_chi:
            break

    # -- calcul du gradient pour la densite finale (retourne pour le post-traitement)
    p = your_compute_adjoint(domain_omega, u, spacestep, mat)
    grad = your_compute_gradient(domain_omega, u, p, spacestep,
                                 f_rob, beta_pde, beta_rob, Alpha, chi) / J0


    # energy = historique de J/J0 (1 -> valeur finale), grad = gradient de J/J0
    return chi, energy[:k + 1], u, grad

def binarize_chi_threshold(chi, domain_omega, threshold=0.5):
    """
    Transforme chi en fonction binaire {0, 1} avec un seuil explicite.
    """
    M, N = chi.shape
    chi_bin = numpy.zeros((M, N), dtype=numpy.float64)

    for i in range(M):
        for j in range(N):
            # On ne modifie que les points situés sur la frontière de Robin
            if domain_omega[i, j] == _env.NODE_ROBIN:
                if chi[i, j] >= threshold:
                    chi_bin[i, j] = 1.0
                else:
                    chi_bin[i, j] = 0.0

    return chi_bin

def your_compute_adjoint(domain_omega, u, spacestep, mat):
    # Discrete adjoint: A^H p = h^2 u on the interior, zero elsewhere.
    g = numpy.zeros_like(u, dtype=numpy.complex128)
    masque = domain_omega == _env.NODE_INTERIOR
    g[masque] = spacestep**2 * u[masque]
    return scipy.sparse.linalg.spsolve(mat.conjugate().transpose().tocsr(),
                                     g.ravel()).reshape(u.shape)


def your_compute_gradient(domain_omega, u, p, spacestep,
                          f_rob, beta_pde, beta_rob, Alpha, chi):
    # Differentiate exactly the Robin elimination used by processing.py.
    # dJ = 2 Re[p^H (db - dA u)]; p here includes neither factor 2 nor a minus sign.
    M, N = domain_omega.shape
    grad = numpy.zeros((M, N), dtype=numpy.float64)
    h = spacestep
    for i in range(1, M-1):
        for j in range(1, N-1):
            if domain_omega[i, j] != _env.NODE_INTERIOR:
                continue
            for a, b in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
                if domain_omega[a, b] == _env.NODE_ROBIN:
                    beta = beta_pde[i, j]
                    denom = beta_rob[a, b] - Alpha * chi[a, b] * h / beta
                    dA = Alpha * h * beta_rob[a, b] / (beta * denom**2)
                    db = f_rob[a, b] * h**2 * Alpha / (beta * denom**2)
                    grad[a, b] += 2 * numpy.real(
                        numpy.conj(p[i, j]) * (db - dA * u[i, j]))
    return grad


def your_project_density(values, domain_omega, V_obj):
    # Euclidean projection onto 0 <= chi <= 1, sum(chi) = V_obj*S.
    masque = domain_omega == _env.NODE_ROBIN
    S = numpy.count_nonzero(masque)
    z = numpy.asarray(values, dtype=float)[masque]
    chi = numpy.zeros(domain_omega.shape, dtype=float)
    if V_obj == 0 or V_obj == 1:
        chi[masque] = V_obj
        return chi
    # These bounds bracket every possible shift, even for large gradient steps.
    lo, hi = -numpy.max(z), 1 - numpy.min(z)
    for _ in range(80):
        shift = (lo + hi) / 2
        if numpy.sum(numpy.clip(z + shift, 0, 1)) < V_obj * S:
            lo = shift
        else:
            hi = shift
    chi[masque] = numpy.clip(z + (lo + hi) / 2, 0, 1)
    return chi


def your_compute_objective_function(domain_omega, u, spacestep):
    """
    This function compute the objective function:
    J(u,domain_omega)= \int_{domain_omega}||u||^2 

    Parameter:
        domain_omega: Matrix (NxP), it defines the domain and the shape of the
        Robin frontier;
        u: Matrix (NxP), it is the solution of the Helmholtz problem, we are
        computing its energy;
        spacestep: float, it corresponds to the step used to solve the Helmholtz
        equation.
    """

    energy = spacestep**2 * numpy.sum(
        numpy.abs(u[domain_omega == _env.NODE_INTERIOR])**2)

    return energy
    wavenumber = 2 * numpy.pi * frequence / geometrie.C_SON
    beta_pde, alpha_pde, alpha_dir, beta_neu, _, beta_rob = preprocessing._set_coefficients_of_pde(M, N)
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)
    f_dir[0, 0:N] = 1.0
    return processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                                      beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)


def energies(domain_omega, u, spacestep, zone_paroi):
    """
    Intégrales de |u|² (u est nul hors des noeuds intérieurs) :
        E_totale       : sur tout le domaine
        densite_paroi  : énergie dans la zone de la paroi / aire de la zone
        densite_piece  : énergie dans la pièce / aire de la pièce
    """
    u2 = numpy.abs(u) ** 2
    zone_piece = (domain_omega == _env.NODE_INTERIOR) & ~zone_paroi
    aire = spacestep ** 2
    E_paroi = aire * numpy.sum(u2[zone_paroi])
    E_piece = aire * numpy.sum(u2[zone_piece])
    return {'E_totale': E_paroi + E_piece,
            'densite_paroi': E_paroi / (aire * zone_paroi.sum()) if zone_paroi.any() else numpy.nan,
            'densite_piece': E_piece / (aire * zone_piece.sum())}


GRANDEURS = [('densite_paroi', "densité d'énergie dans la paroi"),
             ('densite_piece', "densité d'énergie dans la pièce"),
             ('E_totale', 'énergie totale')]


def afficher_champs(domain_omega, u_sans, u_avec, e_sans, e_avec, frequence, titre):
    """|u|² sans et avec Robin (même échelle), puis comparaison des énergies."""
    hors_domaine = domain_omega != _env.NODE_INTERIOR
    u2_sans = numpy.ma.masked_where(hors_domaine, numpy.abs(u_sans) ** 2)
    u2_avec = numpy.ma.masked_where(hors_domaine, numpy.abs(u_avec) ** 2)
    vmax = max(u2_sans.max(), u2_avec.max())
    norme = matplotlib.colors.LogNorm(vmin=vmax * 1e-4, vmax=vmax)

    matplotlib.pyplot.figure(figsize=(16, 5))
    for k, (u2, nom) in enumerate([(u2_sans, 'sans Robin (paroi rigide)'), (u2_avec, 'avec Robin (chi = 1)')]):
        matplotlib.pyplot.subplot(1, 3, k + 1)
        matplotlib.pyplot.imshow(u2, cmap='jet', norm=norme)
        matplotlib.pyplot.colorbar()
        matplotlib.pyplot.title(f'$|u|^2$ à {frequence:.0f} Hz, {nom}', fontsize=9)
        matplotlib.pyplot.axis('off')

    matplotlib.pyplot.subplot(1, 3, 3)
    position = numpy.arange(len(GRANDEURS))
    matplotlib.pyplot.bar(position - 0.2, [e_sans[g] for g, _ in GRANDEURS], 0.4, label='sans Robin')
    matplotlib.pyplot.bar(position + 0.2, [e_avec[g] for g, _ in GRANDEURS], 0.4, label='avec Robin')
    matplotlib.pyplot.xticks(position, ['densité\nparoi', 'densité\npièce', 'énergie\ntotale'])
    matplotlib.pyplot.yscale('log')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.title(f'énergies à {frequence:.0f} Hz', fontsize=9)
    matplotlib.pyplot.suptitle(titre)
    matplotlib.pyplot.tight_layout()


def afficher_balayage(frequences, liste_sans, liste_avec, titre, f0=None):
    """Densité dans la paroi, dans la pièce et énergie totale en fonction de la fréquence."""
    matplotlib.pyplot.figure(figsize=(16, 4.5))
    for k, (g, nom) in enumerate(GRANDEURS):
        matplotlib.pyplot.subplot(1, 3, k + 1)
        matplotlib.pyplot.semilogy(frequences, [e[g] for e in liste_sans], 'k--o', markersize=3, label='sans Robin')
        matplotlib.pyplot.semilogy(frequences, [e[g] for e in liste_avec], '-o', markersize=3, label='avec Robin')
        if f0 is not None:
            matplotlib.pyplot.axvline(f0, color='r', linestyle=':', label=f'f0 = {f0:.0f} Hz')
        matplotlib.pyplot.xlabel('fréquence de la source (Hz)')
        matplotlib.pyplot.ylabel(nom)
        matplotlib.pyplot.legend()
        matplotlib.pyplot.title(nom)
    matplotlib.pyplot.suptitle(titre)
    matplotlib.pyplot.tight_layout()


if __name__ == '__main__':

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- set parameters of the geometry
    TYPE_BORD = 'fractale'          # 'fractale', 'koch', 'resonateur', 'resonateur_fractal'
    level = 1                       # level of the fractal
    FREQUENCE_SOURCE = 1000.0        # fréquence de la source (Hz)
    FREQUENCE_RESONATEUR = 1000.0    # fréquence visée par le résonateur (Hz), si résonateur

    # -- balayage en fréquence (comparaison avec / sans Robin)
    BALAYAGE = False
    if TYPE_BORD in ('resonateur', 'resonateur_fractal'):
        frequences = numpy.linspace(0.5 * FREQUENCE_RESONATEUR, 1.5 * FREQUENCE_RESONATEUR, 50)
    else:
        frequences = numpy.linspace(100.0, 2000.0, 20)
    f_max = max(FREQUENCE_SOURCE, frequences.max()) if BALAYAGE else FREQUENCE_SOURCE

    # -- set geometry of domain (le maillage respecte 10 points par longueur d'onde jusqu'à f_max)
    domain_omega, M, N, spacestep, zone_paroi = geometrie.creer_geometrie(TYPE_BORD, level, f_max, FREQUENCE_RESONATEUR)
    for frequence in [FREQUENCE_SOURCE] + (list(frequences) if BALAYAGE else []):
        geometrie.verifier_resolution(spacestep, frequence)

    # -- set parameters of the partial differential equation
    kx = -1.0
    ky = -1.0
    wavenumber = numpy.sqrt(kx**2 + ky**2)  # wavenumber
    wavenumber, Alpha_material = material_alpha.solver_parameters(
        omega_physical, Alpha_table[frequency_index], length_m, sound_speed)
    wavenumber = float(wavenumber)
    print('Materiau :', material)
    print(f'Pulsation : {omega_physical:.8f} rad/s')
    print(f'Frequence : {omega_physical / (2 * numpy.pi):.8f} Hz')
    print(f'Nombre d\'onde du solveur : {wavenumber:.8f}')
    print(f'Echelle physique L : {length_m:g} m ; c : {sound_speed:g} m/s')
    print(f'Erreur d\'ajustement fournie : {error_table[frequency_index]:.6e}')

    # ----------------------------------------------------------------------
    # -- Do not modify this cell, these are the values that you will be assessed against.
    # ----------------------------------------------------------------------
    # --- set coefficients of the partial differential equation
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = preprocessing._set_coefficients_of_pde(M, N)

    # -- set right hand sides of the partial differential equation
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- define boundary conditions
    # planar wave defined on top
    f_dir[:, :] = 0.0

    # ----------------------------------------------------------------------
    # 2. Modélisation des sources f : Mur absorbant SUR LE CÔTÉ du train
    # ----------------------------------------------------------------------
    f[:, :] = 0.0

    # Source harmonique a la meme frequence que k et Alpha.
    # Modele spectral simplifie : references et hypotheses dans README_SOURCE_TRAIN.md.
    frequency_hz = float(omega_physical / (2 * numpy.pi))
    V_kmh = 300.0
    V_ref = 300.0  # vitesse de reference, pas une vitesse de croisement calibree
    sigma = 0.03  # 1.5 mailles pour N=50 ; largeur fixe si le maillage change
    spectral_parameters = dict(
        reference_speed_kmh=V_ref,
        rolling_peak_hz=1000.0, aero_peak_hz=500.0,
        rolling_width_octaves=1.0, aero_width_octaves=1.0,
        rolling_gain=1.0, aero_gain=1.0)
    f, (A_roul, A_aero) = train_source.gaussian_train_source(
        domain_omega, spacestep, frequency_hz, V_kmh, sigma=sigma,
        rolling_center=(0.9, 0.5), aero_center=(0.5, 0.5),
        **spectral_parameters)
    print('Source train : enveloppes spectrales de scenario, non calibrees.')
    print(f'Frequence source : {frequency_hz:.3f} Hz ; vitesse : {V_kmh:g} km/h')
    print(f'Amplitudes relatives : roulement={A_roul:.6f}, aero={A_aero:.6f}')

    # -- initialize
    alpha_rob[:, :] = - wavenumber * 1j

    # -- set parameters for optimization
    S = numpy.count_nonzero(domain_omega == _env.NODE_ROBIN)
    V_0 = 1  # initial volume of the domain
    V_obj = 0.3  # constraint on the density (30% of the Robin boundary)

    # -- define material density matrix
    chi = preprocessing._set_chi(M, N, x, y)
    chi = preprocessing.set2zero(chi, domain_omega)
    chi = your_project_density(chi, domain_omega, V_obj)

    # -- define absorbing material
    Alpha = 10.0 - 10.0 * 1j
    # -- this is the function you have written during your project
    #import compute_alpha
    #Alpha = compute_alpha.compute_alpha(...)
    alpha_rob = Alpha * chi

    mu = 5  # initial gradient step
    mu1 = 10**(-5)  # parameter of the volume functional

    # ----------------------------------------------------------------------
    # -- Do not modify this cell, these are the values that you will be assessed against.
    # ----------------------------------------------------------------------
    # -- compute finite difference solution
    u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
    chi0 = chi.copy()
    u0 = u.copy()

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- compute optimization
    # energy = numpy.zeros((100+1, 1), dtype=numpy.float64)
    # chi, energy, u, grad = your_optimization_procedure(...)
    #chi, energy, u, grad = solutions.optimization_procedure(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
    #                    beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
    #                    Alpha, mu, chi, V_obj, mu1, V_0)
    # --- en of optimization

    chin = chi.copy()
    un = u.copy()

    # ----------------------------------------------------------------------
    # -- Comparaison avec / sans condition de Robin
    # ----------------------------------------------------------------------
    titre = TYPE_BORD if TYPE_BORD == 'resonateur' else f"{TYPE_BORD}, niveau {level}"
    alpha_rob_sans = numpy.zeros((M, N), dtype=numpy.complex128)   # alpha = 0 : paroi rigide (Neumann)
    u_sans = resoudre(domain_omega, spacestep, FREQUENCE_SOURCE, alpha_rob_sans)
    e_avec = energies(domain_omega, u0, spacestep, zone_paroi)
    e_sans = energies(domain_omega, u_sans, spacestep, zone_paroi)
    print(f"\nÀ {FREQUENCE_SOURCE:.0f} Hz :            sans Robin   avec Robin")
    for g, nom in GRANDEURS:
        print(f"  {nom:32s} {e_sans[g]:11.4e}  {e_avec[g]:11.4e}")

    geometrie.afficher_geometrie(domain_omega, zone_paroi, f"géométrie : {titre} (M = {M}, N = {N})")
    afficher_champs(domain_omega, u_sans, u0, e_sans, e_avec, FREQUENCE_SOURCE, titre)

    if BALAYAGE:
        t0 = time.time()
        liste_sans, liste_avec = [], []
        for frequence in frequences:
            liste_sans.append(energies(domain_omega, resoudre(domain_omega, spacestep, frequence, alpha_rob_sans),
                                       spacestep, zone_paroi))
            liste_avec.append(energies(domain_omega, resoudre(domain_omega, spacestep, frequence, alpha_rob),
                                       spacestep, zone_paroi))
        print(f"Balayage de {len(frequences)} fréquences : {time.time() - t0:.0f} s")
        f0 = FREQUENCE_RESONATEUR if TYPE_BORD in ('resonateur', 'resonateur_fractal') else None
        afficher_balayage(frequences, liste_sans, liste_avec, titre, f0)

    # -- affichage de la solution initiale
    postprocessing._plot_uncontroled_solution(u0, chi0)

    # -- binarisation APRES l'optimisation continue (sans affichage intermediaire)
    # chin et un conservent la solution continue pour la comparaison.
    threshold = 0.5
    chi = binarize_chi_threshold(chin, domain_omega, threshold=threshold)
    alpha_rob = Alpha * chi
    # Le champ doit etre recalcule avec le nouveau coefficient de Robin.
    u = processing.solve_helmholtz(
        domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
    energy_continuous = your_compute_objective_function(domain_omega, un, spacestep)
    energy_binary = your_compute_objective_function(domain_omega, u, spacestep)

    print('\nBILAN DE LA BINARISATION')
    print(f'Seuil : {threshold:g} ; valeurs de chi : {numpy.unique(chi)}')
    print(f'Fraction absorbante binaire : {chi.sum() / S:.6f} (cible : {V_obj:g})')
    print('Le seuillage ne garantit pas la conservation de la quantite de materiau.')
    print(f'Energie initiale : {your_compute_objective_function(domain_omega, u0, spacestep):.12e}')
    print(f'Energie absolue binaire : {energy_binary:.12e}')

    postprocessing.myimshow(
        chi, title='Densite binaire (seuil 0.5)', colorbar='colorbar',
        cmap='gray_r', vmin=0, vmax=1, filename='fig_chi_binaire.jpg')
    postprocessing.myimshow(
        numpy.real(u), title='Champ binaire : partie reelle', colorbar='colorbar',
        cmap='jet', filename='fig_u_binaire_re.jpg')
    postprocessing.myimshow(
        numpy.imag(u), title='Champ binaire : partie imaginaire', colorbar='colorbar',
        cmap='jet', filename='fig_u_binaire_im.jpg')

    matplotlib.pyplot.show()
    print('End.')
