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
import compute_alpha
#import solutions


# ----------------------------------------------------------------------
# -- Optimisation de chi : mise en commentaire pour l'instant (chi = 1 partout)
# ----------------------------------------------------------------------
# def your_optimization_procedure(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
#                            beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
#                            Alpha, mu, chi, V_obj):
#     """This function return the optimized density.
#
#     Parameter:
#         cf solvehelmholtz's remarks
#         Alpha: complex, it corresponds to the absorbtion coefficient;
#         mu: float, it is the initial step of the gradient's descent;
#         V_obj: float, it characterizes the volume constraint on the density chi.
#     """
#
#     k = 0
#     (M, N) = numpy.shape(domain_omega)
#     numb_iter = 100
#     energy = numpy.zeros((numb_iter+1, 1), dtype=numpy.float64)
#     while k < numb_iter and mu > 10**(-5):
#         print('---- iteration number = ', k)
#         print('1. computing solution of Helmholtz problem, i.e., u')
#         print('2. computing solution of adjoint problem, i.e., p')
#         print('3. computing objective function, i.e., energy')
#         print('4. computing parametric gradient')
#         while ene >= energy[k] and mu > 10 ** -5:
#             print('    a. computing gradient descent')
#             print('    b. computing projected gradient')
#             print('    c. computing solution of Helmholtz problem, i.e., u')
#             print('    d. computing objective function, i.e., energy (E)')
#             ene = compute_objective_function(domain_omega, u, spacestep)
#             if bool_a:
#                 # The step is increased if the energy decreased
#                 mu = mu * 1.1
#             else:
#                 # The step is decreased is the energy increased
#                 mu = mu / 2
#         k += 1
#
#     print('end. computing solution of Helmholtz problem, i.e., u')
#
#     return chi, energy, u, grad
#
#
# def your_compute_objective_function(domain_omega, u, spacestep):
#     """
#     This function compute the objective function:
#     J(u,domain_omega)= \int_{domain_omega}||u||^2
#
#     Parameter:
#         domain_omega: Matrix (NxP), it defines the domain and the shape of the
#         Robin frontier;
#         u: Matrix (NxP), it is the solution of the Helmholtz problem, we are
#         computing its energy;
#         spacestep: float, it corresponds to the step used to solve the Helmholtz
#         equation.
#     """
#
#     energy = 0.0
#
#     return energy


# ----------------------------------------------------------------------
# -- Résolution et énergies 
# ----------------------------------------------------------------------

def resoudre(domain_omega, spacestep, frequence, alpha_rob):
    """Solution de Helmholtz à la fréquence donnée (onde plane imposée en haut)."""
    (M, N) = numpy.shape(domain_omega)
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
    MATERIAL = 'RECYCLED_RUBBER'
    TYPE_BORD = 'resonateur_fractal'          # 'fractale', 'koch', 'resonateur', 'resonateur_fractal'
    level = 1                       # level of the fractal
    FREQUENCE_SOURCE = 1000.0        # fréquence de la source (Hz)
    FREQUENCE_RESONATEUR = 1000.0    # fréquence visée par le résonateur (Hz), si résonateur

    # -- balayage en fréquence (comparaison avec / sans Robin)
    BALAYAGE = True 
    if TYPE_BORD in ('resonateur', 'resonateur_fractal'):
        frequences = numpy.linspace(0.5 * FREQUENCE_RESONATEUR, 1.5 * FREQUENCE_RESONATEUR, 15)
    else:
        frequences = numpy.linspace(100.0, 2000.0, 20)
    f_max = max(FREQUENCE_SOURCE, frequences.max()) if BALAYAGE else FREQUENCE_SOURCE

    # -- set geometry of domain (le maillage respecte 10 points par longueur d'onde jusqu'à f_max)
    domain_omega, M, N, spacestep, zone_paroi = geometrie.creer_geometrie(TYPE_BORD, level, f_max, FREQUENCE_RESONATEUR)
    for frequence in [FREQUENCE_SOURCE] + (list(frequences) if BALAYAGE else []):
        geometrie.verifier_resolution(spacestep, frequence)

    # -- set parameters of the partial differential equation
    wavenumber = 2 * numpy.pi * FREQUENCE_SOURCE / geometrie.C_SON  # wavenumber

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
    f_dir[0, 0:N] = 1.0
    # spherical wave defined on top
    #f_dir[:, :] = 0.0
    #f_dir[0, int(N/2)] = 10.0

    # -- initialize
    alpha_rob[:, :] = - wavenumber * 1j

    # -- define material density matrix : chi = 1 sur toute la paroi de Robin
    chi = numpy.ones((M, N), dtype=numpy.float64)
    chi = preprocessing.set2zero(chi, domain_omega)

    # -- define absorbing material
    omegas = frequences * 2.0 * numpy.pi
    alphas = compute_alpha.run_compute_alpha_omegas(MATERIAL, omegas)
    # Alpha = 10.0 - 10.0 * 1j
    # -- this is the function you have written during your project
    #import compute_alpha
    #Alpha = compute_alpha.compute_alpha(...)
    # alpha_rob = Alpha * chi
    alphas_rob = numpy.array([alpha * chi for alpha in alphas])

    # -- set parameters for optimization
    # S = 0  # surface of the fractal
    # for i in range(0, M):
    #     for j in range(0, N):
    #         if domain_omega[i, j] == _env.NODE_ROBIN:
    #             S += 1
    # V_0 = 1  # initial volume of the domain
    # V_obj = numpy.sum(numpy.sum(chi)) / S  # constraint on the density
    # mu = 5  # initial gradient step
    # mu1 = 10**(-5)  # parameter of the volume functional

    # ----------------------------------------------------------------------
    # -- Do not modify this cell, these are the values that you will be assessed against.
    # ----------------------------------------------------------------------
    # -- compute finite difference solution

    #u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
    #                    beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alphas_rob)
    #chi0 = chi.copy()
    #u0 = u.copy()

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

    # chin = chi.copy()
    # un = u.copy()

    # ----------------------------------------------------------------------
    # -- Comparaison avec / sans condition de Robin
    # ----------------------------------------------------------------------
    #titre = TYPE_BORD if TYPE_BORD == 'resonateur' else f"{TYPE_BORD}, niveau {level}"
    #alpha_rob_sans = numpy.zeros((M, N), dtype=numpy.complex128)   # alpha = 0 : paroi rigide (Neumann)
    #u_sans = resoudre(domain_omega, spacestep, FREQUENCE_SOURCE, alpha_rob_sans)
    #e_avec = energies(domain_omega, u0, spacestep, zone_paroi)
    #e_sans = energies(domain_omega, u_sans, spacestep, zone_paroi)
    #print(f"\nÀ {FREQUENCE_SOURCE:.0f} Hz :            sans Robin   avec Robin")
    #for g, nom in GRANDEURS:
    #    print(f"  {nom:32s} {e_sans[g]:11.4e}  {e_avec[g]:11.4e}")

    #geometrie.afficher_geometrie(domain_omega, zone_paroi, f"géométrie : {titre} (M = {M}, N = {N})")
    #afficher_champs(domain_omega, u_sans, u0, e_sans, e_avec, FREQUENCE_SOURCE, titre)

    alpha_rob_sans = numpy.zeros((M, N), dtype=numpy.complex128)   # alpha = 0 : paroi rigide (Neumann)
    titre = TYPE_BORD if TYPE_BORD == 'resonateur' else f"{TYPE_BORD}, niveau {level}"

    if BALAYAGE:
        t0 = time.time()
        liste_sans, liste_avec = [], []
        for frequence, alpha in zip(frequences, alphas_rob):
            u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha)
            chi0 = chi.copy()
            u0 = u.copy()
            
            u_sans = resoudre(domain_omega, spacestep, FREQUENCE_SOURCE, alpha_rob_sans)
            e_avec = energies(domain_omega, u0, spacestep, zone_paroi)
            e_sans = energies(domain_omega, u_sans, spacestep, zone_paroi)
            '''print(f"\nÀ {FREQUENCE_SOURCE:.0f} Hz :            sans Robin   avec Robin")
                                                for g, nom in GRANDEURS:
                                                    print(f"  {nom:32s} {e_sans[g]:11.4e}  {e_avec[g]:11.4e}")'''

            liste_sans.append(energies(domain_omega, resoudre(domain_omega, spacestep, frequence, alpha_rob_sans),
                                       spacestep, zone_paroi))
            liste_avec.append(energies(domain_omega, resoudre(domain_omega, spacestep, frequence, alpha),
                                       spacestep, zone_paroi))
        print(f"Balayage de {len(frequences)} fréquences : {time.time() - t0:.0f} s")
        f0 = FREQUENCE_RESONATEUR if TYPE_BORD in ('resonateur', 'resonateur_fractal') else None
        afficher_balayage(frequences, liste_sans, liste_avec, titre, f0)

    geometrie.afficher_geometrie(domain_omega, zone_paroi, f"géométrie : {titre} (M = {M}, N = {N})")

    # -- plot chi, u, and energy
    postprocessing._plot_uncontroled_solution(u0, chi0)
    # postprocessing._plot_controled_solution(un, chin)
    # err = un - u0
    # postprocessing._plot_error(err)
    # postprocessing._plot_energy_history(energy)

    matplotlib.pyplot.show()
    print('End.')
