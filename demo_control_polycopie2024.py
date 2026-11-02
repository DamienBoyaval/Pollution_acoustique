# -*- coding: utf-8 -*-


# Python packages
import matplotlib.pyplot
import numpy
import os


# MRG packages
import _env
import preprocessing
import processing
import postprocessing
#import solutions

import scipy.sparse.linalg


def your_optimization_procedure(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                           beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                           Alpha, mu, chi, V_obj):
    """This function return the optimized density.

    Parameter:
        cf solvehelmholtz's remarks
        Alpha: complex, it corresponds to the absorbtion coefficient;
        mu: float, it is the initial step of the gradient's descent;
        V_obj: float, it characterizes the volume constraint on the density chi.
    """

    k = 0
    (M, N) = numpy.shape(domain_omega)
    numb_iter = 100
    energy = numpy.zeros((numb_iter+1, 1), dtype=numpy.float64)


# Resolution initiale de Helmhotz
    u, mat, rhs = processing.solve_helmholtz( domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob, beta_pde, alpha_pde, alpha_dir,  beta_neu, beta_rob, alpha_rob, return_system=True)
    # Calcul de l'énergie initiale
    energy[0] = your_compute_objective_function( domain_omega, u, spacestep)

    print("Énergie initiale :", energy[0, 0])

    # Second membre du problème adjoint
    g = numpy.zeros((M, N), dtype=numpy.complex128)

    g[domain_omega == _env.NODE_INTERIOR] = (spacestep**2 * u[domain_omega == _env.NODE_INTERIOR])

    # Passage en vecteur
    g = g.ravel()

    # Résolution du problème adjoint
    p_vector = scipy.sparse.linalg.spsolve( mat.conjugate().transpose(),g)

    p = p_vector.reshape((M, N))

    print("Norme de l'adjoint :", numpy.linalg.norm(p))

    # Calcul du gradient par rapport à chi
    grad = numpy.zeros((M, N), dtype=numpy.float64)

    for i in range(1, M - 1):
        for j in range(1, N - 1):

            if domain_omega[i, j] != _env.NODE_INTERIOR:
                continue

            row = i * N + j

            voisins = [
                (i - 1, j),
                (i + 1, j),
                (i, j - 1),
                (i, j + 1)
            ]

            for a, b in voisins:

                if domain_omega[a, b] == _env.NODE_ROBIN:

                    alpha = Alpha * chi[a, b]
                    beta = beta_pde[i, j]
                    beta_R = beta_rob[a, b]
                    h = spacestep

                    denom = beta_R - alpha * h / beta

                    # Derivee du coefficient diagonal de A
                    dA = Alpha * h * beta_R / (beta * denom**2)

                    # Derivee du second membre b
                    db = f_rob[a, b] * h**2 * Alpha / (
                        beta * denom**2
                    )

                    # Gradient reel
                    grad[a, b] += 2 * numpy.real(
                        numpy.conj(p_vector[row])
                        * (db - dA * u[i, j])
                    )

    print("Norme du gradient :", numpy.linalg.norm(grad))
    print("Gradient minimum :", numpy.min(grad))
    print("Gradient maximum :", numpy.max(grad))

    # Calcul de l'énergie finale après la descente de gradient
        # Sauvegarde de la répartition initiale
    chi_initial = chi.copy()

    # Descente de gradient
    pas = mu
    masque = domain_omega == _env.NODE_ROBIN

    for k in range(numb_iter):

        # Nouvelle répartition du matériau
        chi_new = chi - pas * grad

        # Projection : chi doit rester entre 0 et 1
        chi_new = numpy.clip(chi_new, 0, 1)

        # Le matériau reste uniquement sur la frontière de Robin
        chi_new[~masque] = 0

        # Conservation de la quantité totale de matériau
        quantite = numpy.sum(chi_initial)

        valeurs = chi_new[masque].copy()

        borne_inf = -1.0
        borne_sup = 1.0

        for _ in range(60):
            decalage = (borne_inf + borne_sup) / 2

            valeurs_projetees = numpy.clip(
                valeurs + decalage, 0, 1
            )

            if numpy.sum(valeurs_projetees) < quantite:
                borne_inf = decalage
            else:
                borne_sup = decalage

        chi_new[masque] = numpy.clip(
            valeurs + (borne_inf + borne_sup) / 2, 0, 1
        )

        # Nouveau champ acoustique
        u_new = processing.solve_helmholtz(
            domain_omega, spacestep, omega,
            f, f_dir, f_neu, f_rob,
            beta_pde, alpha_pde, alpha_dir,
            beta_neu, beta_rob, Alpha * chi_new
        )

        # Nouvelle énergie
        energie_new = your_compute_objective_function(
            domain_omega, u_new, spacestep
        )

        print("Pas :", pas)
        print("Énergie testée :", energie_new)
        print("Énergie actuelle :", energy[k, 0])
        print("Modification de chi :", numpy.linalg.norm(chi_new - chi))

        # On accepte uniquement si l'énergie diminue
        if energie_new < energy[numpy.count_nonzero(energy) - 1, 0]:

            chi = chi_new
            u = u_new
            energy[numpy.count_nonzero(energy), 0] = energie_new

            print("Itération", k + 1, ": énergie =", energie_new)

        else:
            pas = pas / 2

            if pas < 1e-5:
                break

            continue

        # Recalcul du gradient à la nouvelle répartition
        u, mat, rhs = processing.solve_helmholtz(
            domain_omega, spacestep, omega,
            f, f_dir, f_neu, f_rob,
            beta_pde, alpha_pde, alpha_dir,
            beta_neu, beta_rob, Alpha * chi,
            return_system=True
        )

        g = numpy.zeros((M, N), dtype=numpy.complex128)
        g[domain_omega == _env.NODE_INTERIOR] = (
            spacestep**2 * u[domain_omega == _env.NODE_INTERIOR]
        )

        p_vector = scipy.sparse.linalg.spsolve(
            mat.conjugate().transpose(), g.ravel()
        )

        grad = numpy.zeros((M, N), dtype=numpy.float64)

        for i in range(1, M - 1):
            for j in range(1, N - 1):

                if domain_omega[i, j] != _env.NODE_INTERIOR:
                    continue

                row = i * N + j

                for a, b in [
                    (i - 1, j), (i + 1, j),
                    (i, j - 1), (i, j + 1)
                ]:

                    if domain_omega[a, b] == _env.NODE_ROBIN:

                        alpha = Alpha * chi[a, b]
                        beta = beta_pde[i, j]
                        beta_R = beta_rob[a, b]
                        h = spacestep

                        denom = beta_R - alpha * h / beta

                        dA = Alpha * h * beta_R / (beta * denom**2)

                        db = f_rob[a, b] * h**2 * Alpha / (
                            beta * denom**2
                        )

                        grad[a, b] += 2 * numpy.real(
                            numpy.conj(p_vector[row])
                            * (db - dA * u[i, j])
                        )

    # On retire les valeurs non utilisées du tableau
    energy = energy[:numpy.count_nonzero(energy)]

    print("Énergie initiale :", energy[0, 0])
    print("Énergie finale :", energy[-1, 0])
    print("Réduction (%) :", 100 * (energy[0, 0] - energy[-1, 0]) / energy[0, 0])



#    while k < numb_iter and mu > 10**(-5):
#        print('---- iteration number = ', k)
#        print('1. computing solution of Helmholtz problem, i.e., u')


#        print('2. computing solution of adjoint problem, i.e., p')
#        print('3. computing objective function, i.e., energy')
#        print('4. computing parametric gradient')
#        while ene >= energy[k] and mu > 10 ** -5:
#            print('    a. computing gradient descent')
#            print('    b. computing projected gradient')
#            print('    c. computing solution of Helmholtz problem, i.e., u')
#            print('    d. computing objective function, i.e., energy (E)')
#            ene = compute_objective_function(domain_omega, u, spacestep)
#            if bool_a:
                # The step is increased if the energy decreased
#                mu = mu * 1.1
#            else:
                # The step is decreased is the energy increased
#                mu = mu / 2
#        k += 1

#    print('end. computing solution of Helmholtz problem, i.e., u')

    return chi, energy, u, grad
#mettre grad a la place de None

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

    #energy = 0.0

def your_compute_objective_function(domain_omega, u, spacestep):

    energy = numpy.sum(
        numpy.abs(u[domain_omega == _env.NODE_INTERIOR])**2
    ) * spacestep**2


    return energy


if __name__ == '__main__':

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- set parameters of the geometry
    N = 50  # number of points along x-axis
    M = 2 * N  # number of points along y-axis
    level = 0 # level of the fractal
    spacestep = 1.0 / N  # mesh size

    # -- set parameters of the partial differential equation
    kx = -1.0
    ky = -1.0
    wavenumber = numpy.sqrt(kx**2 + ky**2)  # wavenumber
    wavenumber = 10.0

    # ----------------------------------------------------------------------
    # -- Do not modify this cell, these are the values that you will be assessed against.
    # ----------------------------------------------------------------------
    # --- set coefficients of the partial differential equation
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = preprocessing._set_coefficients_of_pde(M, N)

    # -- set right hand sides of the partial differential equation
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)

    # -- set geometry of domain
    domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(M, N, level)

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

    # -- define material density matrix
    chi = preprocessing._set_chi(M, N, x, y)
    chi = preprocessing.set2zero(chi, domain_omega)

    # -- define absorbing material
    Alpha = 10.0 - 10.0 * 1j
    # -- this is the function you have written during your project
    #import compute_alpha
    #Alpha = compute_alpha.compute_alpha(...)
    alpha_rob = Alpha * chi

    # -- set parameters for optimization
    S = 0  # surface of the fractal
    for i in range(0, M):
        for j in range(0, N):
            if domain_omega[i, j] == _env.NODE_ROBIN:
                S += 1
    V_0 = 1  # initial volume of the domain
    V_obj = numpy.sum(numpy.sum(chi)) / S  # constraint on the density
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

    E0 = your_compute_objective_function(domain_omega, u0, spacestep)





    print("Énergie acoustique initiale :", E0)

    u_test, mat_test, rhs_test = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob, beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob, return_system=True)

    print("Dimensions de A :", mat_test.shape)
    print("Dimensions de b :", rhs_test.shape)
    print("Solutions identiques :", numpy.allclose(u0, u_test))



    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- compute optimization
    energy = numpy.zeros((100+1, 1), dtype=numpy.float64)


    chi, energy, u, grad = your_optimization_procedure(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob, beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob, Alpha, mu, chi, V_obj)
    # --- en of optimization



    chin = chi.copy()
    un = u.copy()

    # -- plot chi, u, and energy
    postprocessing._plot_uncontroled_solution(u0, chi0)
    postprocessing._plot_controled_solution(un, chin)
    err = un - u0
    postprocessing._plot_error(err)
    postprocessing._plot_energy_history(energy)

    print('End.')

# Test de la fonction objectif
u_test = numpy.ones((10, 10))
domain_test = numpy.full((10, 10), _env.NODE_INTERIOR)
h_test = 0.1

energie_test = your_compute_objective_function(
    domain_test, u_test, h_test
)

print("Énergie calculée :", energie_test)
print("Énergie attendue :", 1.0)