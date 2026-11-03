# -*- coding: utf-8 -*-


# Python packages
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import matplotlib.pyplot
import numpy 
import scipy.sparse.linalg


# MRG packages
import _env
import preprocessing
import processing
import postprocessing
import material_alpha
#import solutions


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
    tolerance_chi = 1e-6
    chi_stabilise = False
    energy = numpy.zeros((numb_iter+1, 1), dtype=numpy.float64)
    chi = your_project_density(chi, domain_omega, V_obj)
    grad = numpy.zeros((M, N), dtype=numpy.float64)
    print(f"Coefficient Alpha utilise : {Alpha}")
    print(f"Quantite de materiau imposee : {V_obj * numpy.count_nonzero(domain_omega == _env.NODE_ROBIN):.8f}")
    while k < numb_iter and mu > 10**(-2):
        print('---- iteration number = ', k)
        print('1. computing solution of Helmholtz problem, i.e., u')
        u, mat, rhs = processing.solve_helmholtz(
            domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
            beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob,
            Alpha * chi, return_system=True)
        print('2. computing solution of adjoint problem, i.e., p')
        p = your_compute_adjoint(domain_omega, u, spacestep, mat)
        print('3. computing objective function, i.e., energy')
        energy[k, 0] = your_compute_objective_function(domain_omega, u, spacestep)
        print(f"   Energie actuelle : {energy[k, 0]:.12f}")
        print('4. computing parametric gradient')
        grad = your_compute_gradient(domain_omega, u, p, spacestep,
                                     f_rob, beta_pde, beta_rob, Alpha, chi)
        print(f"   Norme de l'adjoint : {numpy.linalg.norm(p):.6e}")
        print(f"   Norme du gradient : {numpy.linalg.norm(grad):.6e}")
        ene = energy[k, 0]
        bool_a = False
        while ene >= energy[k, 0] and mu > 10 ** -5:
            print('    a. computing gradient descent')
            chi_test = chi - mu * grad
            print('    b. computing projected gradient')
            chi_test = your_project_density(chi_test, domain_omega, V_obj)
            print('    c. computing solution of Helmholtz problem, i.e., u')
            u_test = processing.solve_helmholtz(
                domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, Alpha * chi_test)
            print('    d. computing objective function, i.e., energy (E)')
            ene = your_compute_objective_function(domain_omega, u_test, spacestep)
            print(f"   Pas teste : {mu:.8g} | Energie testee : {ene:.12f}")
            print(f"   Variation de chi : {numpy.linalg.norm(chi_test - chi):.6e}")
            bool_a = numpy.isfinite(ene) and ene < energy[k, 0]
            print("   Essai accepte" if bool_a else "   Essai refuse : reduction du pas")
            if bool_a:
                # Evaluate the accepted change before replacing the current density.
                variation_chi = numpy.linalg.norm(chi_test - chi)
                chi_stabilise = variation_chi < tolerance_chi
                chi = chi_test
                u = u_test
                # The step is increased if the energy decreased
                mu = mu * 1.1
            else:
                # The step is decreased is the energy increased
                mu = mu / 2
                ene = energy[k, 0]  # force another trial, also for non-finite energy
        if not bool_a:
            break  # no rejected trial may replace the last accepted density
        k += 1
        energy[k, 0] = ene
        if chi_stabilise:
            break  # keep this accepted state and its energy in the history

    print('end. computing solution of Helmholtz problem, i.e., u')
    u, mat, rhs = processing.solve_helmholtz(
        domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob,
        Alpha * chi, return_system=True)
    energy[k, 0] = your_compute_objective_function(domain_omega, u, spacestep)
    p = your_compute_adjoint(domain_omega, u, spacestep, mat)
    grad = your_compute_gradient(domain_omega, u, p, spacestep,
                                 f_rob, beta_pde, beta_rob, Alpha, chi)
    print("\nBILAN DE L'OPTIMISATION")
    print(f"Iterations acceptees : {k}")
    print(f"Energie initiale : {energy[0, 0]:.12f}")
    print(f"Energie finale : {energy[k, 0]:.12f}")
    if energy[0, 0] > 0:
        print(f"Reduction : {100 * (1 - energy[k, 0] / energy[0, 0]):.4f} %")
    print(f"Quantite de materiau finale : {numpy.sum(chi):.8f}")
    print(f"Dernier pas : {mu:.8g}")
    if chi_stabilise:
        print(f"Arret : stabilisation de chi ({variation_chi:.6e} < {tolerance_chi:.1e})")
    elif k == numb_iter:
        print("Arret : limite d'iterations atteinte")
    else:
        print("Arret : pas inferieur au seuil de la boucle")
    return chi, energy[:k+1], u, grad


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
    if S == 0 or not numpy.isfinite(V_obj) or not 0 <= V_obj <= 1:
        raise ValueError('A nonempty Robin boundary and 0 <= V_obj <= 1 are required.')
    z = numpy.asarray(values, dtype=float)[masque]
    if not numpy.all(numpy.isfinite(z)):
        raise ValueError('Non-finite density.')
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


if __name__ == '__main__':
    # Choose 'un_materiau' to reproduce the previous single-frequency run.
    mode = 'comparaison'
    if mode == 'comparaison':
        from comparaison_materiaux import run_comparison
        run_comparison(length_m=1.0, sound_speed=340.0, max_iter=100)
        raise SystemExit(0)

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- set parameters of the geometry
    N = 50  # number of points along x-axis
    M = 2 * N  # number of points along y-axis
    level = 0 # level of the fractal
    spacestep = 1.0 / N  # mesh size

    # -- coefficients calculated by the team (exact table index, no interpolation)
    material = 'ROCK_WOOL'  # or 'RECYCLED_RUBBER'
    frequency_index = 0  # 0..999: first entry is 500 Hz, last entry is 4000 Hz
    length_m = 1.0  # physical length scale L, physical grid step = L/N
    sound_speed = 340.0
    omega_table, Alpha_table, error_table = material_alpha.load_material(material)
    omega_physical = omega_table[frequency_index]

    # -- set parameters of the partial differential equation
    kx = -1.0
    ky = -1.0
    wavenumber = numpy.sqrt(kx**2 + ky**2)  # wavenumber
    wavenumber, Alpha_material = material_alpha.solver_parameters(
        omega_physical, Alpha_table[frequency_index], length_m, sound_speed)
    wavenumber = float(wavenumber)
    if wavenumber * spacestep > 2 * numpy.pi / 10:
        raise ValueError('Mesh too coarse: increase N to keep at least 10 points per wavelength.')
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

    # -- set geometry of domain
    domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(M, N, level)

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # 1. Condition aux limites g (bord supérieur) : pas d'onde incidente externe
    # ----------------------------------------------------------------------
    f_dir[:, :] = 0.0

    # ----------------------------------------------------------------------
    # 2. Modélisation des sources f : Mur absorbant SUR LE CÔTÉ du train
    # ----------------------------------------------------------------------
    f[:, :] = 0.0

    # Paramètres du train
    V_kmh = 300.0  # Vitesse du train en km/h
    V_ref = 300.0  # Vitesse de croisement roulement/aéro (littérature SNCF/TGV)

    # Amplitudes de pression (Helmholtz : énergie |u|^2 varie en V^3 et V^6)
    A_roul = (V_kmh / V_ref) ** 1.5
    A_aero = (V_kmh / V_ref) ** 3.0

    # Configuration par rotation de la grille : le train roule SUR LE MUR DROIT (j = 49)
    # - Le sol/voie ferrée est la paroi droite (j = 49).
    # - Les roues sont au contact du mur droit : j_roul = 45 (à 4 mailles du sol j=49).
    # - Le pantographe est en hauteur vers l'intérieur : j_aero = 25 (au-dessus des roues).
    # - Le mur absorbant (Robin) est sur le côté du train, en i = 50.
    # - Le train circule à une distance fixe du mur absorbant : i_train = 44 (à 6 mailles du mur).
    i_train = 25              # Train centré au milieu de la longueur du mur (i = 25)
    j_roul = 45               # Roues au contact du mur droit (sol j=49)
    j_aero = 25               # Pantographe en hauteur (vers l'intérieur)

    sigma = 1.5 * spacestep   # Largeur spatiale de la source (lissage gaussien)

    # Remplissage du terme source f dans l'intérieur du domaine
    for i in range(1, 50):
        for j in range(1, N - 1):
            if domain_omega[i, j] == _env.NODE_INTERIOR:
                x_coord = j * spacestep
                y_coord = i * spacestep

                # Coordonnées des sources (y_train = distance latérale, x_* = hauteurs)
                y_train = i_train * spacestep
                x_roul  = j_roul * spacestep
                x_aero  = j_aero * spacestep

                # Profil gaussien pour le roulement
                dist2_roul = (x_coord - x_roul)**2 + (y_coord - y_train)**2
                f_roul = A_roul * numpy.exp(-dist2_roul / (2.0 * sigma**2))

                # Profil gaussien pour l'aérodynamique (pantographe)
                dist2_aero = (x_coord - x_aero)**2 + (y_coord - y_train)**2
                f_aero = A_aero * numpy.exp(-dist2_aero / (2.0 * sigma**2))

                # Somme des contributions
                f[i, j] = f_roul + f_aero

    # -- initialize
    alpha_rob[:, :] = - wavenumber * 1j

    # -- define material density matrix
    chi = preprocessing._set_chi(M, N, x, y)
    chi = preprocessing.set2zero(chi, domain_omega)

    # -- define absorbing material
    Alpha = complex(Alpha_material)  # coefficient supplied for this material and omega
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

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- compute optimization
    energy = numpy.zeros((100+1, 1), dtype=numpy.float64)
    chi, energy, u, grad = your_optimization_procedure(
        domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
        Alpha, mu, chi, V_obj)
    #chi, energy, u, grad = solutions.optimization_procedure(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
    #                    beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
    #                    Alpha, mu, chi, V_obj, mu1, V_0)
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
