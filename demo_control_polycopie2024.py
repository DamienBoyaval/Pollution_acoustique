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
import train_source
#import solutions


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
    if not J0 > 0:
        raise ValueError("Energie initiale nulle ou non definie : impossible de normaliser.")
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
    mode = 'un_materiau'
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

    # Source harmonique a la meme frequence que k et Alpha.
    # Modele spectral simplifie : references et hypotheses dans README_SOURCE_TRAIN.md.
    frequency_hz = float(omega_physical / (2 * numpy.pi))
    V_kmh = 300.0
    V_ref = 300.0  # vitesse de reference, pas une vitesse de croisement calibree
    sigma = 0.03  # 1.5 mailles pour N=50 ; largeur fixe si le maillage change
    spectral_parameters = dict(
        reference_speed_kmh=V_ref,
        rolling_peak_hz=train_source.ROLLING_PEAK_HZ, aero_peak_hz=train_source.AERO_PEAK_HZ,
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
    Alpha = complex(Alpha_material)  # coefficient supplied for this material and omega
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

    print('End.')
