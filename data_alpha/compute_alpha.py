# -*- coding: utf-8 -*-


# Python packages
import os
import matplotlib.pyplot
import numpy
import scipy
from scipy.optimize import minimize
import scipy.io


def real_to_complex(z):
    return z[0] + 1j * z[1]


def complex_to_real(z):
    return numpy.array([numpy.real(z), numpy.imag(z)])


class Memoize:
    def __init__(self, f):
        self.f = f
        self.memo = {}

    def __call__(self, *args):
        if args not in self.memo:
            self.memo[args] = self.f(*args)
        # .. todo: deepcopy here if returning objects
        return self.memo[args]


def compute_alpha(omega, material):
    """
    .. warning: $w = 2 \pi f$
    w is called circular frequency
    f is called frequency
    """

    porosity, resistivity, tortuosity = materials_data[material]

    # Birch LT
    phi = 0.529  # porosity
    gamma_p = 7.0 / 5.0
    sigma = 151429.0  # resitivity
    rho_0 = 1.2
    alpha_h = 1.37  # tortuosity
    c_0 = 340.0

    # parameters material
    phi = porosity
    gamma_p = 7.0 / 5.0
    sigma = resistivity
    rho_0 = 1.2
    alpha_h = tortuosity
    c_0 = 340.0

    # parameters of the geometry
    L = 0.01

    # parameters of the mesh
    resolution = 12  # := number of elements along L

    # parameters of the material (cont.)
    mu_0 = 1.0
    ksi_0 = 1.0 / (c_0 ** 2)
    mu_1 = phi / alpha_h
    ksi_1 = phi * gamma_p / (c_0 ** 2)
    a = sigma * (phi ** 2) * gamma_p / ((c_0 ** 2) * rho_0 * alpha_h)

    ksi_volume = phi * gamma_p / (c_0 ** 2)
    a_volume = sigma * (phi ** 2) * gamma_p / ((c_0 ** 2) * rho_0 * alpha_h)
    mu_volume = phi / alpha_h
    k2_volume = (1.0 / mu_volume) * ((omega ** 2) / (c_0 ** 2)) * (ksi_volume + 1j * a_volume / omega)
    # print(k2_volume)

    # parameters of the objective function
    A = 1.0
    B = 1.0

    # defining k, omega and alpha dependant parameters' functions
    @Memoize
    def lambda_0(k, omega):
        if k ** 2 >= (omega ** 2) * ksi_0 / mu_0:
            return numpy.sqrt(k ** 2 - (omega ** 2) * ksi_0 / mu_0)
        else:
            return numpy.sqrt((omega ** 2) * ksi_0 / mu_0 - k ** 2) * 1j

    @Memoize
    def lambda_1(k, omega):
        temp1 = (omega ** 2) * ksi_1 / mu_1
        temp2 = numpy.sqrt((k ** 2 - temp1) ** 2 + (a * omega / mu_1) ** 2)
        real = (1.0 / numpy.sqrt(2.0)) * numpy.sqrt(k ** 2 - temp1 + temp2)
        im = (-1.0 / numpy.sqrt(2.0)) * numpy.sqrt(temp1 - k ** 2 + temp2)
        return complex(real, im)

    @Memoize
    def g(y):
        # ..warning: not validated ***********************
        return 1.0

    @Memoize
    def g_k(k):
        # ..warning: not validated ***********************
        if k == 0:
            return 1.0
        else:
            return 0.0

    @Memoize
    def f(x, k):
        return ((lambda_0(k, omega) * mu_0 - x) * numpy.exp(-lambda_0(k, omega) * L) \
                + (lambda_0(k, omega) * mu_0 + x) * numpy.exp(lambda_0(k, omega) * L))

    @Memoize
    def chi(k, alpha, omega):
        return (g_k(k) * ((lambda_0(k, omega) * mu_0 - lambda_1(k, omega) * mu_1) \
                          / f(lambda_1(k, omega) * mu_1, k) - (lambda_0(k, omega) * mu_0 - alpha) / f(alpha, k)))

    @Memoize
    def eta(k, alpha, omega):
        return (g_k(k) * ((lambda_0(k, omega) * mu_0 + lambda_1(k, omega) * mu_1) \
                          / f(lambda_1(k, omega) * mu_1, k) - (lambda_0(k, omega) * mu_0 + alpha) / f(alpha, k)))

    @Memoize
    def e_k(k, alpha, omega):
        expm = numpy.exp(-2.0 * lambda_0(k, omega) * L)
        expp = numpy.exp(+2.0 * lambda_0(k, omega) * L)

        if k ** 2 >= (omega ** 2) * ksi_0 / mu_0:
            return ((A + B * (numpy.abs(k) ** 2)) \
                    * ( \
                                (1.0 / (2.0 * lambda_0(k, omega))) \
                                * ((numpy.abs(chi(k, alpha, omega)) ** 2) * (1.0 - expm) \
                                   + (numpy.abs(eta(k, alpha, omega)) ** 2) * (expp - 1.0)) \
                                + 2 * L * numpy.real(chi(k, alpha, omega) * numpy.conj(eta(k, alpha, omega)))) \
                    + B * numpy.abs(lambda_0(k, omega)) / 2.0 * ((numpy.abs(chi(k, alpha, omega)) ** 2) * (1.0 - expm) \
                                                                 + (numpy.abs(eta(k, alpha, omega)) ** 2) * (
                                                                             expp - 1.0)) \
                    - 2 * B * (lambda_0(k, omega) ** 2) * L * numpy.real(
                        chi(k, alpha, omega) * numpy.conj(eta(k, alpha, omega))))
        else:
            return ((A + B * (numpy.abs(k) ** 2)) * (L \
                                                     * ((numpy.abs(chi(k, alpha, omega)) ** 2) + (
                                numpy.abs(eta(k, alpha, omega)) ** 2)) \
                                                     + complex(0.0, 1.0) * (1.0 / lambda_0(k, omega)) * numpy.imag(
                        chi(k, alpha, omega) * numpy.conj(eta(k, alpha, omega) \
                                                          * (1.0 - expm))))) + B * L * (
                               numpy.abs(lambda_0(k, omega)) ** 2) \
                   * ((numpy.abs(chi(k, alpha, omega)) ** 2) + (numpy.abs(eta(k, alpha, omega)) ** 2)) \
                   + complex(0.0, 1.0) * B * lambda_0(k, omega) * numpy.imag(
                chi(k, alpha, omega) * numpy.conj(eta(k, alpha, omega) \
                                                  * (1.0 - expm)))

    @Memoize
    def sum_e_k(omega):
        def sum_func(alpha):
            s = 0.0
            for n in range(-resolution, resolution + 1):
                k = n * numpy.pi / L
                s += e_k(k, alpha, omega)
            return s

        return sum_func

    @Memoize
    def alpha(omega):
        alpha_0 = numpy.array(complex(40.0, -40.0))
        temp = real_to_complex(minimize(lambda z: numpy.real(sum_e_k(omega)(real_to_complex(z))), complex_to_real(alpha_0), tol=1e-4).x)
        # print(temp, "------", "je suis temp")
        return temp

    @Memoize
    def error(alpha, omega):
        temp = numpy.real(sum_e_k(omega)(alpha))
        return temp

    temp_alpha = alpha(omega)
    temp_error = error(temp_alpha, omega)

    return temp_alpha, temp_error


def run_compute_alpha(material):
    porosity, resistivity, tortuosity = materials_data[material]

    print('Computing alpha...')
    numb_omega = 1000  # 1000
    # omegas = numpy.logspace(numpy.log10(600), numpy.log10(30000), num=numb_omega)
    omegas = numpy.linspace(2.0 * numpy.pi * 500, 2.0 * numpy.pi * 4000, num=numb_omega)
    temp = [compute_alpha(omega, material) for omega in omegas]
    print("temp:", "------", temp)
    alphas, errors = map(list, zip(*temp))
    alphas = numpy.array(alphas)
    errors = numpy.array(errors)

    combined_data = numpy.zeros((len(omegas), 4))

    combined_data[:, 0] = numpy.real(omegas)
    combined_data[:, 1] = numpy.real(alphas)
    combined_data[:, 2] = numpy.imag(alphas)
    combined_data[:, 3] = numpy.real(errors)

    print('Writing alpha...')
    # output_filename = 'dta_omega_' + str(material) + '.mtx'
    # scipy.io.mmwrite(output_filename, omegas.reshape(alphas.shape[0], 1), field='complex', symmetry='general')
    # output_filename = 'dta_alpha_' + str(material) + '.mtx'
    # scipy.io.mmwrite(output_filename, alphas.reshape(alphas.shape[0], 1), field='complex', symmetry='general')
    # output_filename = 'dta_error_' + str(material) + '.mtx'
    # scipy.io.mmwrite(output_filename, errors.reshape(errors.shape[0], 1), field='complex', symmetry='general')

    output_filename = 'dta_alpha_' + str(material) + '.csv'
    header = "Frequency,Alpha_Real,Alpha_Imaginary,Error"
    
    numpy.savetxt(
        output_filename,
        combined_data,
        delimiter=",",
        header=header,
        comments=""
    )

    print(f'File saved as {output_filename}.')
    
    return


def run_compute_alpha_omegas(material, omegas):
    porosity, resistivity, tortuosity = materials_data[material]

    print('Computing alpha...')
    # numb_omega = 1000  # 1000
    # omegas = numpy.logspace(numpy.log10(600), numpy.log10(30000), num=numb_omega)
    # omegas = numpy.linspace(2.0 * numpy.pi * 500, 2.0 * numpy.pi * 4000, num=numb_omega)
    temp = [compute_alpha(omega, material) for omega in omegas]
    print("temp:", "------", temp)
    alphas, errors = map(list, zip(*temp))
    alphas = numpy.array(alphas)
    errors = numpy.array(errors)

    combined_data = numpy.zeros((len(omegas), 4))

    combined_data[:, 0] = numpy.real(omegas)
    combined_data[:, 1] = numpy.real(alphas)
    combined_data[:, 2] = numpy.imag(alphas)
    combined_data[:, 3] = numpy.real(errors)

    print('Writing alpha...')
    # output_filename = 'dta_omega_' + str(material) + '.mtx'
    # scipy.io.mmwrite(output_filename, omegas.reshape(alphas.shape[0], 1), field='complex', symmetry='general')
    # output_filename = 'dta_alpha_' + str(material) + '.mtx'
    # scipy.io.mmwrite(output_filename, alphas.reshape(alphas.shape[0], 1), field='complex', symmetry='general')
    # output_filename = 'dta_error_' + str(material) + '.mtx'
    # scipy.io.mmwrite(output_filename, errors.reshape(errors.shape[0], 1), field='complex', symmetry='general')

    output_filename = 'dta_alpha_omegas_' + str(material) + '.csv'
    header = "Frequency,Alpha_Real,Alpha_Imaginary,Error"
    
    numpy.savetxt(
        output_filename,
        combined_data,
        delimiter=",",
        header=header,
        comments=""
    )

    print(f'File saved as {output_filename}.')
    
    return


def run_plot_alpha(material):
    color = 'darkblue'

    print('Reading alpha...')
    input_filename = 'dta_omega_' + str(material) + '.mtx'
    omegas = scipy.io.mmread(input_filename)
    omegas = omegas.reshape(omegas.shape[0])
    input_filename = 'dta_alpha_' + str(material) + '.mtx'
    alphas = scipy.io.mmread(input_filename)
    alphas = alphas.reshape(alphas.shape[0])
    input_filename = 'dta_error_' + str(material) + '.mtx'
    errors = scipy.io.mmread(input_filename)
    errors = errors.reshape(errors.shape[0])

    print('Plotting alpha...')
    fig = matplotlib.pyplot.figure()
    matplotlib.pyplot.subplot(1, 1, 1)
    matplotlib.pyplot.plot(numpy.real(omegas), numpy.real(alphas), color=color)
    matplotlib.pyplot.xlabel(r'$\omega$')
    matplotlib.pyplot.ylabel(r'$\operatorname{Re}(\alpha)$')
    #matplotlib.pyplot.ylim(0, 35)
    # matplotlib.pyplot.show()
    matplotlib.pyplot.savefig('fig_alpha_real_' + str(material) + '.jpg')
    matplotlib.pyplot.close(fig)

    fig = matplotlib.pyplot.figure()
    matplotlib.pyplot.subplot(1, 1, 1)
    matplotlib.pyplot.plot(numpy.real(omegas), numpy.imag(alphas), color=color)
    matplotlib.pyplot.xlabel(r'$\omega$')
    matplotlib.pyplot.ylabel(r'$\operatorname{Im}(\alpha)$')
    #matplotlib.pyplot.ylim(-120, 10)
    # matplotlib.pyplot.show()
    matplotlib.pyplot.savefig('fig_alpha_imag_' + str(material) + '.jpg')
    matplotlib.pyplot.close(fig)

    fig = matplotlib.pyplot.figure()
    ax = matplotlib.pyplot.axes()
    ax.fill_between(numpy.real(omegas), numpy.real(errors), color=color)
    matplotlib.pyplot.ylim(1.e-9, 1.e-4)
    matplotlib.pyplot.yscale('log')
    matplotlib.pyplot.xlabel(r'$\omega$')
    matplotlib.pyplot.ylabel(r'$e(\alpha)$')
    # matplotlib.pyplot.show()
    matplotlib.pyplot.savefig('fig_error_' + str(material) + '.jpg')
    matplotlib.pyplot.close(fig)

    return


def run():

    for material in materials_data.keys():
        path = 'dta_alpha_' + str(material) + '.csv'

        if not os.path.isfile(path):
            print(f'Running alpha for {material}...')
            run_compute_alpha(material)
            #run_plot_alpha(material)

    omega_example = 2.0 * numpy.pi * 1000
    omegas = [omega_example]  # CHANGE HERE FOR CRITICAL FREQUENCIES

    for material in materials_data.keys():
        print(f'Running {material} alpha for critical frequencies...')
        run_compute_alpha_omegas(material, omegas)
        #run_plot_alpha(material)

    return

materials_data = {
    # (Material Name, Porosity phi, Resistivity sigma, Tortuosity alpha_h)
    'RECYCLED_RUBBER': (0.42, 18000.0, 1.95),
    'POROUS_CONCRETE': (0.38, 15000.0, 2.10),
    'ROCK_WOOL': (0.96, 32000.0, 1.02),
    'ALUMINIUM_FOAM': (0.78, 10000.0, 1.40),
    'WOOD_WOOL': (0.65, 12000.0, 1.75),
    'MELANINE_FOAM': (0.98, 11000.0, 1.02),
    'POROUS_ASPHALT': (0.25, 12000.0, 2.50),
    'RECYCLED_GLASS': (0.48, 19000.0, 1.85),
    'POLYESTER_FIBER': (0.95, 20000.0, 1.02),
    'HEMP_CONCRETE': (0.74, 8000.0, 1.80)
}

if __name__ == '__main__':
    run()
    print('End.')
