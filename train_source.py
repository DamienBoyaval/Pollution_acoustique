"""Source ferroviaire simplifiee, non calibree : voir README_SOURCE_TRAIN.md.

Les gaussiennes spatiales localisent les sources. Les enveloppes en log2(f)
ponderent leur amplitude harmonique ; elles ne sont pas des spectres mesures.
"""
import numpy as np
import _env


def spectral_amplitudes(frequency_hz, speed_kmh=300., reference_speed_kmh=300.,
                        rolling_peak_hz=1000., aero_peak_hz=500.,
                        rolling_width_octaves=1., aero_width_octaves=1.,
                        rolling_gain=1., aero_gain=1.):
    """Amplitudes relatives des forcages roulement/aero (pas des pascals).

    S(f) = exp(-0.5 * (log2(f/fc)/width)**2), A = gain*(V/Vref)**(n/2)*sqrt(S).
    n=3 pour le roulement et n=6 pour l'aero : lois en vitesse idealisees.
    fc et width sont des choix de scenario, fixes quand V varie.
    """
    frequency = np.asarray(frequency_hz, dtype=float)

    ratio = speed_kmh / reference_speed_kmh
    rolling = rolling_gain * ratio**1.5 * np.exp(
        -0.25 * (np.log2(frequency / rolling_peak_hz) / rolling_width_octaves)**2)
    aero = aero_gain * ratio**3 * np.exp(
        -0.25 * (np.log2(frequency / aero_peak_hz) / aero_width_octaves)**2)
    return rolling, aero


def gaussian_train_source(domain, spacestep, frequency_hz, speed_kmh=300.,
                          sigma=0.03, rolling_center=(0.9, 0.5),
                          aero_center=(0.5, 0.5), **spectral_parameters):
    """Forcage complexe a phase commune, nul hors de l'interieur.

    Centres (x, y) et sigma dans les coordonnees normalisees du solveur.
    Les positions et la largeur restent fixes lors d'un raffinement de grille.
    La somme coherente conserve l'hypothese du programme d'origine.
    """
    centers = np.asarray([rolling_center, aero_center], dtype=float)
    amplitudes = spectral_amplitudes(frequency_hz, speed_kmh, **spectral_parameters)
    y, x = np.indices(domain.shape) * spacestep
    source = np.zeros(domain.shape, dtype=complex)
    interior = domain == _env.NODE_INTERIOR
    for amplitude, (xc, yc) in zip(amplitudes, centers):
        source[interior] += amplitude * np.exp(
            -((x[interior] - xc)**2 + (y[interior] - yc)**2) / (2 * sigma**2))
    return source, amplitudes
