"""Read the colleague's paired omega/Alpha Matrix Market tables.

The supplied Alpha values are used in the professor's Robin convention.
Assumption for physical scaling: the supplied values are in m^-1.
No material identification or recalibration is performed here.
"""
from pathlib import Path
import io
import numpy as np
from scipy.io import mmread

MATERIALS = ('ROCK_WOOL', 'RECYCLED_RUBBER')


def load_material(material, directory=None):
    if material not in MATERIALS:
        raise ValueError('Choose ROCK_WOOL or RECYCLED_RUBBER.')
    directory = Path(directory) if directory is not None else Path(__file__).resolve().parent/'data_alpha'
    # Python opens Unicode Windows paths reliably; pass bytes to SciPy's reader.
    arrays = [np.asarray(mmread(io.BytesIO((directory/('dta_%s_%s.mtx'%(kind,material))).read_bytes()))).ravel()
              for kind in ('omega','alpha','error')]
    omega, alpha, error = arrays
    if len(omega) < 2 or not (omega.shape == alpha.shape == error.shape):
        raise ValueError('The omega, Alpha and error tables must have the same length.')
    if not all(np.all(np.isfinite(a)) for a in arrays):
        raise ValueError('Non-finite material data.')
    if np.any(omega.imag != 0) or np.any(error.imag != 0):
        raise ValueError('Omega and fitting error must be real.')
    omega, error = omega.real, error.real
    if np.any(omega <= 0) or np.any(np.diff(omega) <= 0) or np.any(error < 0):
        raise ValueError('Require increasing positive omega and nonnegative fitting errors.')
    return omega, alpha.astype(complex), error


def solver_parameters(omega, alpha, length_m=1., sound_speed=343.):
    """For the normalized geometry h=1/N: k=omega*L/c, Alpha_num=Alpha_phys*L."""
    if not np.isfinite(length_m) or length_m <= 0 or not np.isfinite(sound_speed) or sound_speed <= 0:
        raise ValueError('Positive finite length and sound speed required.')
    return np.asarray(omega)*length_m/sound_speed, np.asarray(alpha)*length_m
