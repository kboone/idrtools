import numpy as np
from scipy.interpolate import RectBivariateSpline

from .spectrum import _get_snf_magnitude
from .tools import InterpolationException

"""Tools to handle interpolations of spectra.

For now, we only handle loading interpolations that have already been done (eg:
Clare and Hannah's GP runs).

We use interpolations on a fairly fine grid, and then use a RectBivariateSpline
to get the interpolation results at arbitrary wavelengths and phases.
"""

# Data on the different interpolation runs that are available.
interpolator_basedir = '/home/kyle/data/snfactory/gp/'
interpolator_meta = [
    # name, type, path
    ('hannah', 'hannah', interpolator_basedir + 'acev2_hannah/%s.dat'),
    ('caballo', 'clare', interpolator_basedir +
     'bedell_clare_2016_02_12/%s.predict'),
    ('alleg', 'clare', interpolator_basedir +
     'alleg_clare_2016_07_11/%s.predict'),
]


class SupernovaInterpolator(object):
    def __init__(self, phases, wave, flux, fluxvar, min_wave=None,
                 max_wave=None, min_phase=None, max_phase=None):
        """Initialize the interpolator with the given data.

        flux and fluxvar must be of the shape (len(phases), len(wave))
        """
        self.flux_spline = RectBivariateSpline(phases, wave, flux)
        self.fluxvar_spline = RectBivariateSpline(phases, wave, fluxvar)

        self._input_wave = wave
        self._input_phases = phases

        # Fill in the bounds if they weren't manually specified. We assume that
        # they are 1/2 of a bin over from the central wavelengths. This works
        # if the bins are roughly linearly spaced. For phases, we just use the
        # ends.
        if min_wave is None:
            min_wave = wave[0] - (wave[1] - wave[0]) / 2.
        if max_wave is None:
            max_wave = wave[-1] + (wave[-1] - wave[-2]) / 2.
        if min_phase is None:
            min_phase = phases[0]
        if max_phase is None:
            max_phase = phases[-1]

        self.min_wave = min_wave
        self.max_wave = max_wave
        self.min_phase = min_phase
        self.max_phase = max_phase

    def check_bounds(self, phase, wave):
        exception = None

        if np.any(phase < self.min_phase):
            exception = 'Phase must be > %f' % self.min_phase
        elif np.any(phase > self.max_phase):
            exception = 'Phase must be < %f' % self.max_phase
        elif np.any(wave < self.min_wave):
            exception = 'Wave must be > %f' % self.min_wave
        elif np.any(wave > self.max_wave):
            exception = 'Wave must be < %f' % self.max_wave

        if exception is not None:
            raise InterpolationException(exception)

    def get_flux(self, phase, wave):
        self.check_bounds(phase, wave)

        flux_interp = self.flux_spline(phase, wave)

        if np.isscalar(wave):
            flux_interp = flux_interp[:, 0]
        if np.isscalar(phase):
            flux_interp = flux_interp[0]

        return flux_interp

    def get_fluxvar(self, phase, wave):
        self.check_bounds(phase, wave)

        fluxvar_interp = self.fluxvar_spline(phase, wave)

        if np.isscalar(wave):
            fluxvar_interp = fluxvar_interp[:, 0]
        if np.isscalar(phase):
            fluxvar_interp = fluxvar_interp[0]

        return fluxvar_interp

    def get_fluxerr(self, phase, wave):
        return np.sqrt(self.get_fluxvar(phase, wave))

    def get_snf_magnitude(self, filter_name, phase, sampling=2.):
        """Return the AB magnitude in a given SNfactory filter."""
        wave = np.arange(self.min_wave, self.max_wave, sampling)

        scalar_phase = np.isscalar(phase)

        phase = np.atleast_1d(phase)

        flux = self.get_flux(phase, wave)

        all_mag = []

        for iter_phase, iter_flux in zip(phase, flux):
            mag = _get_snf_magnitude(wave, iter_flux, filter_name)
            all_mag.append(mag)

        all_mag = np.array(all_mag)

        if scalar_phase:
            all_mag = all_mag[0]

        return all_mag

    @classmethod
    def load_interpolator(cls, interpolator_name, supernova_name):
        # Find the interpolator information
        found = False
        all_names = []

        for iter_interp_name, interp_type, interp_path in interpolator_meta:
            all_names.append(iter_interp_name)

            if iter_interp_name == interpolator_name:
                found = True
                break

        if not found:
            exception = (
                'Unknown GP name: %s. Valid names are: %s' %
                (interpolator_name, ', '.join(all_names))
            )
            raise InterpolationException(exception)

        interp_path = interp_path % supernova_name

        try:
            data = np.genfromtxt(interp_path)
        except IOError:
            exception = (
                'No interpolator found with name %s for %s' %
                (interpolator_name, supernova_name)
            )
            raise InterpolationException(exception)

        phases = data[:, 0]
        wave = data[:, 1]
        flux = data[:, 2]

        unique_phases = np.unique(phases)
        unique_wave = np.unique(wave)

        min_wave = unique_wave[0]
        max_wave = unique_wave[-1]

        if len(unique_phases) * len(unique_wave) != data.shape[0]:
            exception = ("Invalid interpolator format. Expecting a grid of "
                         "phases and wavelengths")
            raise InterpolationException(exception)

        if interp_type == 'hannah':
            fluxvar = data[:, 3]
        elif interp_type == 'clare':
            # A bit tricky. There is one column for each phase and covariance
            # terms between different phases for the same wavelength. We drop
            # the covariance terms and just keep the variance.
            unique_phases = np.unique(phases)
            start_phase = unique_phases[0]
            phase_step = unique_phases[1] - unique_phases[0]

            indices = 3 + np.asarray(np.around((phases - start_phase) /
                                               phase_step), dtype=int)

            fluxvar = np.array([data[i, j] for i, j in enumerate(indices)])
        else:
            exception = "Unknown GP type %s! Can't handle" % interp_type
            raise InterpolationException(exception)

        if interp_type == 'hannah' or interp_type == 'clare':
            # Overwrite with the real min and max wavelengths
            min_wave = 3300.
            max_wave = 8600.

            # These formats use pseudo-integrals to get the flux in every bin
            # so they are in units of erg/s/cm^2 / 2. Ugh. Divide by the bin
            # widths to get them back to normal units of erg/s/cm^2/A
            bin_edges = np.logspace(np.log10(min_wave), np.log10(max_wave),
                                    289)
            norm_factor = (bin_edges[1:] - bin_edges[:-1]) / 2.

            tile_norm_factor = np.tile(norm_factor, len(unique_phases))

            flux = flux / tile_norm_factor
            fluxvar = fluxvar / tile_norm_factor

        # Reformat the data into the grid format that is required.
        new_shape = (len(unique_phases), len(unique_wave))
        flux = np.reshape(flux, new_shape)
        fluxvar = np.reshape(fluxvar, new_shape)

        # Good to go, create a new SupernovaInterpolation object
        return cls(unique_phases, unique_wave, flux, fluxvar,
                   min_wave=min_wave, max_wave=max_wave)
