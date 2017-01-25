import numpy as np
from matplotlib import pyplot as plt

from .spectrum import IdrSpectrum, Spectrum
from .tools import InvalidMetaDataException


class Supernova(object):
    def __init__(self, idr_directory, meta, restframe=True):
        self.idr_directory = idr_directory
        self.meta = meta

        # Load the spectra
        try:
            spectra_dict = self.meta['spectra']
        except KeyError:
            # Whatever this is, it isn't a supernova in the IDR. Sometimes
            # these slip in weirdly.
            raise InvalidMetaDataException('Invalid SN metadata %s' % meta)

        all_spectra = []

        for exposure, exposure_data in spectra_dict.items():
            spectrum = IdrSpectrum(idr_directory, exposure_data, self,
                                   restframe=restframe)

            if spectrum is None:
                continue

            # Require both channels
            spec_meta = spectrum.meta
            if (('idr.spec_R' not in spec_meta) or
                    ('idr.spec_B' not in spec_meta)):
                continue

            all_spectra.append(spectrum)

        all_spectra = sorted(all_spectra, key=lambda spectrum: spectrum.phase)

        self.all_spectra = np.array(all_spectra)

    @property
    def name(self):
        return self.meta['target.name']

    def __str__(self):
        return self.name

    def __repr__(self):
        return 'Supernova(name="%s")' % (self.name,)

    def __getitem__(self, key):
        return self.meta[key]

    def __lt__(self, other):
        """Order by the string name"""
        return str(self) < str(other)

    def __le__(self, other):
        """Order by the string name"""
        return str(self) <= str(other)

    def __gt__(self, other):
        """Order by the string name"""
        return str(self) > str(other)

    def __ge__(self, other):
        """Order by the string name"""
        return str(self) >= str(other)

    @property
    def subset(self):
        return self.meta['idr.subset']

    def keys(self):
        """Return a list of keys for the meta"""
        return list(self.meta.keys())

    @property
    def spectra(self):
        """Return the list of spectra that are usable for this supernova.

        Spectra that have been flagged an unusable will not be in this list.
        """
        return np.array([i for i in self.all_spectra if i.usable])

    @property
    def unusable_spectra(self):
        """Return the list of spectra that are unusable for this supernova."""
        return np.array([i for i in self.all_spectra if not i.usable])

    @property
    def phases(self):
        return np.array([i.phase for i in self.spectra])

    def get_nearest_spectrum(self, phase, max_diff=None):
        """Return the nearest spectrum to a phase.

        If the nearest spectrum is off by more than max_diff, then None is
        returned
        """
        if len(self.spectra) == 0:
            return None

        phases = self.phases
        diff = np.abs(phases - phase)

        min_idx = np.argmin(diff)
        min_diff = diff[min_idx]

        if max_diff is not None and min_diff > max_diff:
            return None

        return self.spectra[min_idx]

    def get_spectra_in_range(self, min_phase, max_phase):
        """Return a list of spectra within a phase range"""
        phases = self.phases

        use_idx = (phases < max_phase) & (phases > min_phase)

        return self.spectra[use_idx]

    def get_spectrum(self, exposure_id):
        """Return a spectrum with a specific exposure id"""
        for spectrum in self.all_spectra:
            if spectrum['obs.exp'] == exposure_id:
                return spectrum

        return None

    def get_next_spectrum(self, spectrum, backwards=False):
        """Return the spectrum after a given spectrum.

        spectrum can be either a Spectrum object or a phase
        """
        if isinstance(spectrum, Spectrum):
            phase = spectrum.phase
        else:
            phase = spectrum

        min_offset = None
        next_spectrum = None

        for other_spectrum in self.spectra:
            if ((backwards and (other_spectrum.phase < phase)) or
                    (not backwards and (other_spectrum.phase > phase))):
                offset = np.abs(other_spectrum.phase - phase)

                if (min_offset is None) or (offset < min_offset):
                    min_offset = offset
                    next_spectrum = other_spectrum

        return next_spectrum

    def get_interpolator(self, interpolator_name):
        from .interpolation import SupernovaInterpolator

        interpolator = SupernovaInterpolator.load_interpolator(
            interpolator_name, self.name
        )

        return interpolator

    def get_interpolated_spectra(self, interpolator_name, phases, wave):
        interpolator = self.get_interpolator(interpolator_name)

        flux = interpolator.get_flux(phases, wave)
        fluxvar = interpolator.get_fluxvar(phases, wave)

        # TODO: do this better. I just copy the meta from the first spectrum of
        # this supernova from now and do a couple hacks to update it. This
        # whole class really needs to be written...
        ref_spectrum = self.spectra[0]

        spectra = []
        for i in range(len(phases)):
            phase = phases[i]
            iter_flux = flux[i]
            iter_fluxvar = fluxvar[i]

            meta = ref_spectrum.meta.copy()
            meta['idrtools.phase'] = phase

            spectrum = ref_spectrum.get_modified_spectrum(
                "Loaded GP with phase %.2f" % phase,
                meta=meta,
                wave=wave,
                flux=iter_flux,
                fluxvar=iter_fluxvar,
                restframe=True
            )

            spectrum.usable = True

            spectra.append(spectrum)

        return spectra

    def get_interpolated_spectrum(self, interpolator_name, phase, wave):
        assert np.isscalar(phase)

        return self.get_interpolated_spectra(interpolator_name, [phase],
                                             wave)[0]

    def plot(self, show_error=False, **kwargs):
        """Plot the spectrum.

        If show_error is True, an error snake is also plotted.

        Any kwargs are passed to plt.plot.
        """
        spectra = self.spectra
        spectra = [i.bin_by_velocity(2000) for i in spectra]

        min_wave = np.min([i.wave for i in spectra])
        max_wave = np.max([i.wave for i in spectra])

        all_flux = [i.flux for i in spectra]
        offset_scale = np.percentile(np.abs(all_flux), 80) * 2.

        for i, spectrum in enumerate(self.spectra):
            spectrum.plot(
                offset=i*offset_scale
            )
            plt.text(
                1.01*max_wave,
                (i+0.1)*offset_scale,
                '%.2f days' % spectrum.phase
            )
            plt.xlim(min_wave, 1.15*max_wave)

        plt.xlabel('Restframe wavelength ($\\AA$)')
        plt.ylabel('Flux + offset')
        plt.title(self)
