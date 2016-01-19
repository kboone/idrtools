from astropy.io import fits
import numpy as np


class Spectrum(object):
    def __init__(self, idr_directory, meta, supernova=None):
        self.idr_directory = idr_directory
        self.meta = meta.copy()
        self.supernova = supernova

    def __str__(self):
        return self.meta['idr.prefix']

    def __repr__(self):
        return '%s(name="%s")' % (type(self).__name__, str(self))

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
    def usable(self):
        try:
            return self.meta['idrtools.usable']
        except KeyError:
            # If the key isn't there, then the spectrum is usable by default.
            # It is best to only use this flag if the spectrum isn't usable, as
            # that makes merging metadata easier.
            return True

    @property
    def phase(self):
        return self.meta['salt2.phase']

    @property
    def target(self):
        return self.meta['target.name']

    @property
    def fluxerr(self):
        return np.sqrt(self.fluxvar)

    def bin_by_velocity(self, velocity=1000, min_wave=3300, max_wave=8600):
        """Bin the spectrum in velocity/log-wavelength space

        min_wave and max_wave are in angstroms, velocity is in km/s

        I don't do interpolation here, I just group by bin. This should be ok,
        and should make our bins more independent than interpolation, which is
        probably a good thing. The output is in units of erg/s/cm2/A
        """
        wave = self.wave
        flux = self.flux
        fluxvar = self.fluxvar

        # Find the right spacing for those bin edges. We get as close as we can
        # to the desired velocity.
        n_bins = int(round(
            np.log10(float(max_wave) / min_wave)
            / np.log10(1 + velocity/3.0e5)
            + 1
        ))
        bin_edges = np.logspace(np.log10(min_wave), np.log10(max_wave), n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # bin_widths = (bin_edges[1:] - bin_edges[:-1])
        new_wave = np.around(bin_centers)

        binned_flux = np.zeros(len(bin_centers))
        binned_fluxvar = np.zeros(len(bin_centers))
        bin_counts = np.zeros(len(bin_centers))
        new_index = 0
        for orig_index in range(len(wave)):
            if wave[orig_index] < bin_edges[new_index]:
                continue
            while (new_index < len(bin_centers)
                   and wave[orig_index] >= bin_edges[new_index+1]):
                new_index += 1
            if new_index >= len(bin_centers):
                break

            binned_flux[new_index] += flux[orig_index]
            binned_fluxvar[new_index] += fluxvar[orig_index]
            bin_counts[new_index] += 1

        bin_counts[bin_counts == 0] = 1.
        binned_flux /= bin_counts
        binned_fluxvar /= (bin_counts * bin_counts)

        modification = "Binned to %.0f km/s in range [%.0f, %.0f]" % (
            velocity, min_wave, max_wave
        )

        return self.get_modified_spectrum(
            modification,
            wave=new_wave,
            flux=binned_flux,
            fluxvar=binned_fluxvar
        )

    def apply_reddening(self, rv, ebv, color_law='CCM'):
        if color_law == 'CCM':
            y = 1.0 / (self.wave / 10000.0) - 1.82
            a = (1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 +
                 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7)
            b = (1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 -
                 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7)
            reddening = (a + b/rv) * rv * ebv
            scale = 10 ** (-0.4 * reddening)

        flux = self.flux * scale
        fluxvar = self.fluxvar * scale * scale

        modification = "Applied color law %s with R_V=%.2f, E(B-V)=%.3f" % (
            color_law, rv, ebv
        )

        return self.get_modified_spectrum(
            modification,
            flux=flux,
            fluxvar=fluxvar
        )

    def get_modified_spectrum(self, modification, idr_directory=None,
                              meta=None, wave=None, flux=None, fluxvar=None,
                              supernova=None):
        """Get a modified version of the current spectrum with new values.

        modification is a string indicating what modification was done.

        Any variables that aren't specified are taken from the current
        spectrum.
        """
        if idr_directory is None:
            idr_directory = self.idr_directory

        if meta is None:
            meta = self.meta

        if wave is None:
            wave = self.wave

        if flux is None:
            flux = self.flux

        if fluxvar is None:
            fluxvar = self.fluxvar

        if supernova is None:
            supernova = self.supernova

        try:
            modifications = self.modifications
        except AttributeError:
            modifications = []

        modifications.append(modification)

        return ModifiedSpectrum(
            idr_directory,
            meta,
            wave,
            flux,
            fluxvar,
            supernova,
            modifications
        )


class IdrSpectrum(Spectrum):
    def __init__(self, idr_directory, meta, supernova):
        super(IdrSpectrum, self).__init__(idr_directory, meta, supernova)

        # Lazy load the wave and flux when we actually use them. This makes
        # things a lot faster.
        self._wave = None
        self._flux = None
        self._fluxvar = None

        # Check if the spectrum is good or not. We drop everything that is
        # flagged in the IDR for now.
        try:
            if (self.meta['procB.Quality'] != 1
                    or self.meta['procR.Quality'] != 1):
                self.meta['idrtools.usable'] = False
        except KeyError:
            pass

    def do_lazyload(self):
        if self._wave is not None:
            return

        path = '%s/%s' % (self.idr_directory, self.meta['idr.spec_restframe'])
        fits_file = fits.open(path)

        cdelt1 = fits_file[0].header['CDELT1']
        naxis1 = fits_file[0].header['NAXIS1']
        crval1 = fits_file[0].header['CRVAL1']

        self._wave = crval1 + cdelt1 * np.arange(naxis1)
        self._flux = fits_file[0].data
        self._fluxvar = fits_file[1].data

        fits_file.close()

    @property
    def wave(self):
        self.do_lazyload()

        return self._wave

    @property
    def flux(self):
        self.do_lazyload()

        return self._flux

    @property
    def fluxvar(self):
        self.do_lazyload()

        return self._fluxvar


class ModifiedSpectrum(Spectrum):
    def __init__(self, idr_directory, meta, wave, flux, fluxvar=None,
                 supernova=None, modifications=[]):
        super(ModifiedSpectrum, self).__init__(idr_directory, meta, supernova)

        self.wave = wave
        self.flux = flux
        self.fluxvar = fluxvar

        self.modifications = modifications
