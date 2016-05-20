from astropy.io import fits
from matplotlib import pyplot as plt
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
            # I only set this flag if the spectrum is not usable which makes
            # merging metadata a lot easier (merging sets not usable data in
            # either set as not usable).
            return True

    @usable.setter
    def usable(self, usable):
        if usable:
            del self.meta['idrtools.usable']
        else:
            self.meta['idrtools.usable'] = False

    @property
    def phase(self):
        if 'salt2.phase' in self.meta:
            return self.meta['salt2.phase']
        else:
            return self.meta['qmagn.phase']

    @property
    def target(self):
        return self.meta['target.name']

    @property
    def fluxerr(self):
        return np.sqrt(self.fluxvar)

    def apply_binning(self, bin_edges, modification=None):
        """Bin the spectrum with the given bin edges.

        Note that the number of bins will be equal to len(bin_edges) - 1.
        """
        wave = self.wave
        flux = self.flux
        fluxvar = self.fluxvar

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

        if modification is None:
            modification = "Rebinned to %d bins in range [%.0f, %.0f]" % (
                len(bin_centers), np.min(bin_edges), np.max(bin_edges)
            )

        return self.get_modified_spectrum(
            modification,
            wave=new_wave,
            flux=binned_flux,
            fluxvar=binned_fluxvar
        )

    def bin_by_velocity(self, velocity=1000, min_wave=3300, max_wave=8600):
        """Bin the spectrum in velocity/log-wavelength space

        min_wave and max_wave are in angstroms, velocity is in km/s

        I don't do interpolation here, I just group by bin. This should be ok,
        and should make our bins more independent than interpolation, which is
        probably a good thing. The output is in units of erg/s/cm2/A
        """
        # Find the right spacing for those bin edges. We get as close as we can
        # to the desired velocity.
        n_bins = int(round(
            np.log10(float(max_wave) / min_wave)
            / np.log10(1 + velocity/3.0e5)
            + 1
        ))
        bin_edges = np.logspace(np.log10(min_wave), np.log10(max_wave), n_bins)

        modification = "Binned to %.0f km/s in range [%.0f, %.0f]" % (
            velocity, min_wave, max_wave
        )

        return self.apply_binning(bin_edges, modification)

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

    def apply_scale(self, scale):
        flux = self.flux * scale
        fluxvar = self.fluxvar * scale * scale

        modification = "Applied scale of %s" % scale

        return self.get_modified_spectrum(
            modification,
            flux=flux,
            fluxvar=fluxvar
        )

    def add_noise(self, fraction):
        """Add noise to a spectrum

        Fraction can be either a single number or a vector with the same length
        as wave.
        """
        noise_std = fraction*self.flux
        noise = noise_std * np.random.randn(len(self.wave))

        flux = self.flux + noise
        fluxvar = self.fluxvar + noise_std**2

        modification = "Applied noise of %s" % np.median(fraction)

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

        # Make a copy
        modifications = modifications + [modification]

        return ModifiedSpectrum(
            idr_directory,
            meta,
            wave,
            flux,
            fluxvar,
            supernova,
            modifications
        )

    def plot(self, show_error=False, offset=0., **kwargs):
        """Plot the spectrum.

        If show_error is True, an error snake is also plotted.
        If offset is non-zero, then the offset is added to the flux.

        Any kwargs are passed to plt.plot"""

        wave = self.wave
        flux = self.flux + offset
        plt.plot(wave, flux, **kwargs)

        if show_error:
            err = self.fluxerr

            plt.fill_between(
                wave,
                flux - err,
                flux + err,
                alpha=0.2,
                **kwargs
            )

        plt.xlabel('Restframe wavelength ($\\AA$)')
        plt.ylabel('Flux')
        plt.title(self)


class IdrSpectrum(Spectrum):
    def __init__(self, idr_directory, meta, supernova, restframe=True):
        super(IdrSpectrum, self).__init__(idr_directory, meta, supernova)

        # Lazy load the wave and flux when we actually use them. This makes
        # things a lot faster.
        self._wave = None
        self._flux = None
        self._fluxvar = None

        self.restframe = restframe

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

        if self.restframe:
            key = 'idr.spec_restframe'
        else:
            key = 'idr.spec_merged'

        try:
            path = '%s/%s' % (self.idr_directory, self.meta[key])
        except KeyError:
            if self.restframe and 'idr.spec_merged' in self.meta:
                print "Did you mean to set restframe=False?"
            raise

        with fits.open(path) as fits_file:
            fits_file = fits.open(path)

            header = fits_file[0].header
            cdelt1 = header['CDELT1']
            naxis1 = header['NAXIS1']
            crval1 = header['CRVAL1']

            self._wave = crval1 + cdelt1 * np.arange(naxis1)
            self._flux = np.copy(fits_file[0].data)
            self._fluxvar = np.copy(fits_file[1].data)

            self.meta['fits.insttemp'] = header['INSTTEMP']
            self.meta['fits.airmass'] = header['AIRMASS']
            self.meta['fits.altitude'] = header['ALTITUDE']
            self.meta['fits.azimuth'] = header['AZIMUTH']
            self.meta['fits.timeon'] = header['TIMEON']
            self.meta['fits.dettemp'] = header['DETTEMP']
            self.meta['fits.bcfocus'] = header['BCFOCUS']
            self.meta['fits.rcfocus'] = header['RCFOCUS']
            self.meta['fits.seeing'] = header['SEEING']

            self.meta['es.chi2'] = header['ES_CHI2']
            self.meta['es.airm'] = header['ES_AIRM']
            self.meta['es.paran'] = header['ES_PARAN']
            self.meta['es.xc'] = header['ES_XC']
            self.meta['es.yc'] = header['ES_YC']
            self.meta['es.xy'] = header['ES_XY']
            self.meta['es.lmin'] = header['ES_LMIN']
            self.meta['es.lmax'] = header['ES_LMAX']
            self.meta['es.e0'] = header['ES_E0']
            self.meta['es.a0'] = header['ES_A0']
            self.meta['es.a1'] = header['ES_A1']
            self.meta['es.a2'] = header['ES_A2']
            self.meta['es.tflux'] = header['ES_TFLUX']
            self.meta['es.sflux'] = header['ES_SFLUX']

            try:
                self.meta['cbft.snx'] = header['CBFT_SNX']
                self.meta['cbft.sny'] = header['CBFT_SNY']
            except KeyError:
                pass

            runid = header['RUNID']
            self.meta['fits.dayofyear'] = int(runid[3:6])

            ha_str = header['HA']
            ha_comps = ha_str.split(':')
            if ha_str[0] == '-':
                ha_sign = -1
            else:
                ha_sign = +1
            ha = ha_sign * (
                abs(int(ha_comps[0])) +
                int(ha_comps[1]) / 60. +
                float(ha_comps[2]) / 3600.
            )
            self.meta['fits.ha'] = ha

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
