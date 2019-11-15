import numpy as np
from matplotlib import pyplot as plt
from astropy.table import Table

try:
    import sncosmo
except ImportError:
    # sncosmo is required for fitting, but I don't always have it installed.
    # Just error out if the fit is called. in that case.
    sncosmo = None

from .spectrum import IdrSpectrum, Spectrum
from .tools import InvalidMetaDataException, IdrToolsException


class Target(object):
    def __init__(self, idr_directory, meta, restframe=True,
                 load_both_headers=False):
        self.idr_directory = idr_directory
        self.meta = meta

        # Load the spectra
        try:
            spectra_dict = self.meta['spectra']
        except KeyError:
            # Whatever this is, it isn't a valid target in the IDR. Sometimes
            # these slip in weirdly.
            raise InvalidMetaDataException('Invalid SN metadata %s' % meta)

        all_spectra = []

        for exposure, exposure_data in spectra_dict.items():
            spectrum = IdrSpectrum(idr_directory, exposure_data, self,
                                   restframe=restframe,
                                   load_both_headers=load_both_headers)

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
        return 'Target(name="%s")' % (self.name,)

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
        """Return the list of spectra that are usable for this target.

        Spectra that have been flagged as unusable will not be in this list.
        """
        return np.array([i for i in self.all_spectra if i.usable])

    @property
    def unusable_spectra(self):
        """Return the list of spectra that are unusable for this target."""
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
        # this target from now and do a couple hacks to update it. This
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

    def plot_lightcurve(self, min_wave, max_wave, **kwargs):
        """Plot a lightcurve integrated over the given wavelength range.

        Any kwargs are passed to plt.plot.
        """
        mags = np.array([i.get_magnitude(min_wave, max_wave) for i in
                         self.spectra])
        phases = self.phases

        plt.scatter(phases, mags, **kwargs)

        plt.xlabel('Phase (days)')
        plt.ylabel('Magnitude + offset')
        plt.title(self)

    def get_photometry(self, filters='BVR', **kwargs):
        redshift = self.meta['host.zhelio']
        day_max = self.meta['salt2.DayMax']

        data = []
        for spectrum in self.spectra:
            for filter_name in filters:
                flux, flux_error = spectrum.get_snf_band_flux(
                    filter_name, calculate_error=True, **kwargs
                )

                # SNf-pipeline calculated photometry. This doesn't work for
                # fitting because it isn't in restframe.
                # mag = spectrum.meta['mag.%sSNf' % filter_name.upper()]
                # mag_err = spectrum.meta['mag.%sSNf.err' % filter_name.upper()]
                # if np.isnan(mag) or np.isnan(mag_err):
                    # continue
                # flux = 10**(-0.4 * mag)
                # flux_error = mag_err * flux / (2.5 / np.log(10))

                # We are fitting with restframe data in the restframe. Need to
                # normalize the MJDs to account for this. We use the initial
                # SALT2 fit for the date of maximum, so the peak of the LC
                # should be close to 0 (although we don't include corrections
                # for B-max)
                mjd = spectrum.meta['obs.mjd']
                time = (mjd - day_max) / (1 + redshift)

                scaling = -20.
                zeropoint = 0.

                total_scale = 10**(-0.4 * (zeropoint + scaling))

                data.append({
                    'time': time,
                    'band': 'snf%s' % (filter_name.lower()),
                    'flux': flux / total_scale,
                    'fluxerr': flux_error / total_scale,
                    'zp': zeropoint,
                    'zpsys': 'ab',
                })

        data = Table(data)

        # Apply a correction to the error. Currently in the SNf pipeline, the
        # errors are rescaled so that the median is 0.05 mag acros the B, V and
        # R filters. This is an implementation of that.
        mag_scale = 2.5 / np.log(10)
        mag_error = mag_scale * data['fluxerr'] / data['flux']

        rescaled_mag_error = mag_error * 0.05 / np.median(mag_error)
        # rescaled_mag_error = np.sqrt(mag_error**2 + 0.05**2)

        rescaled_flux_error = rescaled_mag_error * data['flux'] / mag_scale

        data['fluxerr_unscaled'] = data['fluxerr']
        data['fluxerr'] = rescaled_flux_error
        data['magerr'] = rescaled_mag_error

        return data

    def fit_salt(self, filters='BVR', plot=False, use_previous_start=False,
                 start_vals={}):
        if sncosmo is None:
            raise IdrToolsException(
                "sncosmo not found. Install sncosmo for fitting"
            )

        photometry = self.get_photometry(filters)

        model = sncosmo.Model(source='salt2')

        # Note, everything has been shifted to restframe. The times have been
        # shifted using the previously estimated day of maximum to get to 0.

        model.set(z=0.)
        model.set(t0=0.)

        if use_previous_start:
            # Start at the result of the previous SALT2 fit.
            model.set(x1=self.meta['salt2.X1'])
            model.set(c=self.meta['salt2.Color'])

        model.set(**start_vals)

        result, fitted_model = sncosmo.fit_lc(
            photometry,
            model,
            ['t0', 'x0', 'x1', 'c'],
            bounds={
                'x1': (-10, 10),
                'c': (-1, 10),
                't0': (-5, 5),
            },
            guess_t0=False,
            modelcov=True,
        )

        if plot:
            sncosmo.plot_lc(photometry, model=fitted_model,
                            errors=result.errors)

        result_dict = {
            't0': fitted_model['t0'],
            'x0': fitted_model['x0'],
            'x1': fitted_model['x1'],
            'c': fitted_model['c'],

            't0_err': result['errors']['t0'],
            'x0_err': result['errors']['x0'],
            'x1_err': result['errors']['x1'],
            'c_err': result['errors']['c'],
        }

        return result_dict
