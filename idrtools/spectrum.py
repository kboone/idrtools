"""
Spectrum.py
Author: Kyle Boone

Class to represent a spectrum and perform common operations on it.
"""

from astropy.io import fits
import numpy as np

from .tools import InvalidMetaDataException, SpectrumBoundsException, \
    InvalidDataException, IdrToolsException, snf_filters


def _get_band_flux(wave, flux, min_wave, max_wave, flux_err=None):
    """Calculate the AB flux for a given tophat filter.

    Flux should be in erg/cm^2/s/A to get the right normalization.

    At some point I should implement non-tophat filters here. Note that
    I am not handling edges properly... blah whatever close enough.
    """

    # Figure out the bin widths for the spectrum. We use half the
    # distance between the nearest two bins, which should be a good
    # approximation in most cases.
    bin_widths = np.zeros(len(wave))
    bin_widths[1:-1] = (wave[2:] - wave[:-2]) / 2.
    bin_widths[0] = bin_widths[1]
    bin_widths[-1] = bin_widths[-2]

    # Ensure that the filter is contained within the spectrum. We allow 1 bin
    # flexibility to avoid bin edge/bin center issues.
    if ((max_wave > wave[-1] + bin_widths[-1]) or
            (min_wave < wave[0] - bin_widths[0])):
        raise SpectrumBoundsException(
            'Filter with edges %d, %d is not contained within the spectrum '
            '(bounds: %d, %d)'
            % (min_wave, max_wave, wave[0] - bin_widths[0], wave[-1] +
               bin_widths[-1])
        )

    # Convert the flux from erg/cm^2/s/A to phot/cm^2/s/A
    h = 6.626070040e-34
    c = 2.99792458e18

    conversion_factor = 10**-7 / h / c * wave

    phot_flux = flux * conversion_factor

    # Find the reference flux for the AB mag system. This is 3.631e-20
    # erg/cm^2/s/Hz which we need to convert to phot/cm^2/s/A
    ref_flux = 3.631e-20 * conversion_factor * (c / wave**2)

    wave_cut = (wave > min_wave) & (wave < max_wave)

    sum_flux = np.sum((phot_flux*bin_widths)[wave_cut])
    sum_ref_flux = np.sum((ref_flux*bin_widths)[wave_cut])

    band_flux = sum_flux / sum_ref_flux

    if flux_err is None:
        return band_flux
    else:
        phot_flux_err = flux_err * conversion_factor
        sum_flux_var = np.sum(((phot_flux_err * bin_widths)**2)[wave_cut])

        band_flux_var = sum_flux_var / sum_ref_flux**2
        band_flux_err = np.sqrt(band_flux_var)

        return band_flux, band_flux_err


def _get_magnitude(wave, flux, min_wave, max_wave, flux_err=None):
    """Calculate the AB magnitude for a given tophat filter.

    Flux should be in erg/cm^2/s/A to get the right normalization.
    """
    band_flux = _get_band_flux(
        wave, flux, min_wave, max_wave, flux_err=flux_err
    )

    if flux_err is not None:
        band_flux, band_flux_err = band_flux

    mag = -2.5*np.log10(band_flux)

    if flux_err is None:
        return mag
    else:
        mag_err = (2.5 / np.log(10) / band_flux) * band_flux_err
        return mag, mag_err


def _get_snf_filter(filter_name, spec_restframe, restframe, redshift):
    """Retrieve the edges of an SNf filter"""
    min_wave, max_wave = snf_filters[filter_name.lower()]

    scale = 1.

    # Get the right combination of restframe/non-restframe filter and
    # spectrum.
    if spec_restframe and not restframe:
        scale = 1. - redshift
    elif not spec_restframe and restframe:
        scale = 1. + redshift

    min_wave *= scale
    max_wave *= scale

    return min_wave, max_wave


def _get_snf_magnitude(wave, flux, filter_name, spec_restframe=True,
                       restframe=True, redshift=0., flux_err=None):
    """Calculate the AB magnitude for a given SNf filter.

    These numbers will agree with the SNf filter data in the headers if you
    use the observer frame filters (i.e. restframe=False) and convert from
    Vega to AB (a constant).
    """
    min_wave, max_wave = _get_snf_filter(
        filter_name, spec_restframe, restframe, redshift
    )

    return _get_magnitude(wave, flux, min_wave, max_wave, flux_err=flux_err)


def _get_snf_band_flux(wave, flux, filter_name, spec_restframe=True,
                       restframe=True, redshift=0., flux_err=None):
    """Calculate the AB flux for a given SNf filter."""
    min_wave, max_wave = _get_snf_filter(
        filter_name, spec_restframe, restframe, redshift
    )

    return _get_band_flux(wave, flux, min_wave, max_wave, flux_err=flux_err)


def _recover_bin_edges(bin_centers):
    """Convert a list of bin centers to bin edges.

    We do a second order correction to try to get this as accurately as
    possible.

    For linear binning there is only machine precision error with either first
    or second order binning.

    For higher order binnings (eg: log), the fractional error is of order (dA /
    A)**2 for linear recovery and (dA / A)**4 for the second order recovery
    that we do here.
    """
    # First order
    o1 = (bin_centers[:-1] + bin_centers[1:]) / 2.

    # Second order correction
    o2 = 1.5*o1[1:-1] - (o1[2:] + o1[:-2]) / 4.

    # Estimate front and back edges
    f2 = 2*bin_centers[1] - o2[0]
    f1 = 2*bin_centers[0] - f2
    b2 = 2*bin_centers[-2] - o2[-1]
    b1 = 2*bin_centers[-1] - b2

    # Stack everything together
    bin_edges = np.hstack([f1, f2, o2, b2, b1])
    bin_starts = bin_edges[:-1]
    bin_ends = bin_edges[1:]

    return bin_starts, bin_ends


def _parse_wavelength_information(data_dict):
    """Parse wavelength information from a data dictionary.

    Internally we store the wavelength as a list of minimum as maximum
    for bins. However, users typically have ether a list of wavelengths
    or a list of bin edges. We support all of the above with the
    following keys:
    - wave or bin_centers
    - bin_edges
    - bin_starts, bin_ends

    This will return (None, None) if there wasn't any wavelength
    information.
    """
    # Count how many different keys we found. Multiple wavelength
    # definitions is not supported.
    num_wave_keys = 0
    bin_starts = None
    bin_ends = None

    # Directly specified starts and ends
    if 'bin_starts' in data_dict and 'bin_ends' in data_dict:
        num_wave_keys += 1
        bin_starts = data_dict['bin_starts']
        bin_ends = data_dict['bin_ends']

    # Bin edges specified
    if 'bin_edges' in data_dict:
        num_wave_keys += 1
        bin_edges = data_dict['bin_edges']
        bin_starts = bin_edges[:-1]
        bin_ends = bin_edges[1:]

    # Bin centers specified
    bin_centers = None
    if 'wave' in data_dict:
        num_wave_keys += 1
        bin_centers = data_dict['wave']
    if 'bin_centers' in data_dict:
        num_wave_keys += 1
        bin_centers = data_dict['bin_centers']
    if bin_centers is not None:
        bin_starts, bin_ends = _recover_bin_edges(bin_centers)

    if num_wave_keys > 1:
        error = 'Wavelength specified multiple times (keys: %s)!' % (
            list(data_dict.keys())
        )
        raise InvalidDataException(error)

    return bin_starts, bin_ends


def _parse_flux_information(data_dict):
    """
    Parse flux information from a data dictionary.

    Internally we store both the flux and the flux variance (if specified).
    The variance may be specified either as a variance (fluxvar) or an
    error (fluxerr)

    This will return (None, None) if there isn't any flux information.

    This function must be called after the wavelength information has
    already been parsed as it does some error checking.
    """
    # Count how many different fluxvar keys we found. Multiple flux
    # variance definitions is not supported.
    num_fluxvar_keys = 0
    flux = None
    fluxvar = None

    if 'flux' in data_dict:
        flux = data_dict['flux']
    if 'fluxvar' in data_dict:
        num_fluxvar_keys += 1
        fluxvar = data_dict['fluxvar']
    if 'fluxerr' in data_dict:
        num_fluxvar_keys += 1
        fluxerr = data_dict['fluxerr']
        fluxvar = fluxerr**2

    if num_fluxvar_keys > 1:
        error = 'Flux variance specified multiple times (keys: %s)!' % (
            list(data_dict.keys())
        )
        raise InvalidDataException(error)

    return flux, fluxvar


class Spectrum(object):
    def __init__(self, meta={}, target=None, **data_dict):
        """Initialize the spectrum.

        Data can be optionally passed in with a variety of keywords.
        """
        self.meta = self._get_default_meta()
        self.meta.update(meta)

        # Parse the data that was passed in.
        self._load_data(data_dict)

        self.target = target

    def _get_default_meta(self):
        """Default meta for a new spectrum"""
        meta = {}

        # Locations of keys. These will be used to map the various properties
        # of this class to physical values in the meta.
        meta['idrtools.keys.redshift'] = 'idrtools.redshift'
        meta['idrtools.keys.name'] = 'idrtools.name'
        meta['idrtools.keys.phase'] = 'idrtools.phase'
        meta['idrtools.keys.target_name'] = 'idrtools.target_name'

        # Default keys
        meta['idrtools.redshift'] = None
        meta['idrtools.name'] = None
        meta['idrtools.phase'] = None
        meta['idrtools.target_name'] = None

        return meta

    def _load_data(self, data_dict):
        """Load data from a data dict"""
        self._bin_starts, self._bin_ends = (
            _parse_wavelength_information(data_dict)
        )
        self._flux, self._fluxvar = _parse_flux_information(data_dict)

        self._validate_data()

    def _validate_data(self):
        """Validate the data in this spectrum.

        This will make sure that the data isn't nonsense (eg: different number
        of wavelength and flux elements) so that sensible errors are thrown.
        """
        error = None

        if self._bin_starts is None:
            if self._flux is not None or self._fluxvar is not None:
                error = 'Must specify the wavelengths for the spectrum!'
        else:
            if len(self._bin_starts) != len(self._bin_ends):
                error = 'bin_starts and bin_ends must be the same length!'

            num_wave = len(self._bin_starts)
            if num_wave != len(self._flux):
                error = ('Must have the same number of wavelength and flux'
                         'elements!')
            elif num_wave != len(self._fluxvar):
                error = ('Must have the same number of wavelength and fluxvar'
                         'elements!')

        if error is not None:
            raise InvalidDataException(error)

    def __str__(self):
        return self.name

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
    def name(self):
        return self.meta[self.meta['idrtools.keys.name']]

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
            try:
                del self.meta['idrtools.usable']
            except KeyError:
                # Not there, already good.
                pass
        else:
            self.meta['idrtools.usable'] = False

    @property
    def phase(self):
        return self.meta[self.meta['idrtools.keys.phase']]

    @property
    def redshift(self):
        return self.meta[self.meta['idrtools.keys.redshift']]

    @property
    def target_name(self):
        return self.meta[self.meta['idrtools.keys.target_name']]

    @property
    def bin_starts(self):
        return self._bin_starts

    @property
    def bin_ends(self):
        return self._bin_ends

    @property
    def flux(self):
        return self._flux

    @property
    def fluxvar(self):
        return self._fluxvar

    @property
    def wave(self):
        return (self.bin_starts + self.bin_ends) / 2.

    @property
    def bin_edges(self):
        # Can only do this if the bins are sequential.
        assert np.all(self.bin_starts[1:] == self.bin_ends[:-1])

        return np.hstack([self.bin_starts, self.bin_ends[-1]])

    @property
    def fluxerr(self):
        return np.sqrt(self.fluxvar)

    def apply_binning(self, modification=None, method='average_interpolate',
                      weighting='variance', integrate=False, interpolate=False,
                      **binning_data):
        """Bin the spectrum with the given bin edges.

        Any of the keywords listed in _parse_wavelength_information can be used
        to specify the binning.

        By default, the output is in units of erg/s/cm2/A. If integrate is
        True, then the output units are erg/s/cm2.

        There are several possible ways to do a rebinning:
        - 'average_point': takes the average value in each bin, scaled by the
        weights of the inputs. This method automatically downweights low
        signal-to-noise data, but doesn't necessarily weight all wavelengths
        within a bin fairly. Input fluxes are treated as being at the exact
        centers of their bins so neighbouring output bins are uncorrelated.
        - 'average_interpolate': Same as average_point, but the input fluxes
        are treated as being uniformly spread over their bin instead of a
        single point. This will produce a more continuous result which is
        necessary in cases such as fitting for a redshift. It will however
        introduce correlations between neighboring bins.
        - 'photometry' (TODO: not yet implemented) treats each bin as a small
        filter. This method treats all wavelengths fairly, but will be limited
        by the lowest signal-to-noise measurement in each bin.

        There are also several choices for how to weight the different flux
        measurements:
        - 'uniform': uniform weights given to each original measurement. Note
        that this implies a uniform weighting in whatever space the wavelengths
        were chosen in so there is an implicit prior.
        - 'variance': use the variance of the individual data points for the
        weighting.

        We require that the new binning is ordered and non overlapping. If the
        old binning is overlapping or non-continuous we will handle it.
        """
        new_bin_starts, new_bin_ends = (
            _parse_wavelength_information(binning_data)
        )

        if method == 'old':
            bin_edges = np.hstack([new_bin_starts, new_bin_ends[-1]])
            return self.apply_binning_old(bin_edges, modification=modification,
                                          method=method, integrate=integrate,
                                          interpolate=interpolate)
        elif method != 'average_interpolate' and method != 'average_point':
            raise IdrToolsException('Unsupported binning method %s' % method)

        old_bin_starts = self.bin_starts
        old_bin_ends = self.bin_ends
        old_flux = self.flux
        old_fluxvar = self.fluxvar

        if method == 'average_point':
            # Treat all old data like individual points.
            bin_centers = (old_bin_starts + old_bin_ends) / 2.
            old_bin_starts = bin_centers
            old_bin_ends = bin_centers

        num_new_bins = len(new_bin_starts)
        num_old_bins = len(old_bin_starts)

        if weighting == 'variance':
            if old_fluxvar is None:
                weights = np.ones(num_old_bins, dtype=float)
            else:
                weights = 1. / old_fluxvar
        elif weighting == 'uniform':
            weights = np.ones(num_old_bins, dtype=float)
        else:
            raise IdrToolsException('Unsupported weighting method %s' %
                                    weighting)

        new_flux_sum = np.zeros(num_new_bins)
        new_fluxvar_sum = np.zeros(num_new_bins)
        new_weights = np.zeros(num_new_bins)

        new_index = 0
        for old_index in range(num_old_bins):
            # Find index of start of old array in new array
            old_start = old_bin_starts[old_index]
            old_end = old_bin_ends[old_index]

            while True:
                if old_start < new_bin_starts[new_index]:
                    if new_index == 0:
                        break
                    new_index -= 1
                    continue

                if old_start > new_bin_ends[new_index]:
                    if new_index == num_new_bins - 1:
                        break
                    new_index += 1
                    continue

                break

            if old_start > new_bin_ends[new_index]:
                continue

            # Split the old bin's data between the new bins.
            while new_bin_starts[new_index] < old_end:
                if method == 'average_interpolate':
                    # Figure out which fraction of the bin we have from the
                    # interpolation.
                    overlap_start = max(old_start, new_bin_starts[new_index])
                    overlap_end = min(old_end, new_bin_ends[new_index])
                    overlap = overlap_end - overlap_start

                    weight = (
                        weights[old_index] *
                        overlap / (old_end - old_start)
                    )
                elif method == 'average_point':
                    # Assign the old bin to this new bin.
                    weight = weights[old_index]

                new_weights[new_index] += weight
                new_flux_sum[new_index] += weight * old_flux[old_index]
                new_fluxvar_sum[new_index] += (
                    weight**2 * old_fluxvar[old_index]
                )

                if new_index == num_new_bins - 1:
                    break

                new_index += 1

            # We almost always go 1 past here, so jump back one to get the
            # search to start in (usually) the right place.
            if new_index > 1:
                new_index -= 1

        mask = new_weights == 0

        new_weights[mask] = 1.
        new_flux = new_flux_sum / new_weights
        new_fluxvar = new_fluxvar_sum / new_weights**2

        if integrate:
            bin_widths = new_bin_ends - new_bin_starts
            new_flux *= bin_widths
            new_fluxvar *= (bin_widths * bin_widths)

        new_flux[mask] = np.nan
        new_fluxvar[mask] = np.nan

        if modification is None:
            modification = "Rebinned to %d bins in range [%.0f, %.0f]" % (
                num_new_bins, new_bin_starts[0], new_bin_ends[-1]
            )

        return self.get_modified_spectrum(
            modification,
            bin_starts=new_bin_starts,
            bin_ends=new_bin_ends,
            flux=new_flux,
            fluxvar=new_fluxvar
        )

    def bin_by_wavelength(self, width=20, min_wave=3300, max_wave=8600,
                          **kwargs):
        """Bin the spectrum in wavelength space

        min_wave and max_wave are in angstroms, delta is the bin width in
        angstroms.
        """
        bin_edges = np.arange(min_wave, max_wave, width)

        modification = "Binned to %.0f Angstroms in range [%.0f, %.0f]" % (
            width, min_wave, max_wave
        )

        return self.apply_binning(modification, bin_edges=bin_edges, **kwargs)

    def bin_by_velocity(self, velocity=1000, min_wave=3300, max_wave=8600,
                        **kwargs):
        """Bin the spectrum in velocity/log-wavelength space

        min_wave and max_wave are in angstroms, velocity is in km/s
        """
        # Find the right spacing for those bin edges. We get as close as we can
        # to the desired velocity.
        n_bins = int(round(
            np.log10(float(max_wave) / min_wave) /
            np.log10(1 + velocity/3.0e5) + 1
        ))
        bin_edges = np.logspace(np.log10(min_wave), np.log10(max_wave), n_bins)

        modification = "Binned to %.0f km/s in range [%.0f, %.0f]" % (
            velocity, min_wave, max_wave
        )

        return self.apply_binning(modification, bin_edges=bin_edges, **kwargs)

    def apply_reddening(self, rv, ebv, color_law='CCM'):
        if color_law == 'CCM':
            y = 1.0 / (self.wave / 10000.0) - 1.82
            a = (1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 +
                 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7)
            b = (1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 -
                 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7)
            reddening = (a + b/rv) * rv * ebv
            scale = 10 ** (-0.4 * reddening)
        elif color_law == 'fm07':
            # TODO: Do this right. FM07 doesn't have a varying R_V!!!
            import extinction
            fm_color_law = extinction.fm07(self.wave, 1.)
            av = ebv * rv
            reddening = fm_color_law * av
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

    def shift_redshift(self, delta_redshift):
        """Shift a spectrum in redshift.

        Note: for now this doesn't account for a cosmology or time dilation
        """
        modification = "Shifted by %f in redshift." % delta_redshift
        return self.get_modified_spectrum(
            modification,
            wave=self.wave * (1 + delta_redshift),
        )

    def get_modified_spectrum(self, modification, meta=None, target=None,
                              restframe=None, **data_dict):
        """Get a modified version of the current spectrum with new values.

        modification is a string indicating what modification was done.

        Any variables that aren't specified are taken from the current
        spectrum.
        """
        if meta is None:
            meta = self.meta

        if target is None:
            target = self.target

        if restframe is None:
            restframe = self.restframe

        # Check for updates to the wavelength or flux
        bin_starts, bin_ends = _parse_wavelength_information(data_dict)
        flux, fluxvar = _parse_flux_information(data_dict)

        if bin_starts is None:
            bin_starts = self.bin_starts

        if bin_ends is None:
            bin_ends = self.bin_ends

        if flux is None:
            flux = self.flux

        if fluxvar is None:
            fluxvar = self.fluxvar

        try:
            modifications = self.modifications
        except AttributeError:
            modifications = []

        # Make a copy
        modifications = modifications + [modification]

        return ModifiedSpectrum(
            meta,
            target,
            restframe,
            bin_starts,
            bin_ends,
            flux,
            fluxvar,
            modifications
        )

    def plot(self, show_error=False, offset=0., **kwargs):
        """Plot the spectrum.

        If show_error is True, an error snake is also plotted.
        If offset is non-zero, then the offset is added to the flux.

        Any kwargs are passed to plt.plot"""
        from matplotlib import pyplot as plt

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

    def get_magnitude(self, min_wave, max_wave, calculate_error=False):
        """Calculate the AB magnitude for a given tophat filter."""
        if calculate_error:
            flux_err = self.fluxerr
        else:
            flux_err = None

        return _get_magnitude(self.wave, self.flux, min_wave, max_wave,
                              flux_err)

    def get_snf_magnitude(self, filter_name, restframe=True,
                          calculate_error=False):
        """Calculate the AB magnitude for a given SNf filter.

        These numbers will agree with the SNf filter data in the headers if you
        use the observer frame filters (i.e. restframe=False) and convert from
        Vega to AB (a constant).
        """
        if calculate_error:
            flux_err = self.fluxerr
        else:
            flux_err = None

        return _get_snf_magnitude(self.wave, self.flux, filter_name,
                                  self.restframe, restframe, self.redshift,
                                  flux_err)

    def get_band_flux(self, min_wave, max_wave, calculate_error=False):
        """Calculate the AB flux for a given tophat filter."""
        if calculate_error:
            flux_err = self.fluxerr
        else:
            flux_err = None

        return _get_band_flux(self.wave, self.flux, min_wave, max_wave,
                              flux_err)

    def get_snf_band_flux(self, filter_name, restframe=True,
                          calculate_error=False):
        """Calculate the AB flux for a given SNf filter.

        These numbers will agree with the SNf filter data in the headers if you
        use the observer frame filters (i.e. restframe=False) and convert from
        Vega to AB (a constant).
        """
        if calculate_error:
            flux_err = self.fluxerr
        else:
            flux_err = None

        return _get_snf_band_flux(self.wave, self.flux, filter_name,
                                  self.restframe, restframe, self.redshift,
                                  flux_err)


class IdrSpectrum(Spectrum):
    def __init__(self, idr_directory, meta, target, restframe=True):
        super(IdrSpectrum, self).__init__(meta, target)

        # Lazy load the wave and flux when we actually use them. This makes
        # things a lot faster.
        self.idr_directory = idr_directory

        self.restframe = restframe

        # Update meta keys
        self.meta['idrtools.keys.name'] = 'idr.prefix'
        self.meta['idrtools.keys.target_name'] = 'target.name'

        # Find redshift
        if 'host.zcmb' in self.target.meta:
            redshift = self.target.meta['host.zcmb']
        elif 'StdStar' in self.target.meta['target.kind']:
            # This is a standard star. Redshift is 0.
            redshift = 0.
        else:
            # Unknown redshift
            raise InvalidMetaDataException('No redshift found for %s!' %
                                           target)
        self.meta['idrtools.redshift'] = redshift

        # Find phase key
        if 'salt2.phase' in self.meta:
            phase_key = 'salt2.phase'
        elif 'qmagn.phase' in self.meta:
            phase_key = 'qmagn.phase'
        else:
            raise InvalidMetaDataException('No phase key found for %s!' %
                                           target)
        self.meta['idrtools.keys.phase'] = phase_key

        # Check if the spectrum is good or not. We drop everything that is
        # flagged in the IDR with one of the "bad" flags.
        bad_flags = [
            "ES_PRIOR_POSITION",
            "ES_PRIOR_SEEING",
            "ES_PRIOR_AIRMASS",
            "ES_PRIOR_PARANGLE",
            "ES_MIS-CENTERED",
            "PFC_XNIGHT",
            "PFC_RELFLX",
        ]
        good_flags = [
            "ARTIFICIAL_ARC",
        ]
        try:
            all_flags = self.meta['procB.Flags'] + self.meta['procR.Flags']

            usable = True

            for flag in all_flags:
                if flag in bad_flags:
                    # print("Not using %s for %s" % (self, flag))
                    usable = False
                elif flag not in good_flags:
                    print("WARNING: Unknown flag %s!" % flag)

            self.usable = usable
        except KeyError:
            pass

    @property
    def path(self):
        """Find the path to the file"""
        if self.restframe:
            key = 'idr.spec_restframe'
        else:
            key = 'idr.spec_merged'

        try:
            path = '%s/%s' % (self.idr_directory, self.meta[key])
        except KeyError:
            if self.restframe and 'idr.spec_merged' in self.meta:
                print("Did you mean to set restframe=False?")
            raise

        return path

    def do_lazyload(self):
        if self._bin_starts is not None:
            return

        with fits.open(self.path) as fits_file:
            header = fits_file[0].header
            cdelt1 = header['CDELT1']
            naxis1 = header['NAXIS1']
            crval1 = header['CRVAL1']

            wave = crval1 + cdelt1 * np.arange(naxis1)
            self._bin_starts, self._bin_ends = _recover_bin_edges(wave)
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
            self.meta['fits.mjd'] = header['JD'] - 2400000.5
            self.meta['fits.latitude'] = header['LATITUDE']
            self.meta['fits.longitude'] = header['LONGITUD']
            self.meta['fits.efftime'] = header['EFFTIME']
            self.meta['fits.exptime'] = header['EXPTIME']
            # self.meta['fits.utc'] = header['UTC']

            # self.meta['fits.hinsid'] = header['HINSID']
            # self.meta['fits.hintube'] = header['HINTUBE']
            # self.meta['fits.houtsd'] = header['HOUTSD']
            # self.meta['fits.tdmoil'] = header['TDMOIL']
            # self.meta['fits.tdmwal'] = header['TDMWAL']
            # self.meta['fits.tglycl'] = header['TGLYCL']
            # self.meta['fits.tinsid'] = header['TINSID']
            # self.meta['fits.tintube'] = header['TINTUBE']
            # self.meta['fits.toutsd'] = header['TOUTSD']
            # self.meta['fits.tpboil'] = header['TPBOIL']
            # self.meta['fits.tpier1'] = header['TPIER1']
            # self.meta['fits.tpmirr'] = header['TPMIRR']
            # self.meta['fits.ttbase'] = header['TTBASE']
            # self.meta['fits.ttdome'] = header['TTDOME']
            # self.meta['fits.tttube'] = header['TTTUBE']
            # self.meta['fits.pressure'] = header['PRESSURE']
            # self.meta['fits.temp'] = header['TEMP']
            # self.meta['fits.humidity'] = header['HUMIDITY']

            # self.meta['fits.rdnoise1'] = header['RDNOISE1']
            # self.meta['fits.rdnoise2'] = header['RDNOISE2']
            # self.meta['fits.ovscmed1'] = header['OVSCMED1']
            # self.meta['fits.ovscmed2'] = header['OVSCMED2']
            # self.meta['fits.saturat1'] = header['SATURAT1']
            # self.meta['fits.saturat2'] = header['SATURAT2']

            # self.meta['fits.arcoff'] = header['ARCOFF']
            # self.meta['fits.arcsca'] = header['ARCSCA']
            # self.meta['fits.scioff'] = header['SCIOFF']
            # self.meta['fits.scisca'] = header['SCISCA']

            # self.meta['es.chi2'] = header['ES_CHI2']
            self.meta['es.airm'] = header['ES_AIRM']
            self.meta['es.paran'] = header['ES_PARAN']
            self.meta['es.xc'] = header['ES_XC']
            self.meta['es.yc'] = header['ES_YC']
            self.meta['es.xy'] = header['ES_XY']
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
    def bin_starts(self):
        self.do_lazyload()
        return super(IdrSpectrum, self).bin_starts

    @property
    def bin_ends(self):
        self.do_lazyload()
        return super(IdrSpectrum, self).bin_ends

    @property
    def flux(self):
        self.do_lazyload()
        return super(IdrSpectrum, self).flux

    @property
    def fluxvar(self):
        self.do_lazyload()
        return super(IdrSpectrum, self).fluxvar


class ModifiedSpectrum(Spectrum):
    def __init__(self, meta, target, restframe, bin_starts, bin_ends, flux,
                 fluxvar, modifications=[]):
        super(ModifiedSpectrum, self).__init__(meta, target)

        self._bin_starts = bin_starts
        self._bin_ends = bin_ends
        self._flux = flux
        self._fluxvar = fluxvar

        self.restframe = restframe

        self.modifications = modifications
