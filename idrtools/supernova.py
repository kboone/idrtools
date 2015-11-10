from .spectrum import Spectrum
from .tools import InvalidMetaDataException


class Supernova(object):
    def __init__(self, idr_directory, meta):
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

        for exposure, exposure_data in spectra_dict.iteritems():
            spectrum = Spectrum(idr_directory, exposure_data)
            if spectrum is not None:
                all_spectra.append(spectrum)

        self.spectra = all_spectra

    def __str__(self):
        return self.meta['target.name']

    def __repr__(self):
        return 'Supernova(name="%s")' % (str(self),)

    def __getitem__(self, key):
        return self.meta[key]

    @property
    def subset(self):
        return self.meta['idr.subset']

    def keys(self):
        """Return a list of keys for the meta"""
        return self.meta.keys()
