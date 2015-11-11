import itertools
import numpy as np
import os
import pickle

from .supernova import Supernova
from .tools import IdrToolsException, InvalidMetaDataException


class Dataset(object):
    def __init__(self, idr_directory, supernovae, meta):
        self.idr_directory = idr_directory
        self.supernovae = supernovae
        self.meta = meta

    @classmethod
    def from_idr(cls, idr_directory):
        idr_meta = pickle.load(open('%s/META.pkl' % (idr_directory,)))

        all_supernovae = []

        for sn_name, sn_meta in idr_meta.iteritems():
            try:
                supernova = Supernova(idr_directory, sn_meta)
                all_supernovae.append(supernova)
            except InvalidMetaDataException:
                # The IDR contains weird entries sometimes (eg: a key of
                # DATASET with no entries). Ignore them.
                pass

        return cls(idr_directory, all_supernovae, idr_meta)

    def __str__(self):
        return os.path.basename(self.idr_directory)

    def __repr__(self):
        return 'Dataset(idr="%s", num_sn=%d)' % (
            str(self), len(self.supernovae)
        )

    def filter(self, filter_obj):
        """Apply a filter to the dataset.

        filter_obj can be either a function, or a mask.

        if filter_obj is a function, then it should take a Supernova object,
        and return True/False indicating whether to keep or discard the object.

        A new Dataset object will be returned with the filter applied.
        """
        if hasattr(filter_obj, '__call__'):
            filter_supernovae = filter(filter_obj, self.supernovae)
        else:
            if len(self.supernovae) != len(filter_obj):
                raise IdrToolsException("Invalid selector (wrong length)")
            filter_supernovae = list(itertools.compress(self.supernovae,
                                                        filter_obj))

        return Dataset(self.idr_directory, filter_supernovae, self.meta)

    def get_subset(self, subset):
        """Return a subset of the data, based on the idr.subset entry

        eg: training, validation, bad.
        """
        return self.filter(lambda sn: sn.subset == subset)

    def __getitem__(self, key):
        """Return a numpy array of properties for the SNe.

        If key is a tuple, then a two-dimensional array is returned. The keys
        can be any property in the IDR.

        eg: dataset['target.name', 'salt2.X1']
        """
        if isinstance(key, basestring):
            key = (key,)
            single = True
        else:
            single = False

        data = [tuple((sn[i] for i in key)) for sn in self.supernovae]

        result = np.rec.fromrecords(data, names=key)

        if single:
            result = result[key[0]]

        return result

    def keys(self, intersection=False):
        """Return a list of keys that are available for the SNe.

        This is a union of all keys by default. An intersection of available
        keys can be obtained by setting intersection=True
        """
        all_keys = set()

        for sn in self.supernovae:
            if intersection:
                all_keys = all_keys.intersection(sn.keys())
            else:
                all_keys = all_keys.union(sn.keys())

        return all_keys

    def get_nearest_spectra(self, phase, max_diff=None):
        """Return the spectrum for each supernova closest to the given phase

        If the nearest spectrum for a supernova is off by more than max_diff,
        then it is omitted from the final sample.
        """
        all_spectra = []

        for sn in self.supernovae:
            spectrum = sn.get_nearest_spectrum(phase, max_diff)
            if spectrum is not None:
                all_spectra.append(spectrum)

        return all_spectra
