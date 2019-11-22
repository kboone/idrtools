import itertools
import numpy as np
import os
import pickle

from .target import Target
from .tools import IdrToolsException, InvalidMetaDataException


class Dataset(object):
    def __init__(self, idr_directory, targets, meta):
        self.idr_directory = idr_directory
        self.targets = sorted(targets)
        self.meta = meta

    @classmethod
    def from_idr(cls, idr_directory, load_both_headers=False):
        with open('%s/META.pkl' % (idr_directory,), 'rb') as idr_file:
            idr_meta = pickle.load(idr_file)

        all_targets = []

        for target_name, target_meta in idr_meta.items():
            try:
                target = Target(idr_directory, target_meta,
                                load_both_headers=load_both_headers)
                all_targets.append(target)
            except InvalidMetaDataException:
                # The IDR contains weird entries sometimes (eg: a key of
                # DATASET with no entries). Ignore them.
                pass

        return cls(idr_directory, all_targets, idr_meta)

    def __str__(self):
        return os.path.basename(self.idr_directory)

    def __repr__(self):
        return 'Dataset(idr="%s", num_targets=%d)' % (
            str(self), len(self.targets)
        )

    def filter(self, filter_obj):
        """Apply a filter to the dataset.

        filter_obj can be either a function, or a mask.

        if filter_obj is a function, then it should take a Target object,
        and return True/False indicating whether to keep or discard the object.

        A new Dataset object will be returned with the filter applied.
        """
        if hasattr(filter_obj, '__call__'):
            filter_targets = list(filter(filter_obj, self.targets))
        else:
            if len(self.targets) != len(filter_obj):
                raise IdrToolsException("Invalid selector (wrong length)")
            filter_targets = list(itertools.compress(self.targets, filter_obj))

        return Dataset(self.idr_directory, filter_targets, self.meta)

    def get_subset(self, subset):
        """Return a subset of the data, based on the idr.subset entry

        eg: training, validation, bad.
        """
        return self.filter(lambda target: target.subset == subset)

    def __getitem__(self, key):
        """Return a numpy array of properties for the targets.

        If key is a tuple, then a two-dimensional array is returned. The keys
        can be any property in the IDR.

        eg: dataset['target.name', 'salt2.X1']
        """
        if isinstance(key, str):
            key = (key,)
            single = True
        else:
            single = False

        data = [tuple((target[i] for i in key)) for target in self.targets]

        result = np.rec.fromrecords(data, names=key)

        if single:
            result = result[key[0]]

        return result

    def keys(self, intersection=False):
        """Return a list of keys that are available for the targets..

        This is a union of all keys by default. An intersection of available
        keys can be obtained by setting intersection=True
        """
        all_keys = set()

        for target in self.targets:
            if intersection:
                all_keys = all_keys.intersection(target.keys())
            else:
                all_keys = all_keys.union(target.keys())

        return all_keys

    def get_nearest_spectra(self, phase, max_diff=None):
        """Return the spectrum for each target closest to the given phase

        If the nearest spectrum for a target is off by more than max_diff,
        then it is omitted from the final sample.
        """
        all_spectra = []

        for target in self.targets:
            spectrum = target.get_nearest_spectrum(phase, max_diff)
            if spectrum is not None:
                all_spectra.append(spectrum)

        return np.array(all_spectra)

    def get_spectra_in_range(self, min_phase, max_phase):
        """Return a list of spectra within a phase range"""
        all_spectra = []

        for target in self.targets:
            spectra = target.get_spectra_in_range(min_phase, max_phase)
            all_spectra.extend(spectra)

        return np.array(all_spectra)

    def merge_metadata(self, pickle_path):
        """Merge the metadata from another pickle file"""
        with open(pickle_path, 'rb') as pickle_file:
            new_meta = pickle.load(pickle_file, encoding='latin1')

        for target, target_dict in new_meta.items():
            if target in self.meta:
                self.meta[target].update(target_dict)

    def cut_target(self, target_name):
        """Cut a single target from the dataset by name."""
        filter_targets = []
        for target in self.targets:
            if target.name != target_name:
                filter_targets.append(target)

        if len(filter_targets) == len(self.targets):
            raise IdrToolsException("No target found with name %s!" %
                                    target_name)

        return Dataset(self.idr_directory, filter_targets, self.meta)

    def cut_target_list(self, target_list, intersection=False):
        """Cut the dataset to a list of targets.

        If intersection is True, then targets in the list are kept.
        Otherwise, targets in the list are rejected.
        """
        target_list = np.genfromtxt(target_list, dtype=None)

        filter_targets = []

        for target in self.targets:
            in_list = target.name.encode('ascii') in target_list
            if ((intersection and in_list) or
                    (not intersection and not in_list)):
                filter_targets.append(target)

        return Dataset(self.idr_directory, filter_targets, self.meta)

    def cut_bad_spectra(self, spectra_list, intersection=False):
        """Cut a list of spectra from the dataset.

        If intersection is True, then spectra in the list are kept.
        Otherwise, spectra in the list are rejected.
        """
        spectra_list = np.genfromtxt(spectra_list, dtype=None)

        for target in self.targets:
            for spectrum in target.spectra:
                in_list = spectrum['obs.exp'] in spectra_list
                if ((intersection and not in_list) or
                        (not intersection and in_list)):
                    spectrum.usable = False

    def get_target(self, name):
        """Find a target that matches the given name.

        name can be the full name or a subset. If more than one item is
        matched, an Exception will be raised
        """

        name = name.lower()

        result = None

        for target in self.targets:
            if name in str(target).lower():
                if result is not None:
                    raise IdrToolsException(
                        "More than one target matches %s!" % name
                    )

                result = target

        return result

    def get_spectrum(self, exposure):
        """Find the spectrum that matchees the given name."""
        for target in self.targets:
            for spectrum in target.spectra:
                if spectrum['obs.exp'] == exposure:
                    return spectrum

        return None

    def do_salt_fits(self, path=None, overwrite=False):
        """Redo all of the SALT2 fits for this dataset.

        The fits are written out to path. If path is not specified, then the fits are
        put in the original IDR directory.
        """
        if path is None:
            path = os.path.join(self.idr_directory, 'idrtools_salt_fits.pkl')

        if os.path.exists(path) and not overwrite:
            raise IdrToolsException("SALT2 fits already exist at %s! Not overwriting!" %
                                    path)

        # Use tqdm for progress if available.
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(x):
                return x

        salt_fits = {}
        for target in tqdm(self.targets):
            salt_fits[target.name] = target.fit_salt()

        with open(path, 'wb') as pickle_file:
            pickle.dump(salt_fits, pickle_file)

    def load_salt_fits(self, path=None):
        """Load the SALT2 fits for this dataset.

        If path is not specified, then a default path is used in the original IDR
        directory.
        """
        if path is None:
            path = os.path.join(self.idr_directory, 'idrtools_salt_fits.pkl')

        if not os.path.exists(path):
            print("SALT2 fits not found at %s. Redoing them. This might take a while" %
                  path)
            self.do_salt_fits(path)

        with open(path, 'rb') as pickle_file:
            salt_fits = pickle.load(pickle_file)

        for target in self.targets:
            target.salt_fit = salt_fits[target.name]
