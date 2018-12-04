# idrtools -- A package to interact with the SNfactory Internal Data Releases

## Usage:

There are three main classes used to represent objects in the IDR: Dataset,
Target and Spectrum. To load a dataset, download the SNfactory IDR and extract
it to a directory that we'll call IDRDIR. You can then instantiate an IDR
object with the following code:

    from idrtools import Dataset
    dataset = Dataset.from_idr(IDRDIR)

From here, you can access the dataset's targets with:

    dataset.targets

There are lots of different methods to select various subsets of the targets.
A target has a list of spectra associated with it that can be accessed with:

    target.spectra

and various other methods to select subsets of the spectra. The spectrum's data
can be accessed with `spectrum.wave`, `spectrum.flux`, `spectrum.fluxerr`. Fits
files are only opened when first accessed. If you want to force the fits file
to be read, call `spectrum.do_lazyload()`.

All objects have a `.meta` attribute that will return a dictionary with all of
the metadata in the IDR for that object. Spectra will also have all of the fits
keywords associated with them in this meta dictionary after the fits file is
first loaded.

`math.py` contains a lot of utility functions to do things like binned or
windowed plotting of aggregate functions. There is a lot of functionality built
into this package, not all of which is fully functional! 
