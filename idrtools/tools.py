class IdrToolsException(Exception):
    pass


class InvalidDataException(IdrToolsException):
    pass


class InvalidMetaDataException(IdrToolsException):
    pass


class InterpolationException(IdrToolsException):
    pass


class SpectrumBoundsException(IdrToolsException):
    pass

snf_filters = {
    'u': (3300., 4102.),
    'b': (4102., 5100.),
    'v': (5200., 6289.),
    'r': (6289., 7607.),
    'i': (7607., 9200.)
}

try:
    # Add SNf filters to sncosmo if sncosmo is available
    import sncosmo

    for filter_name, filter_edges in snf_filters.items():
        band = sncosmo.Bandpass(
            filter_edges,
            [1., 1.],
            name='snf%s' % filter_name
        )

        sncosmo.register(band)
except ModuleNotFoundError:
    # sncosmo is required for fitting, but I don't always have it installed.
    # Everything but the fitting code will work fine.
    pass
