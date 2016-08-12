class IdrToolsException(Exception):
    pass


class InvalidMetaDataException(IdrToolsException):
    pass


class InterpolationException(IdrToolsException):
    pass


class SpectrumBoundsException(IdrToolsException):
    pass
