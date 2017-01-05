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
