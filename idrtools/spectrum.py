
class Spectrum(object):
    def __init__(self, idr_directory, meta):
        self.idr_directory = idr_directory
        self.meta = meta

    def __str__(self):
        return self.meta['idr.prefix']

    def __repr__(self):
        return 'Spectrum(name="%s")' % (str(self),)

    def __getitem__(self, key):
        return self.meta[key]

    @property
    def phase(self):
        return self.meta['salt2.phase']
