# CST-Mock para evitar AttributeError en tl.extra
class ExtraMock:
    def __init__(self):
        self.cuda = type('CudaMock', (), {'libdevice': None})()
        self.libdevice = None

extra = ExtraMock()
