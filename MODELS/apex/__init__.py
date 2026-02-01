class amp:
    @staticmethod
    def initialize(models, optimizers=None, **kwargs):
        # En TPU no inicializamos Apex, XLA maneja la precisión
        return models, optimizers

    @contextmanager
    def scale_loss(loss, optimizer, **kwargs):
        yield loss # No escalamos pérdida manualmente en TPU