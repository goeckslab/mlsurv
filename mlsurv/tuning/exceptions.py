"""Tuning-specific exceptions."""


class NoTunerError(ValueError):
    """Raised when no tuner is registered for a model.

    Inherits ValueError so that external code catching ValueError
    from get_tuner() continues to work. Internal callers catch
    NoTunerError directly.
    """

    def __init__(self, model_name: str, available: list[str] | None = None):
        self.model_name = model_name
        self.available = available or []
        super().__init__(model_name)

    def __str__(self):
        msg = f"No tuner registered for '{self.model_name}'."
        if self.available:
            msg += f" Available: {self.available}"
        return msg

    def __reduce__(self):
        return (type(self), (self.model_name,), {"available": self.available})
