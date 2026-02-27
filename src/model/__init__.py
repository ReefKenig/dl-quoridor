"""Neural network architecture for AlphaZero-style policy + value heads."""
try:
    from src.model.network import QuoridorModel, QuoridorNetwork
    __all__ = ["QuoridorModel", "QuoridorNetwork"]
except ImportError:
    # torch not installed yet — network module won't be available
    __all__ = []
