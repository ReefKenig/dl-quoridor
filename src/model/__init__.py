"""Neural network architecture for AlphaZero-style policy + value heads."""
try:
    from src.model.network import QuoridorModel, QuoridorNetwork  # noqa: F401
    from src.model.network_mp import QuoridorModelMP, QuoridorNetworkMP  # noqa: F401
    __all__ = ["QuoridorModel", "QuoridorNetwork",
               "QuoridorModelMP", "QuoridorNetworkMP"]
except ImportError:
    # torch not installed yet — network module won't be available
    __all__ = []
