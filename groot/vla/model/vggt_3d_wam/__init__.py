__all__ = ["VGGT3DWAMConfig", "VGGT3DWAMModel"]


def __getattr__(name: str):
    # Keep geometry/data imports lightweight; importing PreTrainedModel also
    # initializes optional DeepSpeed components in this research environment.
    if name == "VGGT3DWAMConfig":
        from .configuration import VGGT3DWAMConfig

        return VGGT3DWAMConfig
    if name == "VGGT3DWAMModel":
        from .model import VGGT3DWAMModel

        return VGGT3DWAMModel
    raise AttributeError(name)
