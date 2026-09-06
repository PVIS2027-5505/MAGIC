from .fvsrn_multires import FVSRNMultires

__all__ = ["FVSRNMultires", "build_backbone"]

def build_backbone(name: str, opt: dict):
    name = name.lower()
    if name == "fvsrn_multires":
        return FVSRNMultires(opt)
    raise ValueError(f"Unknown backbone {name!r}")
