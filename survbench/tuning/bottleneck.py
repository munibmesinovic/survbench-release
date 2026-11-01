from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

Width = Union[int, str]


class PerModalityBottleneck(nn.Module):
    def __init__(self, slot_map: Sequence[Tuple[str, Tuple[int, int]]],
                 widths: Dict[str, Width]):
        super().__init__()
        names = {n for n, _ in slot_map}
        for m in widths:
            if m not in names:
                raise ValueError(
                    f"bottleneck width given for '{m}', which is not in the slot map {sorted(names)}")
        self.slot_map = list(slot_map)
        self.widths = dict(widths)
        self.projections = nn.ModuleDict({
            name: nn.Linear(hi - lo, int(widths[name]))
            for name, (lo, hi) in slot_map
            if widths.get(name, "none") != "none"})

    @property
    def out_features(self) -> int:
        return sum(int(self.widths[n]) if self.widths.get(n, "none") != "none"
                   else hi - lo
                   for n, (lo, hi) in self.slot_map)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts: List[torch.Tensor] = []
        for name, (lo, hi) in self.slot_map:
            block = x[..., lo:hi]


            proj = self.projections[name] if name in self.projections else None
            parts.append(block if proj is None else proj(block))
        return torch.cat(parts, dim=-1)


BOTTLENECK_MODALITIES: Tuple[str, ...] = ("icd", "radiology")


def requested_widths(**per_modality: Width) -> Dict[str, Width]:
    return {name: width for name, width in per_modality.items()
            if width != "none"}


def dimension_name(modality: str) -> str:
    return f"{modality}_bottleneck"


def echo_widths(bottleneck: Optional[PerModalityBottleneck],
                names: Sequence[str] = BOTTLENECK_MODALITIES,
                ) -> Dict[str, Width]:
    projections = {} if bottleneck is None else bottleneck.projections
    return {dimension_name(name): (int(projections[name].out_features)
                                   if name in projections else "none")
            for name in names}
