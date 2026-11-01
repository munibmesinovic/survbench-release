from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence


@dataclass(frozen=True)
class Dim:
    name: str
    kind: str
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[Sequence[Any]] = None
    log: bool = False
    repo_declared: bool = True

    def __post_init__(self):
        if self.kind == "categorical":
            if not self.choices:
                raise ValueError(f"{self.name}: categorical needs choices")
        elif self.low is None or self.high is None:
            raise ValueError(f"{self.name}: {self.kind} needs low and high")

        if self.choices is not None:
            object.__setattr__(self, "choices", tuple(self.choices))


@dataclass(frozen=True)
class SearchSpace:
    source: str
    repo_declared: bool
    dims: Sequence[Dim] = field(default_factory=tuple)

    def __post_init__(self):
        object.__setattr__(self, "dims", tuple(self.dims))
        if not self.source.strip():
            raise ValueError(
                "SearchSpace needs a source describing where its bounds come from")


TRANSFORMER = SearchSpace(
    source=(
        "Repo-declared. TransformerSurv is this repo's own model; bounds are centred on the shipped defaults."
    ),
    repo_declared=True,
    dims=[
        Dim("lr", "float", low=1e-5, high=1e-3, log=True),
        Dim("weight_decay", "float", low=1e-7, high=1e-2, log=True),
        Dim("d_model", "categorical", choices=[64, 128, 256]),
        Dim("n_layers", "int", low=1, high=4),
        Dim("n_heads", "categorical", choices=[2, 4, 8]),
        Dim("ff_dim", "categorical", choices=[128, 256, 512]),
        Dim("encoder_dropout", "float", low=0.0, high=0.5),
        Dim("head_dropout", "float", low=0.0, high=0.5),
        Dim("head_hidden", "categorical", choices=[16, 32, 64]),
        Dim("alpha", "float", low=0.1, high=0.9),
        Dim("batch_size", "categorical", choices=[64, 128, 256]),
    ])

_BOTTLENECK_CHOICES = ("none", 32, 64, 128)

COX = SearchSpace(
    source=(
        "Repo-declared. No published bounds for lifelines' penalizer; log-spaced grid selected on validation Ctd."
    ),
    repo_declared=True,
    dims=[Dim("penalizer", "float", low=1e-4, high=1e0, log=True)])


_DEEPHIT_SET_LAYERS = (1, 2, 3, 5)
_DEEPHIT_SET_NODES = (50, 100, 200, 300)
_DEEPHIT_WIDTHS = tuple(
    ",".join([str(n)] * layers)
    for layers in _DEEPHIT_SET_LAYERS
    for n in _DEEPHIT_SET_NODES
)

DEEPHIT = SearchSpace(
    source=(
        "Mixed. shared_dims, cs_dims, batch_size and alpha_rank follow Lee et al. (AAAI 2018) "
        "main_RandomSearch.py; lr, dropout and weight_decay are ours."
    ),
    repo_declared=True,
    dims=[
        Dim("shared_dims", "categorical", choices=_DEEPHIT_WIDTHS,
            repo_declared=False),
        Dim("cs_dims", "categorical", choices=_DEEPHIT_WIDTHS,
            repo_declared=False),
        Dim("batch_size", "categorical", choices=(32, 64, 128),
            repo_declared=False),
        Dim("alpha_rank", "categorical", choices=(0.1, 0.5, 1.0, 3.0, 5.0),
            repo_declared=False),
        Dim("lr", "float", low=1e-5, high=1e-3, log=True),
        Dim("dropout", "float", low=0.0, high=0.6),
        Dim("weight_decay", "float", low=1e-7, high=1e-2, log=True),
    ])


DYNAMIC_DEEPHIT = SearchSpace(
    source=(
        "Repo-declared. Bounds anchored on the fixed values in chl8856/Dynamic-DeepHit main.py "
        "(Lee et al., IEEE TBME 2020), widened one step each way."
    ),
    repo_declared=True,
    dims=[
        Dim("lr", "float", low=1e-5, high=1e-3, log=True),
        Dim("weight_decay", "float", low=1e-7, high=1e-3, log=True),
        Dim("batch_size", "categorical", choices=[16, 32, 64]),
        Dim("rnn_hidden", "categorical", choices=[50, 100, 200]),
        Dim("cs_hidden", "categorical", choices=[50, 100, 200]),
        Dim("rnn_layers", "int", low=1, high=3),
        Dim("dropout", "float", low=0.0, high=0.6),
        Dim("alpha", "float", low=0.01, high=1.0, log=True),
        Dim("gamma", "float", low=0.1, high=10.0, log=True),
    ])


DYSURV = SearchSpace(
    source=(
        "Repo-declared. Bounds anchored on this repo's DySurv defaults "
        "(Mesinovic et al., JAMIA 2024), widened one step each way."
    ),
    repo_declared=True,
    dims=[
        Dim("lr", "float", low=1e-4, high=1e-2, log=True),
        Dim("batch_size", "categorical", choices=[32, 64, 128]),
        Dim("encoded_features", "categorical", choices=[16, 32, 64]),
        Dim("weight_decay", "float", low=1e-6, high=1e-2, log=True),
    ])


_SPACES: Dict[str, SearchSpace] = {
    "cox": COX,
    "deephit": DEEPHIT,
    "dynamic_deephit": DYNAMIC_DEEPHIT,
    "dysurv": DYSURV,
    "transformer": TRANSFORMER,
}


def space_for(model: str, modalities: Sequence[str] = ("dynamic", "static")
              ) -> SearchSpace:
    from survbench import registry
    base = registry.model_space(model) or _SPACES.get(model)
    if base is None:
        raise KeyError(f"No search space for model {model!r}")
    if model != "transformer":
        return base

    extra = tuple(Dim(f"{m}_bottleneck", "categorical",
                      choices=_BOTTLENECK_CHOICES)
                  for m in ("icd", "radiology") if m in modalities)
    if not extra:
        return base
    return SearchSpace(source=base.source, repo_declared=base.repo_declared,
                       dims=tuple(base.dims) + extra)
