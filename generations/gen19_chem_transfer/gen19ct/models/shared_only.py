"""``models/shared_only.py`` -- the shared-only metal embedding of pre-registration section 11's last bullet (H3
negative-transfer investigation), implemented by POST-HOC addendum 2 item 4.

Addendum 2, "What changes" item 4: "Resolved: it is implemented in a NEW module (an H3-side subclass of
``neural.FactorisedNet`` with the ``e_series`` and ``e_ox`` offsets held at zero; no existing file edited) before
``scripts/g19_run_h3.py`` runs, and executes only under section 11's condition that WITHOUT beats WITH; otherwise D03
reports it as not run."  Section 11, last bullet of the negative-transfer investigation: "the WITH arm re-run with a
shared-only metal embedding (no ``e_series``, ``e_ox``) against the full section 15 embedding".

What is held at zero and how
----------------------------
:class:`SharedOnlyFactorisedNet` is ``neural.FactorisedNet`` with the two series / oxidation-state offset tables
``e_series`` and ``e_ox`` (i) zeroed after construction, (ii) excluded from the gradient (``requires_grad = False``, so
no optimizer step, weight decay or gradient clipping touches them) and (iii) left out of :meth:`metal_embedding`, which
becomes ``e_m = A z_m + e_element[z]``.  Everything else -- the shared linear map ``A``, the element offset, the ligand
side, the condition and head MLPs, the bilinear term, the initialisation stream, the training loop of
``neural.train_network`` (loss, optimizer, warm-up, batches, epochs, seed) -- is identical: the subclass consumes exactly
the RNG draws the base class does, so at the same model seed the two networks start from the same ``A``, MLP and ``P`` /
``Q`` values.  :func:`assert_offsets_zero` is run after every fit (the H3 arm refuses to return a network whose offsets
moved).

The module edits no existing file (``neural.py`` is in the running discovery's import closure): the arm class
:class:`SharedOnlyArm` overrides ``FactorisedArm.fit`` and constructs the network through :func:`shared_only_training`,
which rebinds ``neural.FactorisedNet`` to the subclass for the duration of the fit (``train_network`` constructs the
network BY NAME) and restores it afterwards.  A ladder step's network (``models.ladder.LadderNet``, itself a
``FactorisedNet`` subclass that inherits ``metal_embedding``) is covered the same way by :func:`shared_only_ladder_training`
(:func:`shared_only_subclass` builds the mixin subclass of any ``FactorisedNet`` descendant).
"""
from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator, Mapping
from typing import Any

import torch
from torch import nn

from gen19ct.models import neural as NN

#: the offset tables section 11's last bullet removes ("no e_series, e_ox")
ZERO_OFFSETS: tuple[str, ...] = ("e_series", "e_ox")
#: the label every prediction record and D03 row of the re-run carries
LABEL = "shared-only metal embedding (no e_series, e_ox); section 11 negative-transfer investigation, addendum 2 item 4"
READING = ("POST-HOC addendum 2 item 4: 'an H3-side subclass of neural.FactorisedNet with the e_series and e_ox offsets "
           "held at zero; no existing file edited' -- models.shared_only.SharedOnlyFactorisedNet zeroes both tables, "
           "freezes them (no gradient) and drops them from metal_embedding (e_m = A z_m + e_element[z]); the fit is "
           "otherwise neural.train_network unchanged (same seed, epochs, optimizer, loss), asserted zero after training")

_LOCK = threading.RLock()


class SharedOnlyMixin:
    """Holds ``e_series`` and ``e_ox`` at exactly zero (addendum 2 item 4) for any ``neural.FactorisedNet`` descendant."""

    shared_only: bool = True

    def _freeze_offsets(self) -> None:
        for name in ZERO_OFFSETS:
            emb = getattr(self, name)
            with torch.no_grad():
                emb.weight.zero_()
            emb.weight.requires_grad_(False)
        self.n_trainable_parameters = int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def metal_embedding(self, t: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """``e_m = A z_m + e_element[z]`` -- the shared map and the element offset only (section 11 last bullet)."""
        return self.e_shared(t["z_m"]) + self.e_element(t["element_id"])


class SharedOnlyFactorisedNet(SharedOnlyMixin, NN.FactorisedNet):
    """``neural.FactorisedNet`` with the series and oxidation-state offsets held at zero (module docstring).  Construct
    inside ``neural.deterministic_torch`` as the base class is; the constructor consumes the same RNG draws."""

    def __init__(self, dims: NN.InputDims, config: NN.NeuralConfig):
        super().__init__(dims, config)
        self._freeze_offsets()


_SUBCLASSES: dict[type, type] = {NN.FactorisedNet: SharedOnlyFactorisedNet}


def shared_only_subclass(base: type) -> type:
    """The shared-only subclass of ``base`` (a ``FactorisedNet`` descendant that inherits ``metal_embedding``): the base
    constructor, then :meth:`SharedOnlyMixin._freeze_offsets`.  Cached per base; ``FactorisedNet`` itself maps to
    :class:`SharedOnlyFactorisedNet`."""
    if getattr(base, "shared_only", False):
        return base
    if not (isinstance(base, type) and issubclass(base, NN.FactorisedNet)):
        raise TypeError(f"{base!r} is not a neural.FactorisedNet subclass")
    with _LOCK:
        if base in _SUBCLASSES:
            return _SUBCLASSES[base]
        # a descendant that overrides metal_embedding would keep its own offsets: refuse rather than silently miss them
        below = [k for k in base.__mro__ if k is not NN.FactorisedNet and issubclass(k, NN.FactorisedNet)]
        if any("metal_embedding" in k.__dict__ for k in below):
            raise TypeError(f"{base.__name__} overrides metal_embedding; the shared-only mixin would not remove the offsets")

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            base.__init__(self, *args, **kwargs)
            self._freeze_offsets()

        cls = type(f"SharedOnly{base.__name__}", (SharedOnlyMixin, base),
                   {"__init__": __init__, "__doc__": f"{base.__name__} with e_series and e_ox held at zero (addendum 2 item 4).",
                    "__module__": __name__})
        _SUBCLASSES[base] = cls
        return cls


def is_shared_only(net: nn.Module) -> bool:
    return bool(getattr(net, "shared_only", False))


def offsets_are_zero(net: nn.Module) -> dict[str, Any]:
    """Per offset table: max |weight|, whether it is exactly zero and whether it is frozen."""
    out: dict[str, Any] = {}
    for name in ZERO_OFFSETS:
        w = getattr(net, name).weight
        out[name] = {"max_abs": float(w.detach().abs().max()) if w.numel() else 0.0,
                     "zero": bool((w.detach() == 0).all()), "frozen": not bool(w.requires_grad)}
    return out


def assert_offsets_zero(net: nn.Module) -> dict[str, Any]:
    """Raise ``AssertionError`` unless both offset tables are exactly zero and frozen (checked after every fit)."""
    if not is_shared_only(net):
        raise AssertionError("not a shared-only network")
    st = offsets_are_zero(net)
    bad = [n for n, v in st.items() if not (v["zero"] and v["frozen"])]
    if bad:
        raise AssertionError(f"shared-only network: offsets {bad} are not zero and frozen after training: {st}")
    return st


# --------------------------------------------------------------------------------------------- #
# construction by name: the training loops build their network class through the module attribute
# --------------------------------------------------------------------------------------------- #

@contextlib.contextmanager
def patched_net(module: Any, attr: str) -> Iterator[list[nn.Module]]:
    """Rebind ``module.attr`` (a ``FactorisedNet`` descendant a training loop constructs BY NAME) to its shared-only
    subclass for the block; the original is restored on exit, every network constructed inside is collected in the
    yielded list, and on a normal exit each is asserted zero-and-frozen.  Re-entrant: an already patched attribute is
    left as it is.  No file is edited -- the binding lives only for the block, in this process."""
    orig = getattr(module, attr)
    made: list[nn.Module] = []
    if getattr(orig, "shared_only", False):
        yield made
        return
    cls = shared_only_subclass(orig)
    base_init = cls.__init__

    class _Collecting(cls):                                  # type: ignore[misc, valid-type]
        shared_only = True

        def __init__(self, *a: Any, **k: Any) -> None:
            base_init(self, *a, **k)
            made.append(self)

    _Collecting.__name__ = cls.__name__
    _Collecting.__qualname__ = cls.__qualname__
    with _LOCK:
        setattr(module, attr, _Collecting)
    try:
        yield made
    finally:
        with _LOCK:
            setattr(module, attr, orig)
    for net in made:
        assert_offsets_zero(net)


@contextlib.contextmanager
def shared_only_training() -> Iterator[list[nn.Module]]:
    """Every ``neural.train_network`` call inside the block builds :class:`SharedOnlyFactorisedNet` (M1 / M2 arms)."""
    with patched_net(NN, "FactorisedNet") as made:
        yield made


@contextlib.contextmanager
def shared_only_ladder_training() -> Iterator[list[nn.Module]]:
    """Every ``models.ladder.train_ladder_network`` call inside the block builds the shared-only ``LadderNet`` (the
    deployed ladder step M3-M7 of section 11's 'retained ladder configuration')."""
    from gen19ct.models import ladder as LAD

    with patched_net(LAD, "LadderNet") as made:
        yield made


# --------------------------------------------------------------------------------------------- #
# the arm: FactorisedArm at a fixed configuration, epoch count and seed, shared-only network
# --------------------------------------------------------------------------------------------- #

class SharedOnlyArm(NN.FactorisedArm):
    """``neural.FactorisedArm`` whose network is :class:`SharedOnlyFactorisedNet`: the WITH arm's SELECTED configuration,
    stopping count and model seed are passed unchanged (the H3 frozen-refit pattern); only the two offsets differ."""

    shared_only = True

    @classmethod
    def from_arm(cls, arm: NN.FactorisedArm) -> "SharedOnlyArm":
        """The shared-only twin of an unfitted (or fitted) ``FactorisedArm``: same config, epochs, seed and stores."""
        return cls(arm.config, n_epochs=arm.n_epochs, model_seed=arm.model_seed, rows=arm.rows,
                   condition_vectors=arm.condition_vectors)

    def clone(self) -> "SharedOnlyArm":
        return SharedOnlyArm(self.config, n_epochs=self.n_epochs, model_seed=self.model_seed, rows=self.rows,
                             condition_vectors=self.condition_vectors)

    def fit(self, train_rows, context=None) -> "SharedOnlyArm":
        with shared_only_training() as made:
            super().fit(train_rows, context)
        if self.result is None or not is_shared_only(self.result.net) or self.result.net not in made:
            raise AssertionError("SharedOnlyArm.fit did not train a shared-only network")
        self.offset_check = assert_offsets_zero(self.result.net)
        return self

    def fit_record(self) -> dict[str, Any]:
        rec = super().fit_record()
        net = self.result.net
        rec.update({"shared_only": True, "zero_offsets": list(ZERO_OFFSETS), "offset_check": offsets_are_zero(net),
                    "n_trainable_parameters": int(getattr(net, "n_trainable_parameters", rec["n_parameters"])),
                    "label": LABEL, "reading": READING})
        return rec
