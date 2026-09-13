"""A directed message-passing neural network over the extractant graph.

Why this is written here rather than imported
---------------------------------------------
The pre-registration names Chemprop v2 as the primary neural candidate and
allows "an equivalent D-MPNN implemented cleanly if repository constraints
require it".  They do.  ``chemprop==2.3.1`` installs without moving any of
numpy, pandas, scipy, scikit-learn, PyTorch, RDKit, CatBoost, XGBoost or pyarrow
— that was checked by dry run before installing — but it cannot be *imported* in
this environment: Lightning pulls ``torchmetrics``, which eagerly imports
``transformers`` for its text metrics, and the venv's ``transformers 5.15.1``
needs a newer ``huggingface_hub`` than the 0.36.2 the ChemBERTa embedding path of
gen7 is pinned against.  Repairing that means mutating a shared environment that
five earlier generations' numbers are defined against, and this repository has
already been bitten once by exactly that (a ``pip install`` silently moved every
result by ~0.0014 against effects of 0.004-0.02).

So the architecture is reproduced rather than imported.  It is the Yang et al.
(2019) D-MPNN that Chemprop v2 implements, in the same form:

* messages live on **directed bonds**, not atoms, so a message cannot immediately
  return along the edge it arrived on;
* ``h_vu^0 = ReLU(W_i [x_u ; e_uv])``;
* ``m_vu^{t+1} = sum_{k in N(u) \\ {v}} h_ku^t``, computed as
  ``(sum over all edges into u) - h_uv^t``, which is what makes the update O(E);
* ``h_vu^{t+1} = ReLU(h_vu^0 + W_h m_vu^{t+1})`` — a residual on the initial
  message, so depth does not wash out the atom's own identity;
* readout ``h_v = ReLU(W_o [x_v ; sum_{k in N(v)} h_kv^T])`` and a mean over atoms.

**Conditions enter after message passing**, as the brief requires: the standardised
experimental-condition vector (and, in one variant, the RDKit descriptor vector)
is concatenated to the learned molecular embedding and the pair goes to the
regression head.  Nothing about the experiment is written into an atom feature,
so the molecular representation stays a property of the molecule and the same
molecule under two acidities has one embedding and two predictions.

Efficiency note: this cohort has 183 distinct molecules and up to 1,329 rows, so
the encoder runs once per forward pass over the *unique* molecules of the batch
and the row-level embedding is an index into that, rather than re-encoding a
molecule once per row.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from .preprocess import (
    FoldPreprocessor, clip_to_training_range, extractant_balanced_weights,
)

TARGET = "log_D"
#: Atomic numbers that appear in these extractants plus a catch-all slot.
ATOM_VOCAB: tuple[int, ...] = (1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53)
DEGREE_VOCAB = (0, 1, 2, 3, 4, 5)
CHARGE_VOCAB = (-2, -1, 0, 1, 2)
NUM_H_VOCAB = (0, 1, 2, 3, 4)


def _one_hot(value, vocab) -> list[float]:
    out = [0.0] * (len(vocab) + 1)
    out[vocab.index(value) if value in vocab else len(vocab)] = 1.0
    return out


def _atom_features(atom) -> list[float]:
    from rdkit.Chem import rdchem
    hybrid = (rdchem.HybridizationType.SP, rdchem.HybridizationType.SP2,
              rdchem.HybridizationType.SP3, rdchem.HybridizationType.SP3D,
              rdchem.HybridizationType.SP3D2)
    return (_one_hot(atom.GetAtomicNum(), ATOM_VOCAB)
            + _one_hot(atom.GetTotalDegree(), DEGREE_VOCAB)
            + _one_hot(atom.GetFormalCharge(), CHARGE_VOCAB)
            + _one_hot(atom.GetTotalNumHs(), NUM_H_VOCAB)
            + _one_hot(atom.GetHybridization(), hybrid)
            + [float(atom.GetIsAromatic()), float(atom.IsInRing()),
               atom.GetMass() * 0.01, float(atom.GetChiralTag() != 0)])


def _bond_features(bond) -> list[float]:
    from rdkit.Chem import rdchem
    types = (rdchem.BondType.SINGLE, rdchem.BondType.DOUBLE,
             rdchem.BondType.TRIPLE, rdchem.BondType.AROMATIC)
    stereo = (rdchem.BondStereo.STEREONONE, rdchem.BondStereo.STEREOZ,
              rdchem.BondStereo.STEREOE)
    return (_one_hot(bond.GetBondType(), types) + _one_hot(bond.GetStereo(), stereo)
            + [float(bond.GetIsConjugated()), float(bond.IsInRing())])


@dataclass(frozen=True)
class MoleculeGraph:
    atom_features: np.ndarray      # [n_atoms, F_a]
    bond_features: np.ndarray      # [n_edges, F_b], directed, paired
    edge_source: np.ndarray        # [n_edges]
    edge_target: np.ndarray        # [n_edges]
    reverse_edge: np.ndarray       # [n_edges]


@lru_cache(maxsize=512)
def featurise(smiles: str) -> MoleculeGraph:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse {smiles!r}")
    atoms = np.array([_atom_features(a) for a in mol.GetAtoms()], dtype=np.float32)
    sources, targets, features = [], [], []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feature = _bond_features(bond)
        sources += [a, b]; targets += [b, a]; features += [feature, feature]
    if not sources:                                  # a single-atom molecule
        return MoleculeGraph(atoms, np.zeros((0, len(_bond_features_width())), np.float32),
                             np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64))
    n = len(sources)
    reverse = np.arange(n, dtype=np.int64) ^ 1       # edges were appended in pairs
    return MoleculeGraph(atoms, np.asarray(features, dtype=np.float32),
                         np.asarray(sources, dtype=np.int64),
                         np.asarray(targets, dtype=np.int64), reverse)


def _bond_features_width() -> list[float]:
    from rdkit import Chem
    mol = Chem.MolFromSmiles("CC")
    return _bond_features(mol.GetBondWithIdx(0))


def batch_graphs(smiles_list):
    """Disjoint union of the given molecules, with an atom -> molecule index."""
    graphs = [featurise(s) for s in smiles_list]
    atom_offset, edge_offset = 0, 0
    atoms, bonds, source, target, reverse, membership = [], [], [], [], [], []
    for index, graph in enumerate(graphs):
        atoms.append(graph.atom_features)
        membership.append(np.full(len(graph.atom_features), index, dtype=np.int64))
        if len(graph.edge_source):
            bonds.append(graph.bond_features)
            source.append(graph.edge_source + atom_offset)
            target.append(graph.edge_target + atom_offset)
            reverse.append(graph.reverse_edge + edge_offset)
            edge_offset += len(graph.edge_source)
        atom_offset += len(graph.atom_features)
    width_b = len(_bond_features_width())
    return {
        "atoms": np.concatenate(atoms).astype(np.float32),
        "bonds": (np.concatenate(bonds).astype(np.float32) if bonds
                  else np.zeros((0, width_b), np.float32)),
        "source": np.concatenate(source) if source else np.zeros(0, np.int64),
        "target": np.concatenate(target) if target else np.zeros(0, np.int64),
        "reverse": np.concatenate(reverse) if reverse else np.zeros(0, np.int64),
        "membership": np.concatenate(membership),
        "n_molecules": len(graphs),
    }


@dataclass
class DMPNN:
    """Graph encoder + condition head.  Variants differ only in what reaches the head.

    ``use_conditions`` and ``use_descriptors`` are the two ablation switches the
    pre-registration names (graph only / graph + conditions / graph + conditions +
    RDKit descriptors).  Everything else — depth, width, dropout, learning rate,
    the early-stopping rule — is fixed across variants so the comparison is about
    the information, not the tuning.
    """

    name: str = "T3_DMPNN"
    tier: str = "T3"
    use_conditions: bool = True
    use_descriptors: bool = False
    hidden: int = 200
    depth: int = 3
    ffn_hidden: int = 200
    dropout: float = 0.15
    weight_decay: float = 1e-5
    learning_rate: float = 8e-4
    max_epochs: int = 250
    patience: int = 30
    n_seeds: int = 3
    batch_size: int = 64
    selected_: dict = field(default_factory=dict)

    # -- torch module ------------------------------------------------------- #
    def _module(self, atom_width: int, bond_width: int, context_width: int, seed: int):
        import torch
        from torch import nn

        hidden, depth, dropout = self.hidden, self.depth, self.dropout
        ffn_hidden = self.ffn_hidden

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.w_i = nn.Linear(atom_width + bond_width, hidden, bias=False)
                self.w_h = nn.Linear(hidden, hidden, bias=False)
                self.w_o = nn.Linear(atom_width + hidden, hidden)
                self.dropout = nn.Dropout(dropout)
                self.head = nn.Sequential(
                    nn.Linear(hidden + context_width, ffn_hidden), nn.ReLU(),
                    nn.Dropout(dropout), nn.Linear(ffn_hidden, 1))

            def encode(self, graph):
                atoms, bonds = graph["atoms"], graph["bonds"]
                source, target, reverse = graph["source"], graph["target"], graph["reverse"]
                n_atoms = atoms.shape[0]
                if bonds.shape[0] == 0:
                    aggregated = torch.zeros(n_atoms, hidden, device=atoms.device)
                else:
                    h0 = torch.relu(self.w_i(torch.cat([atoms[source], bonds], dim=1)))
                    h = h0
                    for _ in range(depth - 1):
                        into = torch.zeros(n_atoms, hidden, device=atoms.device)
                        into.index_add_(0, target, h)
                        # sum over N(source) \ {target}: everything entering the source
                        # atom except the message that would travel straight back.
                        message = into[source] - h[reverse]
                        h = self.dropout(torch.relu(h0 + self.w_h(message)))
                    aggregated = torch.zeros(n_atoms, hidden, device=atoms.device)
                    aggregated.index_add_(0, target, h)
                atom_state = self.dropout(torch.relu(self.w_o(torch.cat([atoms, aggregated], 1))))
                molecules = torch.zeros(graph["n_molecules"], hidden, device=atoms.device)
                molecules.index_add_(0, graph["membership"], atom_state)
                counts = torch.zeros(graph["n_molecules"], 1, device=atoms.device)
                counts.index_add_(0, graph["membership"],
                                  torch.ones(n_atoms, 1, device=atoms.device))
                return molecules / counts.clamp(min=1.0)

            def forward(self, graph, molecule_index, context):
                embedding = self.encode(graph)[molecule_index]
                if context is not None:
                    embedding = torch.cat([embedding, context], dim=1)
                return self.head(embedding)

        torch.manual_seed(seed)
        return Encoder()

    # -- data --------------------------------------------------------------- #
    def _context(self, train, frames, blocks):
        """Standardised context vector, fitted on training rows only."""
        columns: list[str] = []
        if self.use_conditions:
            columns += list(blocks["COND"]) + list(blocks["MASSACT"])
        if self.use_descriptors:
            columns += list(blocks["LIG2D"]) + list(blocks["PHYSCHEM"]) + list(blocks["DONORS"])
        if not columns:
            return [None for _ in frames], 0
        pre = FoldPreprocessor(tuple(columns), standardise=True,
                               clip_to_train_range=True).fit(train)
        return [pre.transform(f) for f in frames], pre.transform(train).shape[1]

    def _graph(self, smiles_list, device):
        import torch
        batched = batch_graphs(list(smiles_list))
        return {k: (torch.tensor(v, device=device) if isinstance(v, np.ndarray) else v)
                for k, v in batched.items()}

    @staticmethod
    def _subgraph(cache, positions):
        """Batched graph over just the molecules a minibatch mentions.

        Message passing never crosses molecules — the batch is a disjoint union —
        so a molecule's embedding does not depend on which others are present.
        Encoding only the batch's molecules is therefore exactly equivalent to
        encoding all of them and indexing, and on this cohort it is several times
        cheaper: one extractant supplies 306 of the 1,329 rows, so a 64-row batch
        typically mentions far fewer than 64 distinct molecules.
        """
        import torch
        unique, inverse = np.unique(positions, return_inverse=True)
        atoms, bonds, source, target, reverse, membership = [], [], [], [], [], []
        atom_offset = edge_offset = 0
        for index, molecule in enumerate(unique):
            graph = cache[int(molecule)]
            atoms.append(graph.atom_features)
            membership.append(np.full(len(graph.atom_features), index, dtype=np.int64))
            if len(graph.edge_source):
                bonds.append(graph.bond_features)
                source.append(graph.edge_source + atom_offset)
                target.append(graph.edge_target + atom_offset)
                reverse.append(graph.reverse_edge + edge_offset)
                edge_offset += len(graph.edge_source)
            atom_offset += len(graph.atom_features)
        width_b = len(_bond_features_width())
        block = {
            "atoms": torch.tensor(np.concatenate(atoms).astype(np.float32)),
            "bonds": torch.tensor(np.concatenate(bonds).astype(np.float32) if bonds
                                  else np.zeros((0, width_b), np.float32)),
            "source": torch.tensor(np.concatenate(source) if source else np.zeros(0, np.int64)),
            "target": torch.tensor(np.concatenate(target) if target else np.zeros(0, np.int64)),
            "reverse": torch.tensor(np.concatenate(reverse) if reverse else np.zeros(0, np.int64)),
            "membership": torch.tensor(np.concatenate(membership)),
            "n_molecules": len(unique),
        }
        return block, torch.tensor(inverse.astype(np.int64))

    #: Pinned, not left to the machine.  Parallel reduction order in ``index_add_``
    #: is not associative, and this repository has already measured a 1e-15
    #: difference compounding into 0.16 log units through a second model's target.
    #: A fixed thread count makes the arm reproducible; it is recorded in the
    #: manifest so a rerun on another machine can match it.
    torch_threads: int = 2

    def _run(self, train, y_train, w_train, context_train, valid, y_valid, context_valid,
             test, context_test, seed, epochs):
        import torch
        torch.set_num_threads(int(self.torch_threads))
        device = "cpu"
        molecules = sorted(set(train["extractant"]) | set(test["extractant"])
                           | (set(valid["extractant"]) if valid is not None else set()))
        position = {m: i for i, m in enumerate(molecules)}
        cache = [featurise(m) for m in molecules]
        graph = self._graph(molecules, device)
        atom_width = graph["atoms"].shape[1]
        bond_width = graph["bonds"].shape[1] if graph["bonds"].shape[0] else len(_bond_features_width())
        context_width = context_train.shape[1] if context_train is not None else 0
        model = self._module(atom_width, bond_width, context_width, seed).to(device)
        optimiser = torch.optim.AdamW(model.parameters(), lr=self.learning_rate,
                                      weight_decay=self.weight_decay)

        def tensors(frame, context, y=None, w=None):
            index = torch.tensor([position[m] for m in frame["extractant"]], device=device)
            ctx = torch.tensor(context, dtype=torch.float32, device=device) if context is not None else None
            out = [index, ctx]
            out.append(torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(1)
                       if y is not None else None)
            out.append(torch.tensor(w, dtype=torch.float32, device=device).unsqueeze(1)
                       if w is not None else None)
            return out

        idx_tr, ctx_tr, y_tr, w_tr = tensors(train, context_train, y_train, w_train)
        if valid is not None:
            idx_va, ctx_va, y_va, _ = tensors(valid, context_valid, y_valid)
        idx_te, ctx_te, _, _ = tensors(test, context_test)

        generator = torch.Generator().manual_seed(seed)
        best_state, best_score, best_epoch, bad = None, np.inf, 0, 0
        limit = epochs or self.max_epochs
        for epoch in range(1, limit + 1):
            model.train()
            order = torch.randperm(len(idx_tr), generator=generator)
            for start in range(0, len(order), self.batch_size):
                batch = order[start:start + self.batch_size]
                optimiser.zero_grad()
                sub, local = self._subgraph(cache, idx_tr[batch].numpy())
                prediction = model(sub, local,
                                   ctx_tr[batch] if ctx_tr is not None else None)
                loss = (w_tr[batch] * (prediction - y_tr[batch]).abs()).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimiser.step()
            if valid is not None:
                model.eval()
                with torch.no_grad():
                    score = float((model(graph, idx_va, ctx_va) - y_va).abs().mean())
                if score < best_score - 1e-5:
                    best_score, best_epoch, bad = score, epoch, 0
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        if valid is not None and best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            prediction = model(graph, idx_te, ctx_te).squeeze(1).cpu().numpy()
        return prediction, max(best_epoch, 1), best_score

    def fit_predict(self, train, y_train, test, context):
        blocks = context.blocks
        inner_train, inner_valid = context.inner_train, context.inner_validation
        contexts, _ = self._context(inner_train, [inner_train, inner_valid], blocks)
        ctx_inner, ctx_valid = contexts
        w_inner = extractant_balanced_weights(inner_train["extractant"])
        epochs, scores = [], []
        for offset in range(self.n_seeds):
            _, best_epoch, best_score = self._run(
                inner_train, context.y_inner_train, w_inner, ctx_inner,
                inner_valid, context.y_inner_validation, ctx_valid,
                inner_valid, ctx_valid, context.model_seed + offset, None)
            epochs.append(best_epoch); scores.append(best_score)
        chosen = int(np.median(epochs))
        self.selected_ = {"epochs": chosen, "inner_val_mae": float(np.mean(scores))}

        outer_contexts, _ = self._context(train, [train, test], blocks)
        ctx_train, ctx_test = outer_contexts
        w_train = extractant_balanced_weights(train["extractant"])
        predictions = [self._run(train, y_train, w_train, ctx_train, None, None, None,
                                 test, ctx_test, context.model_seed + offset, chosen)[0]
                       for offset in range(self.n_seeds)]
        return clip_to_training_range(np.mean(predictions, axis=0), y_train)


def dmpnn_variants() -> list:
    return [
        DMPNN(name="T3_DMPNN_GRAPH_ONLY", use_conditions=False, use_descriptors=False),
        DMPNN(name="T3_DMPNN_COND", use_conditions=True, use_descriptors=False),
        DMPNN(name="T3_DMPNN_COND_DESC", use_conditions=True, use_descriptors=True),
    ]
