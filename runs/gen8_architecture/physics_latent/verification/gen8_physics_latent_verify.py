"""Adversarial re-run of gen8 physics-latent with the target destroyed.

Identical to gen8.evaluate.evaluate_fewshot except that the DataFrame handed to
adapter.predict and to the acquisition policies has its log_D column replaced.
Modes: clean (control) | nan | shuffled (finite but wrong).  All modes are
evaluated in one pass so each fold is fitted exactly once, and the adapter
design cache is keyed by mode so a cached value from one mode can never be
served to another (which would mask a leak).

Also carries OFFSET_PHYS: plain ridge on the SAME physics design, using no
training-row targets at all -- the clean steel-man for "the win is the design".
"""
from __future__ import annotations
import sys, time, argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/lilshadix/PycharmProjects/ml_separator")
sys.path.insert(0, str(ROOT / "src"))

from lanthanide_separation.gen8.adapters import AdaptContext, is_trainable, default_adapters
from lanthanide_separation.gen8.evaluate import _metrics, summarise
from lanthanide_separation.gen8.kshot import (POLICIES, PolicyContext, stable_hash,
                                              DEFAULT_RIDGE, ridge_fit)
from lanthanide_separation.gen8.protocols import (K_VALUES, POOL_CAP, _standardised_axes,
                                                  make_p2_split)
from lanthanide_separation.gen8.physics_latent import (build_physics_adapters, PhysicsAdapter,
                                                       build_design_spec, design as phys_design)

COHORT = ROOT / "runs/gen7_architecture/cache/cohort.parquet"
OOF = ROOT / "runs/gen7_architecture/finalists/oof_predictions.parquet"
MODEL = "REC_ecfp_plus_recovered"

MODE = {"value": "clean"}
FALLBACK = {"exception": {}, "nonfinite": {}, "no_state": {}, "calls": {}}


def bump(bucket, name):
    FALLBACK[bucket][name] = FALLBACK[bucket].get(name, 0) + 1


def instrument():
    """Replace PhysicsAdapter.predict with a mode-keyed, fallback-counting copy."""
    def counted(self, block, prediction, selected, observed, context):
        bump("calls", self.name)
        prediction = np.asarray(prediction, dtype=float)
        state = self.fitter.state
        if state is None:
            bump("no_state", self.name)
            return prediction
        try:
            key = (MODE["value"], context.split_seed, context.fold, context.extractant, len(block))
            entry = self._cache.get(key)
            if entry is None:
                entry = (phys_design(block, state.spec), self.fitter.theta_prior(block))
                self._cache[key] = entry
            X, prior = entry
            base = prediction if self.fitter.target == "residual" else np.zeros(len(block))
            if not self.use_observations or len(selected) == 0:
                theta = prior
            else:
                target = np.asarray(observed, dtype=float) - base[np.asarray(selected, int)]
                theta = self.fitter.map_update(X, prior, np.asarray(selected, int), target)
            out = base + X @ theta
        except Exception:
            bump("exception", self.name)
            return prediction
        if not np.isfinite(out).all():
            bump("nonfinite", self.name)
            return prediction
        return out
    PhysicsAdapter.predict = counted


class OffsetPhys:
    """Steel-man: ridge on the physics design, zero training-row targets used."""

    def __init__(self, penalty=DEFAULT_RIDGE, n_basis=4, level_penalty=0.0):
        self.penalty = float(penalty)
        self.n_basis = int(n_basis)
        self.level_penalty = float(level_penalty)
        self.name = (f"OFFSET_PHYS_p{penalty:g}" if level_penalty == 0
                     else f"OFFSET_PHYS_p{penalty:g}_L{level_penalty:g}")
        self.spec = None
        self._cache = {}

    def fit_fold(self, train, y_train, *, split_seed, fold, model_seed):
        # y_train is deliberately ignored: the design reads feature columns only.
        self._cache.clear()
        self.spec = build_design_spec(train, metal_basis="rbf", n_basis=self.n_basis)

    def predict(self, block, prediction, selected, observed, context):
        bump("calls", self.name)
        prediction = np.asarray(prediction, dtype=float)
        if self.spec is None or len(selected) == 0:
            return prediction
        try:
            key = (MODE["value"], context.split_seed, context.fold, context.extractant, len(block))
            X = self._cache.get(key)
            if X is None:
                X = phys_design(block, self.spec)
                self._cache[key] = X
            sel = np.asarray(selected, int)
            r = np.asarray(observed, dtype=float) - prediction[sel]
            if self.level_penalty == 0.0:
                beta = ridge_fit(X[sel], r, penalty=self.penalty)
            else:
                pen = np.full(X.shape[1], self.penalty)
                pen[0] = self.level_penalty
                Xs = X[sel]
                beta = np.linalg.solve(Xs.T @ Xs + np.diag(pen), Xs.T @ r)
            out = prediction + X @ beta
        except Exception:
            bump("exception", self.name)
            return prediction
        if not np.isfinite(out).all():
            bump("nonfinite", self.name)
            return prediction
        return out


def run(oof, cohort, adapters, *, modes, policies=("RANDOM", "CENTRAL"), k_values=K_VALUES,
        repeats=12, seed=20260820, min_rows=4, pool_cap=POOL_CAP, fold_trainer=None):
    feature_columns = [c for c in cohort.columns if c not in ("log_D",)]
    merged = oof.merge(cohort[feature_columns], on="row_id", how="left",
                       validate="many_to_one", suffixes=("", "__cohort"))
    assert len(merged) == len(oof)
    trainable = [a for a in adapters if is_trainable(a)]
    records = {m: [] for m in modes}
    for (split_seed, fold), fold_block in merged.groupby(["split_seed", "fold"], sort=True):
        started = time.time()
        train, y_train, model_seed = fold_trainer(int(split_seed), int(fold))
        for adapter in trainable:
            adapter.fit_fold(train, y_train, split_seed=int(split_seed), fold=int(fold),
                             model_seed=int(model_seed))
        for ligand, block in fold_block.groupby("extractant", sort=True):
            block = block.reset_index(drop=True)
            n = len(block)
            if n < min_rows:
                continue
            truth = block["log_D"].to_numpy(dtype=float)
            prediction = block["prediction"].to_numpy(dtype=float)
            axes = _standardised_axes(block)
            uncertainty = (block["extra__prediction_sd"].to_numpy(dtype=float)
                           if "extra__prediction_sd" in block.columns else np.full(n, np.nan))
            disagreement = (block["disagreement"].to_numpy(dtype=float)
                            if "disagreement" in block.columns else np.full(n, np.nan))
            context = AdaptContext(split_seed=int(split_seed), fold=int(fold), extractant=ligand)
            for mode in modes:
                MODE["value"] = mode
                if mode == "clean":
                    shown = block
                else:
                    shown = block.copy()
                    shown["log_D"] = (np.nan if mode == "nan"
                                      else np.random.default_rng(12345).permutation(truth) + 7.0)
                for repeat in range(repeats):
                    rng = np.random.default_rng((seed, repeat, stable_hash(ligand)))
                    split = make_p2_split(n, rng, pool_cap=pool_cap)
                    if len(split.pool) < 1 or len(split.evaluation) < 2:
                        continue
                    base = {"split_seed": int(split_seed), "fold": int(fold),
                            "extractant": ligand, "repeat": repeat, "n_rows": n,
                            "n_pool": int(len(split.pool)), "n_eval": int(len(split.evaluation))}
                    records[mode].append(
                        {**base, "policy": "NONE", "adapter": "ZERO_SHOT_REF", "k": 0,
                         **_metrics(truth[split.evaluation], prediction[split.evaluation])})
                    sequences = {}
                    for name in policies:
                        pc = PolicyContext(block=shown, prediction=prediction, pool=split.pool,
                                           evaluation=split.evaluation, axes=axes,
                                           uncertainty=uncertainty, disagreement=disagreement,
                                           rng=np.random.default_rng((seed, repeat,
                                                                      stable_hash(name))),
                                           truth=None)
                        chosen = []
                        for _ in range(min(max(k_values), len(split.pool))):
                            chosen.append(int(POLICIES[name](pc, chosen)))
                        sequences[name] = chosen
                    for policy_name, sequence in sequences.items():
                        for k in k_values:
                            if k < 1 or k > len(sequence):
                                continue
                            sel = np.asarray(sequence[:k], dtype=int)
                            for adapter in adapters:
                                values = np.asarray(
                                    adapter.predict(shown, prediction, sel, truth[sel], context),
                                    dtype=float)
                                if not np.isfinite(values[split.evaluation]).all():
                                    continue
                                records[mode].append(
                                    {**base, "policy": policy_name, "adapter": adapter.name,
                                     "k": k, **_metrics(truth[split.evaluation],
                                                        values[split.evaluation])})
        print(f"  seed {split_seed} fold {fold}: {time.time()-started:.1f}s", flush=True)
    return {m: pd.DataFrame(v) for m, v in records.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="*", default=["clean", "nan", "shuffled"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[104729])
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from lanthanide_separation.gen6.cohorts import seeded_group_kfold
    cohort = pd.read_parquet(COHORT)
    oof = pd.read_parquet(OOF)
    oof = oof[(oof["model"] == MODEL) & (oof["split_seed"].isin(args.seeds))].reset_index(drop=True)
    groups = cohort["tanimoto_cluster"].astype(str).to_numpy()
    rid = cohort["row_id"].to_numpy()
    plan = {}
    for s in args.seeds:
        marked = oof[oof["split_seed"] == s]
        for f, (tr, te) in enumerate(seeded_group_kfold(groups, 5, int(s))):
            assert set(rid[te]) == set(marked.loc[marked["fold"] == f, "row_id"])
            plan[(int(s), int(f))] = tr
    frozen = {int(s): oof.loc[oof["split_seed"] == s].set_index("row_id")["prediction"]
              for s in args.seeds}

    def fold_trainer(split_seed, fold):
        idx = plan[(int(split_seed), int(fold))]
        train = cohort.iloc[idx].copy()
        train["prediction"] = train["row_id"].map(frozen[int(split_seed)]).to_numpy(dtype=float)
        assert np.isfinite(train["prediction"]).all()
        return train, train["log_D"].to_numpy(dtype=float), 42 + int(fold) * 1009 + 9999991

    instrument()
    adapters = (build_physics_adapters()
                + [OffsetPhys(penalty=p) for p in (0.25, 0.5, 1.0, 4.0)]
                + [OffsetPhys(penalty=0.5, level_penalty=lp) for lp in (0.1, 0.25, 0.5, 1.0)]
                + default_adapters())
    print("adapters:", [a.name for a in adapters], flush=True)
    out = run(oof, cohort, adapters, modes=args.modes, repeats=args.repeats,
              fold_trainer=fold_trainer)
    for m, detail in out.items():
        detail.to_parquet(f"{args.out}_{m}.parquet", index=False)
        print(f"### {m}: {len(detail):,} records", flush=True)
    print("FALLBACKS", json.dumps(FALLBACK, indent=1), flush=True)


if __name__ == "__main__":
    main()
