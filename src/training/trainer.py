"""Augmented-Lagrangian trainer for the joint causal DAG-SSM (spec 4, 9.5).

NOTEARS-style optimisation. For each of ``outer_steps`` outer iterations we run
``inner_steps`` of Adam on

    L = -L_obs + lambda_int * L_int + alpha*h + rho/2 * h^2

then update the Lagrangian variables from the achieved acyclicity ``h``:

  * if ``h`` has not shrunk to ``progress_rate`` of its previous value, multiply ``rho`` by
    ``rho_mult`` (push harder on the constraint);
  * always take a dual ascent step ``alpha += rho * h``.

Observational (fMRI windows) and interventional (region TEPs) batches are drawn together
each inner step so both data terms shape ``W`` jointly. The interventional set is tiny
(~109 records) so we cycle it; the observational loader drives the step count.
"""
from __future__ import annotations

import itertools
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .losses import acyclicity_penalty, intervention_loss, obs_nll


@dataclass
class TrainState:
    rho: float
    alpha: float
    h: float = float("inf")
    history: list = field(default_factory=list)


class Trainer:
    def __init__(self, model, cfg, device: str | None = None):
        self.model = model
        self.cfg = cfg
        self.device = device or cfg.train.device
        self.model.to(self.device)
        self.opt = torch.optim.Adam(
            self.model.parameters(), lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay)
        self.state = TrainState(rho=cfg.dag.rho_init, alpha=cfg.dag.alpha_init)

    # ------------------------------------------------------------------ one inner step
    def _inner_step(self, obs_batch, itv_batch) -> dict:
        self.opt.zero_grad()
        y = obs_batch.to(self.device)
        l_obs = obs_nll(self.model, y)

        stim = itv_batch["stim_parcel"].to(self.device)
        tep = itv_batch["region_tep"].to(self.device)
        l_int = intervention_loss(self.model, stim, tep)

        h = self.model.acyclicity()
        pen = acyclicity_penalty(h, self.state.alpha, self.state.rho)

        loss = l_obs + self.cfg.train.lambda_int * l_int + pen
        loss.backward()
        self.opt.step()
        return {"loss": loss.item(), "l_obs": l_obs.item(), "l_int": l_int.item(),
                "h": h.item()}

    # ------------------------------------------------------------------ checkpoint freeze
    def _save_train_ckpt(self, path: Path, outer_done: int, inner: int, outer: int) -> None:
        """Atomically freeze full training state after an outer iteration.

        Captures model + optimizer + Lagrangian state + how many outer iters are DONE, so a
        restart can resume mid-fold instead of redoing the whole fold. Atomic (tmp + os.replace)
        so a power-off mid-write cannot corrupt the file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "rho": self.state.rho, "alpha": self.state.alpha, "h": self.state.h,
            "history": self.state.history,
            "outer_done": outer_done, "inner": inner, "outer": outer,
            "lambda_int": float(self.cfg.train.lambda_int),
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)            # atomic on POSIX

    def _load_train_ckpt(self, path: Path, inner: int, outer: int, verbose: bool) -> int:
        """Restore a mid-fold checkpoint; return the next outer index to run (0 if unusable)."""
        try:
            ck = torch.load(path, map_location=self.device, weights_only=False)
        except (OSError, RuntimeError, EOFError) as e:
            if verbose:
                print(f"  [resume] checkpoint {path} unreadable ({e}); starting fresh")
            return 0
        # only resume if it matches this run's schedule (else the ckpt is from a different config)
        if ck.get("inner") != inner or ck.get("outer") != outer:
            if verbose:
                print(f"  [resume] checkpoint schedule mismatch "
                      f"(ckpt inner/outer={ck.get('inner')}/{ck.get('outer')} "
                      f"vs {inner}/{outer}); starting fresh")
            return 0
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["opt"])
        self.state.rho = ck["rho"]; self.state.alpha = ck["alpha"]; self.state.h = ck["h"]
        self.state.history = ck.get("history", [])
        start = int(ck["outer_done"]) + 1
        if verbose:
            print(f"  [resume] restored mid-fold checkpoint: {start}/{outer} outer iters done, "
                  f"h={self.state.h:.2e} rho={self.state.rho:.1e}")
        return start

    # ------------------------------------------------------------------ outer loop
    def fit(self, obs_loader: DataLoader, itv_loader: DataLoader,
            inner_steps: int | None = None, outer_steps: int | None = None,
            verbose: bool = True, ckpt_path=None, resume: bool = False):
        """Run the augmented-Lagrangian outer loop.

        ``ckpt_path``: if given, a mid-fold checkpoint is frozen after every outer iteration.
        ``resume``: if True and ``ckpt_path`` exists, training continues from that checkpoint
        (so a crash/power-off only loses the in-progress outer iter, not the whole fold).
        """
        inner = inner_steps or self.cfg.train.inner_steps
        outer = outer_steps or self.cfg.train.outer_steps
        log_every = self.cfg.train.log_every
        d_cfg = self.cfg.dag

        start_outer = 0
        if ckpt_path is not None and resume and Path(ckpt_path).exists():
            start_outer = self._load_train_ckpt(Path(ckpt_path), inner, outer, verbose)

        itv_cycle = itertools.cycle(itv_loader)
        for outer_i in range(start_outer, outer):
            obs_cycle = itertools.cycle(obs_loader)
            last = {}
            for step in range(inner):
                last = self._inner_step(next(obs_cycle), next(itv_cycle))
                if verbose and step % log_every == 0:
                    print(f"  [outer {outer_i:02d} | step {step:04d}] "
                          f"loss={last['loss']:.3f} L_obs={last['l_obs']:.3f} "
                          f"L_int={last['l_int']:.3f} h={last['h']:.3e}")

            # --- augmented-Lagrangian variable update ---
            h_new = last["h"]
            if h_new > d_cfg.progress_rate * self.state.h:
                self.state.rho = min(self.state.rho * d_cfg.rho_mult, d_cfg.rho_max)
            self.state.alpha = self.state.alpha + self.state.rho * h_new
            self.state.h = h_new
            self.state.history.append(
                {"outer": outer_i, **last, "rho": self.state.rho,
                 "alpha": self.state.alpha})
            if verbose:
                print(f"[outer {outer_i:02d}] h={h_new:.3e} -> rho={self.state.rho:.1e} "
                      f"alpha={self.state.alpha:.3e}")
            # freeze AFTER the outer update so a resume continues from a consistent state
            if ckpt_path is not None:
                self._save_train_ckpt(Path(ckpt_path), outer_i, inner, outer)
            if h_new < d_cfg.h_tol:
                if verbose:
                    print(f"acyclicity converged (h={h_new:.2e} < {d_cfg.h_tol:.0e})")
                break
        return self.state


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config import load_config
    from data.dataset import (ObservationalDataset, InterventionalDataset,
                              collate_observational, collate_interventional)
    from model.causal_dag_ssm import CausalDAGSSM

    cfg = load_config()
    proc = cfg.paths.processed_dir
    d = cfg.parcellation.d

    obs_ds = ObservationalDataset(proc, window=cfg.train.batch_size * 0 + 60)
    itv_ds = InterventionalDataset(proc)
    obs_loader = DataLoader(obs_ds, batch_size=8, shuffle=True,
                            collate_fn=collate_observational)
    itv_loader = DataLoader(itv_ds, batch_size=8, shuffle=True,
                            collate_fn=collate_interventional)
    print(f"obs windows={len(obs_ds)} itv records={len(itv_ds)} d={d}")

    model = CausalDAGSSM(d, variant=cfg.dag.variant, input_dim=cfg.model.input_dim,
                         init_scale=cfg.model.init_state_scale)
    trainer = Trainer(model, cfg)
    # Smoke run: a couple of tiny outer iterations to confirm the loop drives the losses.
    state = trainer.fit(obs_loader, itv_loader, inner_steps=20, outer_steps=2)
    print("final h:", state.h, "rho:", state.rho)
    n_edges = int((model.extract_dag(cfg.dag.threshold).abs() > 0).sum())
    print("edges after smoke train:", n_edges)
