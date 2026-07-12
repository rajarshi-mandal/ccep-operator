"""Tests for Trainer mid-fold checkpoint freeze + resume (crash/power-off recovery)."""
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import load_config  # noqa: E402
from model.causal_dag_ssm import CausalDAGSSM  # noqa: E402
from training.trainer import Trainer  # noqa: E402


def _tiny_loaders(d=5):
    obs = torch.randn(6, 12, d)
    itv = [{"stim_parcel": i % d, "region_tep": torch.randn(d, 8)} for i in range(4)]

    def collate_obs(b): return torch.stack(b)

    def collate_itv(b):
        return {"stim_parcel": torch.tensor([x["stim_parcel"] for x in b]),
                "region_tep": torch.stack([x["region_tep"] for x in b])}
    ol = DataLoader(list(obs), batch_size=2, collate_fn=collate_obs)
    il = DataLoader(itv, batch_size=2, collate_fn=collate_itv)
    return ol, il


def _model(cfg, d=5):
    torch.manual_seed(0)
    return CausalDAGSSM(d, variant=cfg.dag.variant, input_dim=cfg.model.input_dim,
                        init_scale=cfg.model.init_state_scale)


def test_freeze_creates_atomic_checkpoint(tmp_path):
    cfg = load_config(); cfg["train"]["lambda_int"] = 5.0
    ol, il = _tiny_loaders()
    ckpt = tmp_path / "fold.train.pt"
    Trainer(_model(cfg), cfg).fit(ol, il, inner_steps=2, outer_steps=3,
                                  verbose=False, ckpt_path=ckpt, resume=False)
    assert ckpt.exists()
    ck = torch.load(ckpt, weights_only=False)
    assert {"model", "opt", "rho", "alpha", "h", "outer_done", "inner", "outer"} <= set(ck)
    assert ck["inner"] == 2 and ck["outer"] == 3
    assert not (tmp_path / "fold.train.pt.tmp").exists()  # tmp cleaned by atomic replace


def test_resume_continues_from_checkpoint(tmp_path):
    cfg = load_config(); cfg["train"]["lambda_int"] = 5.0
    ol, il = _tiny_loaders()
    ckpt = tmp_path / "fold.train.pt"
    t1 = Trainer(_model(cfg), cfg)
    t1.fit(ol, il, inner_steps=2, outer_steps=3, verbose=False, ckpt_path=ckpt, resume=False)
    done = torch.load(ckpt, weights_only=False)["outer_done"]

    # a fresh trainer resuming the SAME schedule must restore and skip completed outers
    t2 = Trainer(_model(cfg), cfg)
    start = t2._load_train_ckpt(ckpt, inner=2, outer=3, verbose=False)
    assert start == done + 1
    # restored weights match the frozen model exactly
    for (k, a), (_, b) in zip(t1.model.state_dict().items(), t2.model.state_dict().items()):
        assert torch.allclose(a, b), f"param {k} not restored"
    assert t2.state.rho == t1.state.rho and t2.state.h == t1.state.h


def test_schedule_mismatch_starts_fresh(tmp_path):
    cfg = load_config(); cfg["train"]["lambda_int"] = 5.0
    ol, il = _tiny_loaders()
    ckpt = tmp_path / "fold.train.pt"
    Trainer(_model(cfg), cfg).fit(ol, il, inner_steps=2, outer_steps=3,
                                  verbose=False, ckpt_path=ckpt, resume=False)
    # a different outer schedule must NOT resume (config changed -> ckpt is stale)
    start = Trainer(_model(cfg), cfg)._load_train_ckpt(ckpt, inner=2, outer=9, verbose=False)
    assert start == 0


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
