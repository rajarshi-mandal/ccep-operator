"""Linear Gaussian state space model backbone + Kalman filter (spec 2.2).

    h(t+1) = A h(t) + B u(t) + eps,   eps ~ N(0, Q)     [transition]
    y(t)   = C h(t) + eta,            eta ~ N(0, R)     [observation]

Each of the d latent dimensions is a brain region. ``A`` is supplied externally (it is
the DAG-constrained transition matrix from ``dag_constraint``), so this module owns only
B, C and the noise covariances and implements:

  * a differentiable Kalman filter giving the exact data log-likelihood (the L_obs term),
  * a forward rollout used by the interventional do-operation.

Edge convention: ``A[i, j]`` = "j influences i", so ``h_next = A @ h`` is the natural
matrix-vector product (row i sums contributions from all influencers j). Noise
covariances are parameterised by log-diagonals to stay positive-definite.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LinearGaussianSSM(nn.Module):
    def __init__(self, d: int, input_dim: int = 1, obs_dim: int | None = None,
                 learn_C: bool = False):
        super().__init__()
        self.d = d
        self.input_dim = input_dim
        self.obs_dim = obs_dim if obs_dim is not None else d

        self.B = nn.Parameter(torch.zeros(d, input_dim))
        # Default observation is identity (latent region == observed region, as for the
        # parcellated fMRI). Optionally learn a full C for richer observation models.
        if learn_C or self.obs_dim != d:
            self.C = nn.Parameter(0.1 * torch.randn(self.obs_dim, d))
            self._identity_C = False
        else:
            self.register_buffer("C", torch.eye(d))
            self._identity_C = True

        self.log_q = nn.Parameter(torch.zeros(d) - 1.0)            # process noise diag
        self.log_r = nn.Parameter(torch.zeros(self.obs_dim) - 1.0)  # obs noise diag
        self.h0 = nn.Parameter(torch.zeros(d))
        self.log_p0 = nn.Parameter(torch.zeros(d))                  # init covariance diag

    # ------------------------------------------------------------------ covariances
    @property
    def Q(self):
        return torch.diag(self.log_q.exp())

    @property
    def R(self):
        return torch.diag(self.log_r.exp())

    # ------------------------------------------------------------------ likelihood
    def kalman_log_likelihood(self, y: torch.Tensor, A: torch.Tensor,
                              u: torch.Tensor | None = None) -> torch.Tensor:
        """Exact data log-likelihood under the linear Gaussian SSM.

        ``y``: ``[B, T, obs_dim]`` observations. ``A``: ``[d, d]`` transition matrix.
        ``u``: optional ``[B, T, input_dim]`` inputs. Returns ``[B]`` log-likelihoods.
        """
        Bsz, T, _ = y.shape
        dev, dt = y.device, y.dtype
        C = self.C.to(dt)
        Q, R = self.Q.to(dt), self.R.to(dt)
        I = torch.eye(self.d, device=dev, dtype=dt)

        m = self.h0.to(dt).expand(Bsz, self.d).clone()             # [B, d]
        P = torch.diag(self.log_p0.exp()).to(dt).expand(Bsz, self.d, self.d).clone()
        ll = torch.zeros(Bsz, device=dev, dtype=dt)

        for t in range(T):
            # --- predict ---
            if t > 0:
                m = m @ A.T                                        # [B, d]
                if u is not None:
                    m = m + u[:, t - 1] @ self.B.T.to(dt)
                P = A @ P @ A.T + Q
            # --- update with y_t ---
            yt = y[:, t]                                           # [B, obs]
            yhat = m @ C.T                                         # [B, obs]
            innov = yt - yhat
            S = C @ P @ C.T + R                                    # [B, obs, obs]
            L = torch.linalg.cholesky(S)
            alpha = torch.cholesky_solve(innov.unsqueeze(-1), L).squeeze(-1)
            # log N(innov; 0, S)
            logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
            quad = (innov * alpha).sum(-1)
            ll = ll - 0.5 * (quad + logdet + self.obs_dim * torch.log(
                torch.tensor(2 * torch.pi, device=dev, dtype=dt)))
            # Kalman gain & posterior
            PCt = P @ C.T                                          # [B, d, obs]
            K = torch.cholesky_solve(PCt.transpose(-1, -2), L).transpose(-1, -2)
            m = m + (K @ innov.unsqueeze(-1)).squeeze(-1)
            P = (I - K @ C) @ P
        return ll

    # ------------------------------------------------------------------ rollout
    def rollout(self, h0: torch.Tensor, steps: int, A: torch.Tensor,
                u: torch.Tensor | None = None) -> torch.Tensor:
        """Deterministic latent rollout (noise-free mean dynamics).

        ``h0``: ``[B, d]``. Returns latent states ``[B, steps, d]``. Used by the
        intervention module to propagate the do-operation forward.
        """
        h = h0
        out = []
        for t in range(steps):
            if u is not None:
                h = h @ A.T + u[:, t] @ self.B.T
            else:
                h = h @ A.T
            out.append(h)
        return torch.stack(out, dim=1)

    def observe(self, h: torch.Tensor) -> torch.Tensor:
        """Map latent states ``[..., d]`` to observations ``[..., obs_dim]``."""
        return h @ self.C.T


if __name__ == "__main__":
    torch.manual_seed(0)
    d, q_std, r_std = 5, 0.1, 0.1
    ssm = LinearGaussianSSM(d).double()
    # set the model noise to match the simulation so the LL is meaningful
    with torch.no_grad():
        ssm.log_q.fill_(torch.log(torch.tensor(q_std ** 2)))
        ssm.log_r.fill_(torch.log(torch.tensor(r_std ** 2)))
    # ground-truth acyclic A
    A = torch.triu(0.3 * torch.randn(d, d, dtype=torch.float64), diagonal=1)
    # simulate data from the model
    T, B = 50, 8
    h = torch.zeros(B, d, dtype=torch.float64)
    ys = []
    for t in range(T):
        h = h @ A.T + q_std * torch.randn(B, d, dtype=torch.float64)
        ys.append(h + r_std * torch.randn(B, d, dtype=torch.float64))
    y = torch.stack(ys, dim=1)
    ll_true = ssm.kalman_log_likelihood(y, A)
    ll_wrong = ssm.kalman_log_likelihood(y, torch.zeros_like(A))
    print("mean LL with true A :", ll_true.mean().item())
    print("mean LL with zero A :", ll_wrong.mean().item())
    print("true-A LL higher    :", (ll_true.mean() > ll_wrong.mean()).item())
