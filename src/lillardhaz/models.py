"""Simultaneous-equations hazard/probit models with a Gaussian-copula
correlation, after Lillard (1993).

Direct port of the companion R/Stata packages' math: equation 1 may be a
probit, a log-normal accelerated-failure-time hazard, or a piecewise
(linear-log-hazard) Gompertz hazard with an arbitrary number of segments;
equation 2 is always a hazard, log-normal or piecewise Gompertz. See
docs/manual.html for the full likelihood derivation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, multivariate_normal

_BAD_PENALTY = 1e10


def _pgompertz_sf(theta, slopes, nodes, t):
    """Piecewise-Gompertz survival/density/hazard, general K segments.

    theta: per-observation location index, shape (n,)
    slopes: per-segment slopes, shape (K,)
    nodes: interior nodes, ascending, shape (K-1,)
    t: per-observation duration, shape (n,)

    Direct vectorized port of the recursion in the R/Stata companion
    packages' pgompertz helper.
    """
    theta = np.asarray(theta, dtype=float)
    slopes = np.asarray(slopes, dtype=float)
    nodes = np.asarray(nodes, dtype=float)
    t = np.asarray(t, dtype=float)
    K = slopes.shape[0]
    n = theta.shape[0]

    levels = np.zeros(K)
    startnodes = np.zeros(K)
    cumlevel = 0.0
    prevnode = 0.0
    for k in range(K):
        levels[k] = cumlevel
        startnodes[k] = prevnode
        if k < K - 1:
            thisnode = nodes[k]
            seglen = thisnode - prevnode
            cumlevel = cumlevel + slopes[k] * seglen
            prevnode = thisnode

    m = np.full(n, K - 1, dtype=int)
    for k in range(K - 1):
        thisnode = nodes[k]
        m[(t < thisnode) & (m == K - 1)] = k

    H = np.zeros(n)
    h = np.zeros(n)
    for k in range(K):
        lev_k = levels[k]
        sl_k = slopes[k]
        sn_k = startnodes[k]

        if k < K - 1:
            thisnode = nodes[k]
            seglen = thisnode - sn_k
            idx_full = m > k
            if np.any(idx_full):
                contrib = (np.exp(lev_k) * seglen if abs(sl_k) < 1e-8
                           else np.exp(lev_k) / sl_k * (np.exp(sl_k * seglen) - 1))
                H[idx_full] = H[idx_full] + np.exp(theta[idx_full]) * contrib

        idx_this = m == k
        if np.any(idx_this):
            tt = t[idx_this] - sn_k
            contrib2 = (np.exp(lev_k) * tt if abs(sl_k) < 1e-8
                        else np.exp(lev_k) / sl_k * (np.exp(sl_k * tt) - 1))
            H[idx_this] = H[idx_this] + np.exp(theta[idx_this]) * contrib2
            h[idx_this] = np.exp(theta[idx_this] + lev_k + sl_k * tt)

    S = np.exp(-H)
    f = h * S
    return S, f, h


def _hazard_marginal(kind, theta, t, lnsigma=None, slopes=None, nodes=None):
    if kind == "lognormal":
        sigma = np.exp(lnsigma)
        z = (np.log(t) - theta) / sigma
        S = 1 - norm.cdf(z)
        f = norm.pdf(z) / (sigma * t)
    else:
        S, f, _ = _pgompertz_sf(theta, slopes, nodes, t)
        z = norm.ppf(np.clip(1 - S, 1e-12, 1 - 1e-12))
    return S, f, z


def _pbivnorm(x, y, rho):
    """Vectorized bivariate normal CDF at fixed correlation rho."""
    mv = multivariate_normal(mean=[0.0, 0.0], cov=[[1.0, rho], [rho, 1.0]])
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    return mv.cdf(np.column_stack([x, y]))


def _numerical_hessian(f, x, eps=1e-4):
    """Central-difference Hessian of scalar-valued f at x (numpy-only,
    avoids pulling in an extra dependency just for this)."""
    n = len(x)
    H = np.zeros((n, n))
    f0 = f(x)
    for i in range(n):
        for j in range(i, n):
            xpp = x.copy(); xpp[i] += eps; xpp[j] += eps
            xpm = x.copy(); xpm[i] += eps; xpm[j] -= eps
            xmp = x.copy(); xmp[i] -= eps; xmp[j] += eps
            xmm = x.copy(); xmm[i] -= eps; xmm[j] -= eps
            val = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * eps * eps)
            H[i, j] = H[j, i] = val
    return H


def _param_layout(eq1type, eq2type, x1names, x2names, K1, K2, corr):
    names = [f"eq1:_cons"] + [f"eq1:{v}" for v in x1names]
    if eq1type == "lognormal":
        names += ["ln_sigma1"]
    elif eq1type == "pgompertz":
        names += [f"s1_{k+1}" for k in range(K1)]

    names += ["eq2:_cons"] + [f"eq2:{v}" for v in x2names]
    if eq2type == "lognormal":
        names += ["ln_sigma2"]
    else:
        names += [f"s2_{k+1}" for k in range(K2)]

    if corr:
        names += ["atanh_rho"]
    return names


def _unpack(par, eq1type, eq2type, p1, p2, nodes1, nodes2, corr):
    idx = 0
    b1 = par[idx:idx + p1]; idx += p1
    lnsigma1 = slopes1 = None
    if eq1type == "lognormal":
        lnsigma1 = par[idx]; idx += 1
    elif eq1type == "pgompertz":
        K1 = len(nodes1) + 1
        slopes1 = par[idx:idx + K1]; idx += K1

    b2 = par[idx:idx + p2]; idx += p2
    lnsigma2 = slopes2 = None
    if eq2type == "lognormal":
        lnsigma2 = par[idx]; idx += 1
    else:
        K2 = len(nodes2) + 1
        slopes2 = par[idx:idx + K2]; idx += K2

    rho = np.tanh(par[idx]) if corr else 0.0
    return b1, lnsigma1, slopes1, b2, lnsigma2, slopes2, rho


def _neg_loglik(par, eq1type, eq2type, X1, X2, y1, d1, y2, d2, nodes1, nodes2, corr):
    try:
        p1, p2 = X1.shape[1], X2.shape[1]
        b1, lnsigma1, slopes1, b2, lnsigma2, slopes2, rho = _unpack(
            par, eq1type, eq2type, p1, p2, nodes1, nodes2, corr
        )
        theta1 = X1 @ b1
        theta2 = X2 @ b2
        a = np.sqrt(1 - rho ** 2)
        n = len(y2)
        cont = np.zeros(n)

        with np.errstate(all="ignore"):
            if eq1type == "probit":
                S2, f2, z2 = _hazard_marginal(eq2type, theta2, y2, lnsigma2, slopes2, nodes2)

                i = (d1 == 1) & (d2 == 1)
                cont[i] = f2[i] * norm.cdf((theta1[i] + rho * z2[i]) / a)
                i = (d1 == 1) & (d2 == 0)
                if np.any(i):
                    cont[i] = norm.cdf(theta1[i]) - _pbivnorm(theta1[i], z2[i], -rho)
                i = (d1 == 0) & (d2 == 1)
                cont[i] = f2[i] * norm.cdf(-(theta1[i] + rho * z2[i]) / a)
                i = (d1 == 0) & (d2 == 0)
                if np.any(i):
                    cont[i] = _pbivnorm(-theta1[i], -z2[i], -rho)
            else:
                S1, f1, z1 = _hazard_marginal(eq1type, theta1, y1, lnsigma1, slopes1, nodes1)
                S2, f2, z2 = _hazard_marginal(eq2type, theta2, y2, lnsigma2, slopes2, nodes2)

                i = (d1 == 1) & (d2 == 1)
                cont[i] = (np.exp(-(z1[i] ** 2 - 2 * rho * z1[i] * z2[i] + z2[i] ** 2) / (2 * a ** 2))
                           / (2 * np.pi * a) * f1[i] / norm.pdf(z1[i]) * f2[i] / norm.pdf(z2[i]))
                i = (d1 == 1) & (d2 == 0)
                cont[i] = f1[i] * (1 - norm.cdf((z2[i] - rho * z1[i]) / a))
                i = (d1 == 0) & (d2 == 1)
                cont[i] = f2[i] * (1 - norm.cdf((z1[i] - rho * z2[i]) / a))
                i = (d1 == 0) & (d2 == 0)
                if np.any(i):
                    cont[i] = _pbivnorm(-z1[i], -z2[i], rho)

            cont = np.maximum(cont, 1e-300)
            total = np.sum(np.log(cont))

        if not np.isfinite(total):
            return _BAD_PENALTY
        return -total
    except Exception:
        return _BAD_PENALTY


@dataclass
class LillardhazResult:
    params: dict = field(default_factory=dict)
    se: dict = field(default_factory=dict)
    loglik: float = float("nan")
    eq1type: str = ""
    eq2type: str = ""
    nodes1: Optional[Sequence[float]] = None
    nodes2: Optional[Sequence[float]] = None
    corr: bool = True
    rho: Optional[float] = None
    rho_se: Optional[float] = None
    x1: Sequence[str] = field(default_factory=list)
    x2: Sequence[str] = field(default_factory=list)
    y1name: str = ""
    y2name: str = ""
    n: int = 0
    converged: bool = False

    def summary(self) -> str:
        lines = [
            f"Lillard-style simultaneous-equations model",
            f"eq1: {self.eq1type}   eq2: {self.eq2type}   corr: {self.corr}",
            f"Observations: {self.n}   Log-likelihood: {self.loglik:.3f}   "
            f"converged: {self.converged}",
            "",
            f"{'parameter':<14}{'estimate':>12}{'std.err':>12}",
        ]
        for name, val in self.params.items():
            se = self.se.get(name, float("nan"))
            lines.append(f"{name:<14}{val:>12.4f}{se:>12.4f}")
        if self.corr:
            lines.append(f"{'rho':<14}{self.rho:>12.4f}{self.rho_se:>12.4f}")
        return "\n".join(lines)

    def __repr__(self):
        return self.summary()

    def predict(self, data: dict, kind: str = "surv2"):
        """Predicted survival/density/index for new data.

        data: dict/DataFrame-like mapping column name -> array.
        kind: one of "surv2" (default), "surv1", "dens1", "dens2",
              "pr1", "xb1", "xb2".
        """
        n = len(next(iter(data.values())))
        X1 = np.column_stack([np.ones(n)] + [np.asarray(data[v], dtype=float) for v in self.x1])
        X2 = np.column_stack([np.ones(n)] + [np.asarray(data[v], dtype=float) for v in self.x2])

        b1 = np.array([self.params[f"eq1:_cons"]] + [self.params[f"eq1:{v}"] for v in self.x1])
        b2 = np.array([self.params[f"eq2:_cons"]] + [self.params[f"eq2:{v}"] for v in self.x2])
        theta1 = X1 @ b1
        theta2 = X2 @ b2

        if kind == "xb1":
            return theta1
        if kind == "xb2":
            return theta2
        if kind == "pr1":
            if self.eq1type != "probit":
                raise ValueError("pr1 is only available when eq1='probit'")
            return norm.cdf(theta1)

        if kind in ("surv1", "dens1"):
            if self.eq1type == "probit":
                raise ValueError("surv1/dens1 are not available when eq1='probit'")
            t1 = np.asarray(data[self.y1name], dtype=float)
            if self.eq1type == "lognormal":
                S1, f1, _ = _hazard_marginal("lognormal", theta1, t1, lnsigma=self.params["ln_sigma1"])
            else:
                K1 = len(self.nodes1) + 1
                slopes1 = np.array([self.params[f"s1_{k+1}"] for k in range(K1)])
                S1, f1, _ = _hazard_marginal("pgompertz", theta1, t1, slopes=slopes1, nodes=self.nodes1)
            return S1 if kind == "surv1" else f1

        t2 = np.asarray(data[self.y2name], dtype=float)
        if self.eq2type == "lognormal":
            S2, f2, _ = _hazard_marginal("lognormal", theta2, t2, lnsigma=self.params["ln_sigma2"])
        else:
            K2 = len(self.nodes2) + 1
            slopes2 = np.array([self.params[f"s2_{k+1}"] for k in range(K2)])
            S2, f2, _ = _hazard_marginal("pgompertz", theta2, t2, slopes=slopes2, nodes=self.nodes2)
        return f2 if kind == "dens2" else S2


def fit_lillardhaz(eq1, eq2, data, y1, y2, d2, d1=None,
                    x1=(), x2=(), nodes1=None, nodes2=None, corr=True,
                    start=None, method="L-BFGS-B"):
    """Fit a Lillard-style simultaneous-equations hazard/probit model.

    Parameters
    ----------
    eq1 : {"probit", "lognormal", "pgompertz"}
    eq2 : {"lognormal", "pgompertz"}
    data : mapping of column name -> array-like (e.g. a dict or DataFrame)
    y1 : str
        Equation 1's outcome column: binary 0/1 if eq1="probit", else the
        duration column.
    d1 : str, optional
        Equation 1's failure-indicator column (ignored if eq1="probit").
    y2 : str
        Equation 2's duration column.
    d2 : str
        Equation 2's failure-indicator column.
    x1, x2 : sequence of str
        Covariate column names for each equation's location index.
    nodes1, nodes2 : sequence of float, optional
        Ascending interior nodes; required exactly when the corresponding
        equation is "pgompertz".
    corr : bool
        If False, fixes the copula correlation at 0 and fits the two
        equations independently.
    start : array-like, optional
        Starting values, in the order given by the model's parameter layout.
    method : str
        Passed to scipy.optimize.minimize; default "L-BFGS-B" (found more
        reliable than "BFGS" for these likelihoods -- see the manual).

    Returns
    -------
    LillardhazResult
    """
    if eq1 not in ("probit", "lognormal", "pgompertz"):
        raise ValueError("eq1 must be 'probit', 'lognormal', or 'pgompertz'")
    if eq2 not in ("lognormal", "pgompertz"):
        raise ValueError("eq2 must be 'lognormal' or 'pgompertz'")
    if eq1 == "pgompertz" and nodes1 is None:
        raise ValueError("nodes1 is required when eq1='pgompertz'")
    if eq2 == "pgompertz" and nodes2 is None:
        raise ValueError("nodes2 is required when eq2='pgompertz'")

    n = len(data[y2])
    X1 = np.column_stack([np.ones(n)] + [np.asarray(data[v], dtype=float) for v in x1])
    X2 = np.column_stack([np.ones(n)] + [np.asarray(data[v], dtype=float) for v in x2])

    y1v = np.asarray(data[y1], dtype=float)
    y2v = np.asarray(data[y2], dtype=float)
    d2v = np.asarray(data[d2], dtype=float)
    if eq1 == "probit":
        d1v = y1v
    else:
        d1v = np.asarray(data[d1], dtype=float)

    K1 = len(nodes1) + 1 if eq1 == "pgompertz" else None
    K2 = len(nodes2) + 1 if eq2 == "pgompertz" else None
    names = _param_layout(eq1, eq2, x1, x2, K1, K2, corr)

    if start is None:
        start = np.zeros(len(names))
        if eq1 != "probit":
            start[0] = np.mean(np.log(y1v))
            if eq1 == "lognormal":
                start[names.index("ln_sigma1")] = np.log(np.std(np.log(y1v)))
            else:
                for k in range(K1):
                    start[names.index(f"s1_{k+1}")] = 0.05
        eq2_cons_idx = names.index("eq2:_cons")
        start[eq2_cons_idx] = np.mean(np.log(y2v))
        if eq2 == "lognormal":
            start[names.index("ln_sigma2")] = np.log(np.std(np.log(y2v)))
        else:
            for k in range(K2):
                start[names.index(f"s2_{k+1}")] = 0.05
    else:
        start = np.asarray(start, dtype=float)

    res = minimize(
        _neg_loglik, start, method=method,
        args=(eq1, eq2, X1, X2, y1v, d1v, y2v, d2v, nodes1, nodes2, corr),
        options={"maxiter": 2000},
    )

    par = res.x
    try:
        H = _numerical_hessian(
            lambda p: _neg_loglik(p, eq1, eq2, X1, X2, y1v, d1v, y2v, d2v, nodes1, nodes2, corr),
            par,
        )
        cov = np.linalg.inv(H)
        se = np.sqrt(np.diag(cov))
    except Exception:
        se = np.full(len(par), np.nan)

    params = dict(zip(names, par))
    ses = dict(zip(names, se))

    rho = None
    rho_se = None
    if corr:
        rho = float(np.tanh(params["atanh_rho"]))
        rho_se = float((1 - rho ** 2) * ses["atanh_rho"])
        del params["atanh_rho"]
        del ses["atanh_rho"]

    return LillardhazResult(
        params=params, se=ses, loglik=-res.fun, eq1type=eq1, eq2type=eq2,
        nodes1=nodes1, nodes2=nodes2, corr=corr, rho=rho, rho_se=rho_se,
        x1=list(x1), x2=list(x2), y1name=y1, y2name=y2, n=n,
        converged=bool(res.success),
    )
