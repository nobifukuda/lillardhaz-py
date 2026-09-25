import numpy as np
from lillardhaz import fit_lillardhaz


def test_lognormal_lognormal_correlated():
    np.random.seed(42)
    n = 20000
    x1 = np.random.randn(n)
    x2 = np.random.binomial(1, 0.5, n)
    b10, b11, lnsig1 = 1.0, -0.3, -0.2
    b20, b21, lnsig2 = 0.6, 0.4, 0.1
    rho_true = 0.5

    u1 = np.random.randn(n)
    u2 = rho_true * u1 + np.sqrt(1 - rho_true ** 2) * np.random.randn(n)
    t1 = np.exp(b10 + b11 * x1 + np.exp(lnsig1) * u1)
    t2 = np.exp(b20 + b21 * x2 + np.exp(lnsig2) * u2)
    c1 = -np.log(np.random.rand(n)) / 0.03
    c2 = -np.log(np.random.rand(n)) / 0.03
    data = dict(
        time1=np.minimum(t1, c1), event1=(t1 <= c1).astype(int),
        time2=np.minimum(t2, c2), event2=(t2 <= c2).astype(int),
        x1=x1, x2=x2,
    )

    fit = fit_lillardhaz("lognormal", "lognormal", data,
                          y1="time1", d1="event1", y2="time2", d2="event2",
                          x1=["x1"], x2=["x2"])

    assert abs(fit.params["eq1:x1"] - b11) < 0.05
    assert abs(fit.params["eq1:_cons"] - b10) < 0.05
    assert abs(fit.params["eq2:x2"] - b21) < 0.05
    assert abs(fit.params["ln_sigma1"] - lnsig1) < 0.05
    assert abs(fit.params["ln_sigma2"] - lnsig2) < 0.05
    assert abs(fit.rho - rho_true) < 0.05


def test_probit_lognormal_correlated():
    np.random.seed(7)
    n = 20000
    x1 = np.random.randn(n)
    x2 = np.random.binomial(1, 0.5, n)
    g0, g1 = 0.3, -0.5
    b20, b21, lnsig2 = 0.7, 0.35, -0.15
    rho_true = -0.4

    e1 = np.random.randn(n)
    u2 = rho_true * e1 + np.sqrt(1 - rho_true ** 2) * np.random.randn(n)
    y1 = (g0 + g1 * x1 + e1 > 0).astype(int)
    t2 = np.exp(b20 + b21 * x2 + np.exp(lnsig2) * u2)
    c2 = -np.log(np.random.rand(n)) / 0.03
    data = dict(
        y1=y1, time2=np.minimum(t2, c2), event2=(t2 <= c2).astype(int),
        x1=x1, x2=x2,
    )

    fit = fit_lillardhaz("probit", "lognormal", data,
                          y1="y1", y2="time2", d2="event2",
                          x1=["x1"], x2=["x2"])

    assert abs(fit.params["eq1:x1"] - g1) < 0.05
    assert abs(fit.params["eq1:_cons"] - g0) < 0.05
    assert abs(fit.params["eq2:x2"] - b21) < 0.05
    assert abs(fit.rho - rho_true) < 0.05


def test_pgompertz_pgompertz_correlated():
    np.random.seed(55)
    n = 25000
    x1 = np.random.randn(n)
    x2 = np.random.binomial(1, 0.5, n)

    b0, b1, alpha0, alpha1, alpha2 = -1.3, 0.3, -0.4, 0.35, -0.1
    lev2_0, lev2_1, gamma0, gamma1, gamma2 = -0.9, -0.25, -0.1, 0.25, -0.2
    rho_true = 0.35

    L1a = b0 + b1 * x1 + alpha0
    L1b_off = alpha1 * 5
    L2a = lev2_0 + lev2_1 * x2 + gamma0
    L2b_off = gamma1 * 4

    e1 = np.random.randn(n)
    e2 = rho_true * e1 + np.sqrt(1 - rho_true ** 2) * np.random.randn(n)
    U1, U2 = norm_cdf(e1), norm_cdf(e2)
    Htarget1 = -np.log(1 - U1)
    Htarget2 = -np.log(1 - U2)

    with np.errstate(invalid="ignore"):
        seg1a_true = np.exp(L1a) / alpha1 * (np.exp(alpha1 * 5) - 1)
        t1 = np.where(
            Htarget1 < seg1a_true,
            np.log(1 + Htarget1 * alpha1 / np.exp(L1a)) / alpha1,
            5 + np.log(1 + (Htarget1 - seg1a_true) * alpha2 / np.exp(L1a + L1b_off)) / alpha2,
        )
        seg2a_true = np.exp(L2a) / gamma1 * (np.exp(gamma1 * 4) - 1)
        t2 = np.where(
            Htarget2 < seg2a_true,
            np.log(1 + Htarget2 * gamma1 / np.exp(L2a)) / gamma1,
            4 + np.log(1 + (Htarget2 - seg2a_true) * gamma2 / np.exp(L2a + L2b_off)) / gamma2,
        )
    # a negative terminal slope makes the segment's cumulative hazard bounded
    # as t -> inf (a defective distribution); a few extreme draws exceed what's
    # reachable and come out NaN -- those observations simply never fail.
    t1 = np.where(np.isnan(t1), np.inf, t1)
    t2 = np.where(np.isnan(t2), np.inf, t2)

    c1 = -np.log(np.random.rand(n)) / 0.03
    c2 = -np.log(np.random.rand(n)) / 0.03
    time1 = np.maximum(np.minimum(t1, c1), 1e-4)
    time2 = np.maximum(np.minimum(t2, c2), 1e-4)
    data = dict(
        time1=time1, event1=(t1 <= c1).astype(int),
        time2=time2, event2=(t2 <= c2).astype(int),
        x1=x1, x2=x2,
    )

    fit = fit_lillardhaz("pgompertz", "pgompertz", data,
                          y1="time1", d1="event1", y2="time2", d2="event2",
                          x1=["x1"], x2=["x2"], nodes1=[5], nodes2=[4])

    assert abs(fit.params["eq1:_cons"] - (b0 + alpha0)) < 0.06
    assert abs(fit.params["eq1:x1"] - b1) < 0.03
    assert abs(fit.params["s1_1"] - alpha1) < 0.03
    assert abs(fit.params["s1_2"] - alpha2) < 0.03
    assert abs(fit.params["eq2:_cons"] - (lev2_0 + gamma0)) < 0.06
    assert abs(fit.params["eq2:x2"] - lev2_1) < 0.03
    assert abs(fit.params["s2_1"] - gamma1) < 0.03
    assert abs(fit.params["s2_2"] - gamma2) < 0.03
    assert abs(fit.rho - rho_true) < 0.03


def test_nocorr_reduces_to_independent_fits():
    np.random.seed(88)
    n = 20000
    x1 = np.random.randn(n)
    x2 = np.random.binomial(1, 0.5, n)
    b10, b11, lnsig1 = 1.0, -0.3, -0.2
    b20, b21, lnsig2 = 0.6, 0.4, 0.1

    t1 = np.exp(b10 + b11 * x1 + np.exp(lnsig1) * np.random.randn(n))
    t2 = np.exp(b20 + b21 * x2 + np.exp(lnsig2) * np.random.randn(n))
    c1 = -np.log(np.random.rand(n)) / 0.03
    c2 = -np.log(np.random.rand(n)) / 0.03
    data = dict(
        time1=np.minimum(t1, c1), event1=(t1 <= c1).astype(int),
        time2=np.minimum(t2, c2), event2=(t2 <= c2).astype(int),
        x1=x1, x2=x2,
    )

    fit = fit_lillardhaz("lognormal", "lognormal", data,
                          y1="time1", d1="event1", y2="time2", d2="event2",
                          x1=["x1"], x2=["x2"], corr=False)

    assert fit.corr is False
    assert abs(fit.params["eq1:x1"] - b11) < 0.05
    assert abs(fit.params["eq2:x2"] - b21) < 0.05


def norm_cdf(x):
    from scipy.stats import norm
    return norm.cdf(x)
