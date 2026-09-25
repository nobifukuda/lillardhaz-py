"""lillardhaz: simultaneous-equations hazard/probit models with a
Gaussian-copula correlation, after Lillard (1993)."""

from .models import fit_lillardhaz, LillardhazResult

__all__ = ["fit_lillardhaz", "LillardhazResult"]
__version__ = "0.1.0"
