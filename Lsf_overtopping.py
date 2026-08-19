"""Deterministic crest-level verification for overtopping and overflow.

LSF: Z_overtopping = q_allow - q(Rc, Hm0, Tm) >= 0
     Z_overflow = z_crest - SWL >= 0

The hydraulic equations are also shared with the response-grid, Monte Carlo
and FORM analyses. The joint-distribution helpers preserve those workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import Parameters as P


# Hydra-NL marginal return levels from the local reports for MM_3_hy10.
RETURN_PERIOD_YR = np.array(
    [10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000], dtype=float
)
SWL_RETURN_LEVEL = np.array(
    [1.397, 1.526, 1.645, 1.755, 1.884, 2.013, 2.171, 2.324, 2.498],
    dtype=float,
)
HM0_RETURN_LEVEL = np.array(
    [0.957, 1.052, 1.149, 1.236, 1.331, 1.416, 1.507, 1.591, 1.684],
    dtype=float,
)
TM_RETURN_LEVEL = np.array(
    [3.828, 4.096, 4.324, 4.505, 4.657, 4.810, 4.957, 5.048, 5.142],
    dtype=float,
)

# ---------------------------------------------------------------------------
# Hydra-NL SWL-Hm0 co-occurrence for the governing 0-100 degree wave sector of
# MM_3_hy10. For each Hm0 return level the wave-height report resolves the
# co-occurring local still-water level by wind direction. The structure only
# takes wave load from the 0-100 deg sector, so only the directions 30, 60 and 90
# deg are retained. This differs from the report's OVERALL main illustration
# point, which at the higher return periods is direction 360 (north) and would
# import a much higher, sector-irrelevant water level (e.g. 1.79 m+NAP at 1/10000
# instead of ~1.43 m+NAP from the sector). Within each return period the
# per-direction exceedance-frequency contributions (ov.freq) are renormalised to
# weights summing to one, and the representative co-occurring SWL is their
# frequency-weighted mean, i.e. E[SWL | Hm0, direction in sector]. These paired
# (SWL, Hm0) values are used to DERIVE the copula correlation instead of assuming
# it. See Hawkes et al. (2002); Deltares (2017), sec. 2.3.4.
#
# Columns: return period [yr], wind direction [deg], co-occurring local SWL
#          [m+NAP], design Hm0 [m], ov.freq contribution within the sector [%].
SECTOR_COOCCURRENCE = np.array(
    [
        [10, 30, 0.91, 0.96, 22.6],
        [10, 60, 0.87, 0.96, 43.8],
        [10, 90, 0.88, 0.96, 2.0],
        [100, 30, 1.16, 1.15, 18.6],
        [100, 60, 1.09, 1.15, 37.4],
        [100, 90, 1.05, 1.15, 1.0],
        [500, 30, 1.30, 1.28, 16.0],
        [500, 60, 1.27, 1.28, 32.5],
        [500, 90, 1.05, 1.28, 0.5],
        [833, 30, 1.37, 1.32, 15.1],
        [833, 60, 1.34, 1.32, 31.2],
        [833, 90, 1.11, 1.32, 0.4],
        [1000, 30, 1.39, 1.33, 14.9],
        [1000, 60, 1.37, 1.33, 30.8],
        [1000, 90, 1.12, 1.33, 0.4],
        [1500, 30, 1.22, 1.36, 14.3],
        [1500, 60, 1.19, 1.36, 29.9],
        [1500, 90, 1.16, 1.36, 0.3],
        [2000, 30, 1.25, 1.38, 14.0],
        [2000, 60, 1.22, 1.38, 29.0],
        [2000, 90, 1.20, 1.38, 0.3],
        [5000, 30, 1.36, 1.45, 12.4],
        [5000, 60, 1.33, 1.45, 25.6],
        [5000, 90, 1.32, 1.45, 0.2],
        [10000, 30, 1.45, 1.51, 11.1],
        [10000, 60, 1.42, 1.51, 23.4],
        [10000, 90, 1.40, 1.51, 0.1],
    ],
    dtype=float,
)


def sector_cooccurrence_points(
    table: np.ndarray = SECTOR_COOCCURRENCE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse the sector table to one (Hm0, SWL) pair per return period.

    Within each return period the sector-direction exceedance-frequency
    contributions are renormalised to weights summing to one, and the co-occurring
    still-water level is their frequency-weighted mean, E[SWL | Hm0, direction in
    sector]. Returns (return_period_yr, Hm0, co-occurring SWL).
    """
    periods = np.unique(table[:, 0])
    hm0 = np.empty_like(periods)
    swl = np.empty_like(periods)
    for i, period in enumerate(periods):
        rows = table[table[:, 0] == period]
        weight = rows[:, 4] / rows[:, 4].sum()
        swl[i] = float(np.sum(weight * rows[:, 2]))
        hm0[i] = float(rows[0, 3])  # Hm0 is the return level, constant within a block
    return periods, hm0, swl


(
    COOCCURRENCE_RETURN_PERIOD_YR,
    COOCCURRENCE_HM0,
    COOCCURRENCE_SWL,
) = sector_cooccurrence_points()


@dataclass(frozen=True)
class StructureConfig:
    """Geometry and admissible discharge used by crest-level LSFs."""

    z_crest: float
    z_bed: float
    q_allow: float
    g: float = 9.81


@dataclass(frozen=True)
class GumbelFit:
    """Gumbel distribution fitted to published return levels."""

    loc: float
    scale: float
    rmse: float
    name: str

    def ppf(self, probability: np.ndarray | float) -> np.ndarray:
        p = np.clip(np.asarray(probability, dtype=float), 1e-12, 1.0 - 1e-12)
        return self.loc - self.scale * np.log(-np.log(p))

    def cdf(self, value: np.ndarray | float) -> np.ndarray:
        z = (np.asarray(value, dtype=float) - self.loc) / self.scale
        return np.exp(-np.exp(-z))


def project_structure_config() -> StructureConfig:
    """Create a StructureConfig from Parameters.py."""
    return StructureConfig(
        z_crest=float(P.z_crest),
        z_bed=float(P.z_bed),
        q_allow=float(P.q_allow),
        g=float(getattr(P, "g", 9.81)),
    )


def fit_gumbel_return_levels(
    return_period: Sequence[float], values: Sequence[float], name: str
) -> GumbelFit:
    """Fit a Gumbel annual-maximum distribution to return-level points."""
    t = np.asarray(return_period, dtype=float)
    x = np.asarray(values, dtype=float)
    if np.any(t <= 1.0):
        raise ValueError("All return periods must be greater than one year.")
    if t.shape != x.shape:
        raise ValueError("return_period and values must have equal length.")

    p = 1.0 - 1.0 / t
    reduced_variate = -np.log(-np.log(p))
    design_matrix = np.column_stack([np.ones_like(reduced_variate), reduced_variate])
    loc, scale = np.linalg.lstsq(design_matrix, x, rcond=None)[0]
    if scale <= 0.0:
        raise RuntimeError(f"Non-positive Gumbel scale fitted for {name}.")

    prediction = loc + scale * reduced_variate
    rmse = float(np.sqrt(np.mean((prediction - x) ** 2)))
    return GumbelFit(float(loc), float(scale), rmse, name)


def fit_hydra_margins() -> dict[str, GumbelFit]:
    """Fit SWL, Hm0 and Tm-1,0 marginals to the Hydra-NL return levels."""
    return {
        "SWL": fit_gumbel_return_levels(
            RETURN_PERIOD_YR, SWL_RETURN_LEVEL, "SWL [m+NAP]"
        ),
        "Hm0": fit_gumbel_return_levels(RETURN_PERIOD_YR, HM0_RETURN_LEVEL, "Hm0 [m]"),
        "Tm": fit_gumbel_return_levels(RETURN_PERIOD_YR, TM_RETURN_LEVEL, "Tm-1,0 [s]"),
    }


def spearman_to_gaussian_copula_rho(rho_s: float) -> float:
    """Convert Spearman rank correlation to Gaussian-copula correlation."""
    if not -0.999 < rho_s < 0.999:
        raise ValueError("Spearman rho must lie strictly between -0.999 and 0.999.")
    return float(2.0 * np.sin(np.pi * rho_s / 6.0))


def gaussian_copula_rho_to_spearman(rho_g: float) -> float:
    """Inverse of spearman_to_gaussian_copula_rho: latent -> Spearman rank rho."""
    rho_g = float(np.clip(rho_g, -1.0, 1.0))
    return float((6.0 / np.pi) * np.arcsin(rho_g / 2.0))


def cooccurrence_dependence_table(
    margins: dict[str, GumbelFit] | None = None,
) -> dict[str, np.ndarray]:
    """Regime-resolved SWL-Hm0 dependence derived from Hydra-NL co-occurrence points.

    Method (Hawkes et al., 2002 normal-score / bivariate-normal dependence):
    each design Hm0 is placed at its marginal non-exceedance quantile and the
    *co-occurring* local SWL at its marginal quantile; both are mapped to standard
    normal scores z_Hm0, z_SWL. Under a Gaussian copula the conditional mean obeys
    E[z_SWL | z_Hm0] = rho_g * z_Hm0, so a per-point latent correlation follows as
    rho_g = z_SWL / z_Hm0. Because the Markermeer water level and waves are both
    functions of the base stochastics wind + meerpeil (Deltares, 2017, sec. 2.3.4),
    this rho_g is regime-dependent: it is near zero in the moderate tail (the design
    wave from the sector then co-occurs with only a modest water level) and rises
    into the storm-dominated tail. It is evaluated on the 0-100 deg sector points
    only (SECTOR_COOCCURRENCE). Note that the co-occurring water level from
    Hydra-NL illustration points is not strictly monotone in the return period
    (the most-probable point jumps between meerpeil- and wind-driven regimes), so
    the per-point rho_g dips around 1/1500-1/2000; see the estimator below.
    """
    from scipy.stats import norm

    if margins is None:
        margins = fit_hydra_margins()
    u_hm0 = np.clip(margins["Hm0"].cdf(COOCCURRENCE_HM0), 1e-12, 1.0 - 1e-12)
    u_swl = np.clip(margins["SWL"].cdf(COOCCURRENCE_SWL), 1e-12, 1.0 - 1e-12)
    z_hm0 = norm.ppf(u_hm0)
    z_swl = norm.ppf(u_swl)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho_g = np.where(z_hm0 > 0.0, z_swl / z_hm0, np.nan)
    return {
        "return_period_yr": COOCCURRENCE_RETURN_PERIOD_YR,
        "Hm0": COOCCURRENCE_HM0,
        "SWL_cooccurring": COOCCURRENCE_SWL,
        "SWL_return_period_yr": 1.0 / (1.0 - u_swl),
        "z_Hm0": z_hm0,
        "z_SWL": z_swl,
        "rho_g": rho_g,
        "spearman_rho": np.array([gaussian_copula_rho_to_spearman(r) for r in rho_g]),
    }


def estimate_gaussian_copula_rho_from_cooccurrence(
    design_return_period_yr: float = 833.0,
    margins: dict[str, GumbelFit] | None = None,
    as_spearman: bool = True,
) -> float:
    """Derive the copula correlation at a chosen design return period.

    Log-linear interpolation of the per-point rho_g over the co-occurrence points
    with return period >= 100 yr (the moderate/low points are unreliable because
    the co-occurring water level sits below the SWL marginal support). Returns the
    Spearman rank correlation by default (the argument of
    build_openturns_hydraulic_distribution), or the latent rho_g if as_spearman
    is False.
    """
    tbl = cooccurrence_dependence_table(margins)
    mask = (tbl["return_period_yr"] >= 100.0) & np.isfinite(tbl["rho_g"]) & (tbl["rho_g"] > 0.0)
    log_t = np.log(tbl["return_period_yr"][mask])
    rho = tbl["rho_g"][mask]
    rho_g = float(np.interp(np.log(design_return_period_yr), log_t, rho,
                            left=rho[0], right=rho[-1]))
    return gaussian_copula_rho_to_spearman(rho_g) if as_spearman else rho_g


# Copula correlation DERIVED at the 1/833 yr design frequency from the Hydra-NL
# co-occurrence points, restricted to the 0-100 deg wave sector (Spearman ~ 0.37).
# It replaces both the earlier hand-set 0.60/0.92 assumption and the all-direction
# co-occurrence, which had inflated the tail dependence by importing the direction-
# 360 water level. The dependence is regime-dependent; pass a different
# design_return_period_yr for a mechanism whose design point sits deeper in the
# tail (Spearman ~ 0.40 at 1/10,000).
DEFAULT_SPEARMAN_RHO = round(estimate_gaussian_copula_rho_from_cooccurrence(833.0), 2)


def build_openturns_hydraulic_distribution(
    spearman_rho: float = DEFAULT_SPEARMAN_RHO,
    margins: dict[str, GumbelFit] | None = None,
):
    """Build the correlated OpenTURNS distribution for X=[SWL, Hm0].

    ``spearman_rho`` defaults to ``DEFAULT_SPEARMAN_RHO``, which is derived from the
    Hydra-NL co-occurrence illustration points rather than assumed. Pass an explicit
    value to run a dependence sensitivity.
    """
    try:
        import openturns as ot
    except ImportError as exc:
        raise ImportError("OpenTURNS is required for hydraulic FORM analysis.") from exc

    if margins is None:
        margins = fit_hydra_margins()

    latent_rho = spearman_to_gaussian_copula_rho(float(spearman_rho))
    correlation = ot.CorrelationMatrix(2)
    correlation[0, 1] = latent_rho
    correlation[1, 0] = latent_rho

    copula = ot.NormalCopula(correlation)
    swl_distribution = ot.Gumbel(margins["SWL"].scale, margins["SWL"].loc)
    hm0_distribution = ot.Gumbel(margins["Hm0"].scale, margins["Hm0"].loc)
    distribution_class = getattr(ot, "ComposedDistribution", ot.JointDistribution)
    distribution = distribution_class([swl_distribution, hm0_distribution], copula)
    distribution.setDescription(["SWL", "Hm0"])
    return distribution


def derive_tm_from_hm0(
    hm0: float | np.ndarray,
    margins: dict[str, GumbelFit] | None = None,
) -> np.ndarray:
    """Derive Tm-1,0 from the same wave-severity quantile as Hm0."""
    if margins is None:
        margins = fit_hydra_margins()
    u_wave = np.clip(margins["Hm0"].cdf(hm0), 1e-12, 1.0 - 1e-12)
    return margins["Tm"].ppf(u_wave)


def compute_q(
    SWL: float | np.ndarray,
    Hs: float | np.ndarray,
    Tm: float | np.ndarray,
    *,
    z_crest: float | None = None,
    z_bed: float | None = None,
    g: float | None = None,
) -> tuple[float | np.ndarray, str | np.ndarray]:
    """Compute mean overtopping discharge q [m3/s/m].

    Uses the EurOtop (2018) vertical-wall equations. Scalar inputs are used
    by the deterministic check and array inputs by the response-grid and
    Monte Carlo calculations.
    """
    z_crest = float(P.z_crest if z_crest is None else z_crest)
    z_bed = float(P.z_bed if z_bed is None else z_bed)
    g = float(P.g if g is None else g)

    SWL, Hs, Tm = np.broadcast_arrays(
        np.asarray(SWL, dtype=float),
        np.asarray(Hs, dtype=float),
        np.asarray(Tm, dtype=float),
    )
    scalar_input = SWL.shape == ()

    Rc = z_crest - SWL
    d = SWL - z_bed
    Lm = g * Tm**2 / (2.0 * np.pi)
    xi = d**2 / (Hs * Lm)
    s_m = Hs / Lm
    relative_freeboard = Rc / Hs
    q_scale = np.sqrt(g * Hs**3)

    # Impulsive waves: EurOtop equations 7.9 and 7.10.
    impulsive_factor = np.sqrt(Hs / (d * s_m))
    q_impulsive = np.where(relative_freeboard >= 1.35,
        q_scale * 0.002 * impulsive_factor * relative_freeboard ** (-3.0),
        q_scale * 0.0155 * impulsive_factor * np.exp(-2.2 * relative_freeboard),    )

    # Non-impulsive waves: EurOtop equation 7.6.
    q_non_impulsive = (
        q_scale * 0.062 * np.exp(-2.61 * relative_freeboard)
    )

    q = np.where(xi < 0.23, q_impulsive, q_non_impulsive)
    regime = np.where(xi < 0.23, "impulsive", "non-impulsive")

    if scalar_input:
        return float(q.item()), str(regime.item())
    return q, regime


def make_lsf_overtopping_joint(
    config: StructureConfig | None = None,
    margins: dict[str, GumbelFit] | None = None,
):
    """Return the FORM-compatible overtopping LSF for X=[SWL, Hm0]."""
    if config is None:
        config = project_structure_config()
    if margins is None:
        margins = fit_hydra_margins()

    def lsf(X: Sequence[float]) -> list[float]:
        if len(X) != 2:
            raise ValueError("Joint overtopping LSF expects X=[SWL, Hm0].")

        swl = float(X[0])
        hm0 = float(X[1])
        tm10 = float(np.asarray(derive_tm_from_hm0(hm0, margins)).item())

        # Overflow is checked with its own LSF below.
        if swl >= config.z_crest:
            return [float(config.q_allow)]

        q, _ = compute_q(
            swl,
            hm0,
            tm10,
            z_crest=config.z_crest,
            z_bed=config.z_bed,
            g=config.g,
        )
        return [float(config.q_allow - q)]

    return lsf


def make_lsf_overflow_joint(config: StructureConfig | None = None):
    """Return the overflow LSF for X=[SWL, Hm0]: Z=z_crest-SWL.

    Hm0 is intentionally present so overflow can use the same joint hydraulic
    distribution as overtopping. It is inert in this LSF because overflow is
    controlled by the still-water level only.
    """
    if config is None:
        config = project_structure_config()

    def lsf(X: Sequence[float]) -> list[float]:
        if len(X) != 2:
            raise ValueError("Overflow LSF expects X=[SWL, Hm0].")
        return [float(config.z_crest - X[0])]

    return lsf


def check_overtopping(
    label: str, SWL: float, Hs: float, Tm: float, q_allow: float
) -> dict:
    """Run the deterministic overtopping LSF: Z=q_allow-q."""
    Rc = P.z_crest - SWL
    d = SWL - P.z_bed
    Lm = P.g * Tm**2 / (2.0 * np.pi)
    xi = d**2 / (Hs * Lm)

    q, regime = compute_q(SWL, Hs, Tm)
    Z = q_allow - q
    UC = q / q_allow

    print(f"\n{'=' * 55}")
    print(f"  Overtopping check - {label}")
    print(f"{'=' * 55}")
    print(f"  Crest level        z_crest  = {P.z_crest:.2f} m+NAP")
    print(f"  Still water level  SWL      = {SWL:.2f} m+NAP")
    print(f"  Actual freeboard   Rc       = {Rc:.3f} m")
    print(f"  Water depth        d        = {d:.3f} m")
    print(f"  Wavelength         Lm       = {Lm:.3f} m")
    print(f"  Impulsiveness      xi       = {xi:.4f} -> {regime}")
    print(f"  Computed discharge q        = {q * 1000:.4f} l/s/m")
    print(f"  Allowable discharge q_allow = {q_allow * 1000:.1f} l/s/m")
    print(f"  Z = q_allow - q            = {Z * 1000:.4f} l/s/m")
    print(f"  UC = q / q_allow           = {UC:.3f} -> {'OK' if UC <= 1 else 'FAIL'}")

    return {
        "label": label,
        "Rc [m]": round(Rc, 3),
        "d [m]": round(d, 3),
        "xi [-]": round(float(xi), 4),
        "regime": regime,
        "q [m3/s/m]": round(float(q), 6),
        "q_allow [m3/s/m]": q_allow,
        "Z [m3/s/m]": round(float(Z), 6),
        "UC [-]": round(float(UC), 3),
        "passes": bool(UC <= 1.0),
    }


def check_overflow(label: str, SWL: float) -> dict:
    """Run the deterministic overflow LSF: Z=z_crest-SWL."""
    z = P.z_crest - SWL
    print(f"\n  Overflow check - {label}")
    print(f"  z_crest = {P.z_crest:.2f} m+NAP,  SWL = {SWL:.2f} m+NAP")
    print(f"  Z = {z:.3f} m -> {'OK' if z > 0 else 'FAIL'}")
    return {"Z_overflow [m]": round(z, 3), "passes": z > 0}


if __name__ == "__main__":
    check_overtopping(
        "ULS (T = 833 yr)",
        SWL=P.h_d833,
        Hs=P.H_s833,
        Tm=P.T_m833,
        q_allow=P.q_allow,
    )
    check_overflow("ULS", SWL=P.h_d833)

    check_overtopping(
        "SLS (T = 10 yr)",
        SWL=0.53,
        Hs=0.96,
        Tm=3.83,
        q_allow=0.1 / 1000,
    )
    check_overflow("SLS", SWL=0.53)