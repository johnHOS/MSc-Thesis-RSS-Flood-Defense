# %%
"""
results/output_summary.py
==========================
Collect, display, and compare probabilistic results to target failure
probabilities from Table 5.4 of the thesis.

Functions
---------
  compare_to_targets(results)  – tabular comparison + pass/fail
  plot_beta_overview(results)  – bar chart of β vs. target β
  print_unity_checks(uc_dict)  – formatted deterministic UC table
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# %%
def _beta_from_pf(pf: float) -> float:
    """Convert probability of failure to reliability index β."""
    pf = max(pf, 1e-12)   # guard against log(0)
    return -norm.ppf(pf)

# %%
def compare_to_targets(results: dict) -> None:
    """
    Print a formatted comparison table of computed vs. target failure
    probabilities.

    Parameters
    ----------
    results : dict with structure
        {
          "mechanism_name": {
              "pf"     : float,   # computed probability of failure
              "beta"   : float,   # reliability index
              "target" : float,   # target Pf from parameters.py
          },
          ...
        }
    """
    header = (f"{'Mechanism':<22} {'Pf computed':>14} {'Pf target':>12} "
              f"{'β computed':>12} {'β target':>10}  {'Status':>8}")
    line   = "-" * len(header)

    print("\n" + "=" * len(header))
    print("  RELIABILITY ANALYSIS RESULTS  –  RSS Strandeiland")
    print("=" * len(header))
    print(header)
    print(line)

    for name, r in results.items():
        pf_comp  = r.get("pf",     np.nan)
        pf_tgt   = r.get("target", np.nan)
        beta_comp= r.get("beta",   _beta_from_pf(pf_comp) if not np.isnan(pf_comp) else np.nan)
        beta_tgt = _beta_from_pf(pf_tgt) if not np.isnan(pf_tgt) else np.nan

        if np.isfinite(pf_comp):
            pf_text = f"{pf_comp:.2e}"
            beta_text = f"{beta_comp:.3f}"
            status = " ✓ OK" if pf_comp <= pf_tgt else " ✗ FAIL"
        else:
            pf_text = "N/A"
            beta_text = "N/A"
            status = "  N/A"

        print(f"  {name:<20} {pf_text:>14} {pf_tgt:>12.2e} "
              f"{beta_text:>12} {beta_tgt:>10.3f}  {status:>8}")

    print(line)
    print()

# %%
def plot_beta_overview(results: dict,
                       title: str = "Reliability Index β per Failure Mechanism",
                       save_path: str = None) -> None:
    """
    Bar chart comparing computed β to target β for each mechanism.

    Parameters
    ----------
    results   : same dict as for compare_to_targets
    save_path : if given, save figure to this path instead of showing
    """
    from scipy.stats import norm

    mechanisms = list(results.keys())
    betas_comp = []
    betas_tgt  = []

    for r in results.values():
        pf_comp = r.get("pf", np.nan)
        pf_tgt  = r.get("target", np.nan)
        betas_comp.append(_beta_from_pf(pf_comp) if not np.isnan(pf_comp) else 0)
        betas_tgt .append(_beta_from_pf(pf_tgt)  if not np.isnan(pf_tgt)  else 0)

    x    = np.arange(len(mechanisms))
    width= 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(mechanisms) * 1.2), 5))
    bars1 = ax.bar(x - width/2, betas_comp, width, label='Computed β',
                   color='steelblue', alpha=0.85, edgecolor='black')
    bars2 = ax.bar(x + width/2, betas_tgt,  width, label='Target β',
                   color='darkorange', alpha=0.85, edgecolor='black')

    # Annotate bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.05, f'{h:.2f}',
                ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.05, f'{h:.2f}',
                ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(mechanisms, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Reliability index β  [-]')
    ax.set_title(title)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

# %%
def print_unity_checks(uc_dict: dict, title: str = "Unity checks") -> None:
    """
    Nicely print a nested dictionary of unity check results.

    Parameters
    ----------
    uc_dict : dict – can be flat or one level nested
    title   : section header
    """
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

    def _print_flat(d: dict, indent: int = 0) -> None:
        for k, v in d.items():
            prefix = "  " * indent
            if isinstance(v, dict):
                print(f"{prefix}{k}:")
                _print_flat(v, indent + 1)
            elif isinstance(v, bool):
                flag = "✓" if v else "✗"
                print(f"{prefix}  {k:<30} {flag}")
            elif isinstance(v, float):
                print(f"{prefix}  {k:<30} {v:.4g}")
            else:
                print(f"{prefix}  {k:<30} {v}")

    _print_flat(uc_dict)
    print()

# %%
# ===========================================================================
#  PARTIAL SAFETY FACTOR DERIVATION  –  JCSS fixed alpha-factor method
# ===========================================================================
# Reference: JCSS Probabilistic Model Code, Part 1, Section 3.7
#
# When FORM converges, PSFs come directly from the design point:
#     gamma_R = x_k / x*    (resistance variable:  x* < x_k)
#     gamma_S = x* / x_k    (load variable:         x* > x_k)
#
# When FORM does NOT converge, the JCSS method assigns standardised
# sensitivity factors alpha_i and computes the design point analytically:
#     u* = -alpha_i * beta_t   (design point in standard normal space)
#     x* = F^{-1}(Phi(u*))    (back-transformed to physical space)
#
# Standard alpha values:
#     +0.80  dominant resistance   (single resistance variable, or largest R)
#     +0.40  secondary resistance  (other resistance variables)
#     -0.70  dominant load         (single load variable, or largest S)
#     -0.28  secondary load        (= 0.40 × 0.70, other load variables)
#      0.00  inert                 (variable does not influence this LSF)
# ===========================================================================

def psf_from_jcss(dist, alpha_i: float, beta_t: float, x_k: float) -> float:
    """
    Compute a partial safety factor using the JCSS fixed alpha-factor method.

    Parameters
    ----------
    dist    : openturns.Distribution  –  stochastic model for this variable
    alpha_i : float  –  JCSS sensitivity factor (see table above)
    beta_t  : float  –  target reliability index (positive)
    x_k     : float  –  characteristic value of the variable

    Returns
    -------
    gamma   : float  –  partial safety factor (always >= 1.0 by definition)
    """
    from scipy.stats import norm as sp_norm

    if abs(alpha_i) < 1e-9:
        return 1.0   # inert variable — no factor needed

    u_star = -alpha_i * beta_t
    p_star = sp_norm.cdf(u_star)
    x_star = dist.computeQuantile(p_star)[0]

    if alpha_i > 0:   # resistance: x* < x_k  →  gamma = x_k / x*
        return float(x_k / x_star) if x_star > 1e-9 else np.inf
    else:             # load:       x* > x_k  →  gamma = x* / x_k
        return float(x_star / x_k) if x_k    > 1e-9 else np.inf


def compute_psf_table(P) -> dict:
    """
    Compute JCSS partial safety factors for all stochastic mechanisms.

    Alpha assignments per mechanism reflect which variable dominates the
    LSF response:

    toe_stone   : phi_stone is the only variable → dominant resistance
    bearing     : su dominant resistance; q_sur dominant load;
                  tan_phi inert (eta_H = 1 in lsf_bearing); hgwl secondary load
    sliding /
    overturning : tan_phi dominant resistance (drives Ka → F_H);
                  hgwl dominant load (drives internal water pressure);
                  q_sur secondary *resistance* — higher q_sur → higher Vd
                  → higher R_H = Vd × f_k  (stabilising for sliding/rotation)
    rupture     : Rt dominant resistance; tan_phi secondary resistance (lowers Ka);
                  q_sur dominant load (increases sigma_v → T_i);
                  hgwl secondary load
    pullout     : fb dominant resistance; tan_phi secondary resistance;
                  q_sur inert — it multiplies both R_a and T_i via sigma_v,
                  so it cancels in the ratio and has no net effect;
                  hgwl secondary load

    Parameters
    ----------
    P : Parameters module

    Returns
    -------
    psf_dict : nested dict  {mechanism: {variable: {alpha, x_k, x_star, gamma}}}
    """
    from scipy.stats import norm as sp_norm

    # Characteristic values: lower 5% fractile for resistance-role variables,
    # upper 95% fractile for load-role variables — consistent with the Level I
    # convention set out in Chapter 6 ("Characteristic values"), and with the
    # tan_phi/Rt/fb/q_surcharge values already used there for rupture/pull-out.
    #
    # q_surcharge changes ROLE by mechanism: it is a destabilising load in
    # bearing and rupture (higher surcharge -> higher design load -> use the
    # upper 95% fractile), but a stabilising, resistance-like term in sliding
    # and overturning (higher surcharge -> higher friction/stabilising moment
    # -> use the lower 5% fractile, the conservative low-side estimate).
    # Both fractiles are therefore computed and selected per mechanism below.
    xk = {
        "su"        : P.su.computeQuantile(0.05)[0],         # lower 5%  [kPa]
        "phi_stone" : P.phi_stone.computeQuantile(0.05)[0],  # lower 5%  [deg]
        "tan_phi"   : P.tan_phi.computeQuantile(0.05)[0],     # lower 5%  [-]
        "hgwl"      : 0.10,   # reported as a design value only, never as a ratio
        "q_sur_load": P.q_sur.computeQuantile(0.95)[0],       # upper 95% [kN/m2]
        "q_sur_res" : P.q_sur.computeQuantile(0.05)[0],       # lower 5%  [kN/m2]
        "Rt"        : P.Rt.computeQuantile(0.05)[0],          # lower 5%  [kN/m]
        "fb"        : P.fb.computeQuantile(0.05)[0],          # lower 5%  [-]
    }

    # {mechanism_key: (pf_target_key, {var: (dist, alpha, x_k)})}
    _mechanisms = {
        "toe_stone": ("toe_stone", {
            "phi_stone": (P.phi_stone, +0.80, xk["phi_stone"]),
        }),
        "bearing": ("bearing", {
            "su"    : (P.su,    +0.80, xk["su"]),
            "tan_phi": (P.tan_phi, 0.00, xk["tan_phi"]),  # inert — eta_H = 1
            "hgwl"  : (P.hgwl,  -0.28, xk["hgwl"]),
            "q_sur" : (P.q_sur, -0.70, xk["q_sur_load"]),
        }),
        "sliding": ("sliding", {
            "tan_phi": (P.tan_phi, +0.80, xk["tan_phi"]),
            "hgwl"  : (P.hgwl,  -0.70, xk["hgwl"]),
            "q_sur" : (P.q_sur, +0.40, xk["q_sur_res"]),  # stabilising for sliding
        }),
        "overturning": ("overturning", {
            "tan_phi": (P.tan_phi, +0.80, xk["tan_phi"]),
            "hgwl"  : (P.hgwl,  -0.70, xk["hgwl"]),
            "q_sur" : (P.q_sur, +0.40, xk["q_sur_res"]),  # stabilising for rotation
        }),
        "sliding_con.": ("sliding", {
            "tan_phi": (P.tan_phi, +0.80, xk["tan_phi"]),
            "hgwl"  : (P.hgwl,  -0.70, xk["hgwl"]),
            "q_sur" : (P.q_sur, +0.40, xk["q_sur_res"]),
        }),
        "overturning_con.": ("overturning", {
            "tan_phi": (P.tan_phi, +0.80, xk["tan_phi"]),
            "hgwl"  : (P.hgwl,  -0.70, xk["hgwl"]),
            "q_sur" : (P.q_sur, +0.40, xk["q_sur_res"]),
        }),
        "rupture": ("rupture", {
            "tan_phi": (P.tan_phi, +0.40, xk["tan_phi"]),
            "Rt"    : (P.Rt,    +0.80, xk["Rt"]),
            "hgwl"  : (P.hgwl,  -0.28, xk["hgwl"]),
            "q_sur" : (P.q_sur, -0.70, xk["q_sur_load"]),
        }),
        "pullout": ("pullout", {
            "tan_phi": (P.tan_phi, +0.40, xk["tan_phi"]),
            "fb"    : (P.fb,    +0.80, xk["fb"]),
            "hgwl"  : (P.hgwl,  -0.28, xk["hgwl"]),
            "q_sur" : (P.q_sur,  0.00, xk["q_sur_load"]),  # inert in pull-out
        }),
    }

    psf_dict = {}
    for mech_name, (pf_key, var_dict) in _mechanisms.items():
        beta_t = _beta_from_pf(P.pf_target(pf_key))
        psf_dict[mech_name] = {"_beta_t": beta_t}
        for var_name, (dist, alpha_i, x_k) in var_dict.items():
            if abs(alpha_i) < 1e-9:
                x_star = x_k
            else:
                u_star = -alpha_i * beta_t
                x_star = dist.computeQuantile(sp_norm.cdf(u_star))[0]
            gamma = psf_from_jcss(dist, alpha_i, beta_t, x_k)
            # If the characteristic value is already more conservative than
            # the target design value (raw gamma < 1), no additional
            # case-specific factor is adopted — same convention as the
            # q_surcharge conclusion already written for reinforcement rupture.
            gamma_adopted = max(1.0, gamma) if np.isfinite(gamma) else gamma
            psf_dict[mech_name][var_name] = {
                "alpha"        : alpha_i,
                "x_k"          : x_k,
                "x_star"       : round(float(x_star), 4),
                "gamma"        : round(gamma, 3),
                "gamma_adopted": round(gamma_adopted, 3),
            }

    return psf_dict


def print_psf_table(psf_dict: dict) -> None:
    """
    Print a formatted partial safety factor table and compare to CUR198 CC2 values.

    Parameters
    ----------
    psf_dict : output of compute_psf_table()
    """
    # CUR198 CC2 / Eurocode code-specified values for comparison
    CODE = {
        "su"   : 1.40,   # CUR198 γ_su
        "tan_phi": 1.25,  # CUR198 γ_phi (applied to tan phi)
        "phi_stone": "—", # no code PSF used in the erosion check
        "Rt"   : 1.35,   # CUR198 γ_T  (tensile force; applied as load factor)
        "fb"   : 1.20,   # CUR198 γ_mu (bond factor)
        "hgwl" : "—",    # no code PSF; treated as design value
        "q_sur": 1.50,   # Eurocode γ_Q (variable action)
    }

    print(f"\n{'='*90}")
    print(f"  PARTIAL SAFETY FACTORS  –  JCSS fixed-α method  (fallback, FORM unavailable)")
    print(f"  Characteristic values: lower 5% (resistance-role) / upper 95% (load-role) fractile")
    print(f"{'='*90}")
    hdr = (f"  {'Mechanism':<16} {'Variable':<9} {'α':>5} {'β_t':>6} "
           f"{'x_k':>8} {'x*':>9} {'γ_raw':>7} {'γ_adopt':>8} {'γ_code':>8}  {'ratio':>6}")
    print(hdr)
    print(f"  {'-'*86}")

    for mech_name, data in psf_dict.items():
        beta_t = data["_beta_t"]
        first  = True
        for var_name, vals in data.items():
            if var_name == "_beta_t":
                continue
            label   = mech_name if first else ""
            first   = False
            alpha   = vals["alpha"]
            x_k     = vals["x_k"]
            x_star  = vals["x_star"]
            gamma   = vals["gamma"]
            gamma_a = vals["gamma_adopted"]
            code    = CODE.get(var_name, "—")
            alpha_s = f"{alpha:+.2f}" if abs(alpha) > 1e-9 else "  —  "
            code_s  = f"{code:.2f}" if isinstance(code, float) else code
            if isinstance(code, float) and gamma_a > 0:
                ratio_s = f"{gamma_a / code:.2f}"
            else:
                ratio_s = "  —"
            print(f"  {label:<16} {var_name:<9} {alpha_s:>5} {beta_t:>6.3f} "
                  f"{x_k:>8.3g} {x_star:>9.4f} {gamma:>7.3f} {gamma_a:>8.3f} {code_s:>8}  {ratio_s:>6}")
        print(f"  {'-'*86}")

    print()
    print("  γ_raw    = X_k/X* or X*/X_k directly from the JCSS design point")
    print("  γ_adopt  = max(1.0, γ_raw) — no reduction is adopted below the")
    print("             characteristic value itself (matches the reinforcement-")
    print("             rupture surcharge convention used in Chapter 6)")
    print("  ratio    = γ_adopt / γ_code")
    print("  Interpretation:")
    print("  γ_JCSS / γ_code < 1  →  code is MORE conservative than required")
    print("  γ_JCSS / γ_code > 1  →  code is LESS conservative than required")
    print("  α = 0  →  variable is inert in this LSF; γ = 1.0 by definition")
    print()
    print("  Note on hgwl: no code PSF exists; γ_JCSS reflects the full")
    print("  distribution width and should be read as a design value ratio,")
    print("  not a conventional partial factor.")
    print()