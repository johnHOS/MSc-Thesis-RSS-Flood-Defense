# %%
import numpy as np
# from scipy.optimize import brentq
# import sys, os
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import Parameters as P

# %%
def _wave_number(T: float, d: float) -> float:
    omega  = 2 * np.pi / T
    alpha  = omega**2 * d / P.g
    k_hat  = alpha * (np.tanh(alpha))**(-0.5) / d   # Fenton initial guess
    tol    = 1e-5                                   # error margin

    while True:
        k = omega**2 / (P.g * np.tanh(k_hat * d))
        if abs(k - k_hat) < tol:
            break
        k_hat = k

    return k

# %%
# Iterative solve for D50_req  (Shields + Chézy)
def _D50_req(h: float, Hs: float, Tm: float, phi_rad: float) -> float:
    """
    Compute required stone diameter D50_req [m] for one hydraulic load case.

    Parameters
    ----------
    h       : still water level [m+NAP]  (NOT water depth)
    Hs      : significant wave height [m]
    Tm      : spectral wave period [s]
    phi_rad : friction angle [rad] for slope correction factor Ks

    Pulls from Parameters.py: g, z_bed, alpha_toe, K_v, Theta_cr, Delta
    """
    d = h - P.z_bed                                     # water depth at toe [m] TODO
    k = _wave_number(Tm, d)
    u = (np.pi * Hs) / (Tm * np.sinh(k * d))           # orbital velocity [m/s]

    Ks = np.sqrt(max(1.0 - (np.sin(P.alpha_toe) / np.sin(phi_rad))**2, 0.0))
    if Ks == 0:
        print("WARNING: slope angle alpha exceeds friction angle phi — slope is unstable")

    # Iterative Shields + Chézy solve
    def residual(D50):
        C        = 18.0 * np.log10(12.0 * d / (2.0 * D50))
        D50_calc = (P.K_v * u)**2 / (P.Theta_cr * P.Delta * C**2 * Ks)
        return D50_calc - D50

    # Bisection between 1mm and 5m
    D50_low, D50_high = 1e-4, 5.0
    for _ in range(100):
        D50_mid = (D50_low + D50_high) / 2
        if residual(D50_mid) * residual(D50_low) < 0:
            D50_high = D50_mid
        else:
            D50_low  = D50_mid
        if (D50_high - D50_low) < 1e-16:
            break

    return (D50_low + D50_high) / 2



# %%
# Depth-limited breaking check for winter case 

def _apply_breaking_limit(Hs: float, h: float, z_b: float) -> float:
    """
    Cap Hs at depth-limited breaking:  Hs_used = min(Hs, gamma_b * d)
    """
    d      = h - z_b #TODO
    Hs_max = P.gamma_b * d
    return min(Hs, Hs_max)

# %%
# LSF: Top-layer stone stability
#   Z_stone = D50_d - D50_req
#   Stochastic input X = [phi_deg]

def lsf_toe_stone(X):
    """
    X[0] : phi [deg] – internal friction angle of stone material

    Evaluates both the T=2500yr case and the winter low-water sensitivity case.
    The governing load case (largest D50_req) determines Z.

    Returns [Z] where Z < 0 means stone is too small → erosion failure.
    """
    phi_rad = np.radians(X[0])

    # Case 1: T = 833yr  (main reliability-based case)
    D50_833 = _D50_req(P.h_d833, P.H_s833, P.T_m833, phi_rad)

    # Case 2: winter low-water sensitivity case
    # Apply breaking limit: wave may not physically exist at lower water depth
    Hs = _apply_breaking_limit(P.H_winter, P.h_winter, P.z_bed)
    D50_winter     = _D50_req(P.h_winter, Hs, P.T_winter, phi_rad)

    # Governing case: largest required diameter
    D50_req_gov = max(D50_833, D50_winter)

    Z = (P.D50_d - D50_req_gov) / P.D50_d #normalized
    return [Z]

# %%
# LSF: Width of toe protection
#   Z_width = B_toe - B_req
#   All deterministic → dummy stochastic input X = [0.0]

def lsf_toe_width(X):
    """
    No stochastic variables.
    Returns [Z] where Z < 0 means toe protection is too narrow.

    Pulls from Parameters.py: gamma_sf, h_max, cot_eps, cot_beta, B_toe
    """
    Hs    = _apply_breaking_limit(P.H_winter, P.h_winter, P.z_toe)
    # h_max = min(Hs, 0.7 * P.h_ref) TODO
  
    h0 = P.h_ref
    Lw = P.T_winter * np.sqrt(P.g * h0)
    h_max = 0.4 * Hs * (np.sinh((2 * np.pi * h0)/Lw))**-1.35

    B_req = P.gamma_sf * h_max * (P.cot_eps + P.cot_beta) / 2.0
    Z     = (P.B_toe - B_req) / P.B_toe #normalized
    return [Z]


# %%
def unity_checks_erosion() -> dict:
    """Deterministic unity checks for erosion using mean phi = 26 degrees."""
    phi_rad = np.radians(35)

    # Winter case (governing — lower water depth)
    Hs = _apply_breaking_limit(P.H_winter, P.h_winter, P.z_bed)
    D50_req   = _D50_req(P.h_winter, Hs, P.T_winter, phi_rad)

    # Width check
    Hs2 = _apply_breaking_limit(P.H_winter, P.h_winter, P.z_bed)
    h0 = P.h_ref
    Lw = P.T_winter * np.sqrt(P.g * h0)
    h_max = 0.4 * Hs2 * (np.sinh((2 * np.pi * h0)/Lw))**-1.35
    #h_max = min(Hs2, 0.7 * (P.h_ref))
    B_req = P.gamma_sf * h_max * (P.cot_eps + P.cot_beta) / 2.0
     
    return {
        "stone_stability": {
            "D50_d [m]"  : P.D50_d,
            "D50_req [m]": round(D50_req, 4),
            "UC"         : round(D50_req / P.D50_d, 3),
            "passes"     : D50_req <= P.D50_d,
        },
        "toe_width": {
            "B_toe [m]"  : P.B_toe,
            "B_req [m]"  : round(B_req, 3),
            "UC"         : round(B_req / P.B_toe, 3),
            "passes"     : bool(B_req <= P.B_toe),
        },
    }

# %%


# %%



