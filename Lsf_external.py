# %%
import numpy as np
import Parameters as P

# %%
def _vertical_load(hgwl, q_surcharge, factored=True):
    """
    Design vertical load Vd [kN/m] per Eurocode partial factor rule.

    Gk = self-weight of fill = (gamma_fill * h1 + gamma_fill_sat * h2) * B
    Qk = traffic surcharge   = q_surcharge * B
    
    h1 = depth of unsaturated zone (z_crest to GWL) [m]
    h2 = depth of saturated zone   (GWL to z_bed)   [m]

    Partial factors:
        1.35 Gk + 1.50 Qk  if Qk/Gk < 0.2
        1.20 Gk + 1.50 Qk  if Qk/Gk >= 0.2
    """
    h1 = max(P.z_crest - hgwl, 0.0)
    h2 = max(hgwl - P.z_bed, 0.0)
    Gk = (P.gamma_fill * h1 + P.gamma_fill_sat * h2) * P.B
    Qk = q_surcharge * P.B
    if not factored:
        return Gk + Qk            # characteristic — use in ALL LSFs
    if Gk > 0 and (Qk / Gk) >= 0.2:
        return 1.20 * Gk + 1.50 * Qk
    return 1.35 * Gk + 1.50 * Qk

# %%
# Horizontal earth pressure load  F_H  [kN/m]

def _horizontal_load(phi_deg: float, hgwl: float,
                     case: str = "operational",
                     q_surcharge: float = 0.0) -> tuple[float, float]:
    """
    Compute total net horizontal load F_H [kN/m] and its effective lever arm
    z_arm [m] measured from the base of the structure.

    The backfill is split into:
      - Unsaturated zone (above GWL):  height d1, unit weight gamma_fill
      - Saturated zone   (below GWL):  height d2, unit weight gamma_fill_eff
        plus internal hydrostatic water pressure

    For Case 1 (operational): external lake water at h_d833 is subtracted.
    For Case 2 (construction): no external water pressure.

    Parameters
    ----------
    phi_deg : friction angle [degrees]
    hgwl    : groundwater level inside structure [m+NAP]
    case    : 'operational' or 'construction'
    """
    phi_rad = np.radians(phi_deg)
    Ka      = (1 - np.sin(phi_rad)) / (1 + np.sin(phi_rad))

    # Heights of the two zones [m]
    d1 = max(P.z_crest - hgwl,   0.0)   # unsaturated zone (top of wall to GWL)
    d2 = max(hgwl      - P.z_bed, 0.0)  # saturated zone   (GWL to foundation)

    # --- Earth pressure resultants ---
    # Unsaturated zone: triangular, acts at d2 + d1/3 from base
    F1     = 0.5 * Ka * P.gamma_fill * d1**2
    z1     = d2 + d1 / 3

    # Saturated zone, rectangular part (overburden from unsaturated zone)
    F2r    = Ka * P.gamma_fill * d1 * d2
    z2r    = d2 / 2

    # Saturated zone, triangular part (effective weight of saturated soil)
    F2t    = 0.5 * Ka * P.gamma_fill_eff * d2**2
    z2t    = d2 / 3

    # Internal water pressure (hydrostatic, acts on backfill side)
    F_w_in = 0.5 * P.gamma_w * d2**2
    z_win  = d2 / 3

    # External water pressure (opposes internal; depends on load case)
    if case == "operational":
        d_ext   = max(P.h_winter - P.z_bed, 0.0)   # external water depth from base
        F_w_ext = 0.5 * P.gamma_w * d_ext**2
        z_wext  = d_ext / 3
    else:
        # Construction: cofferdam in place, no external water on RSS
        F_w_ext = 0.0
        z_wext  = 0.0


    F_surch = Ka * q_surcharge * (d1 + d2)
    z_surch = (d1 + d2) / 2.0






    # --- Total net horizontal force and weighted lever arm ---
    F_H = F1 + F2r + F2t + F_w_in + F_surch - F_w_ext

    # Weighted lever arm (avoid division by zero if F_H ≈ 0)
    if abs(F_H) > 1e-6:
        M_total = (F1*z1 + F2r*z2r + F2t*z2t
                   + F_w_in*z_win + F_surch*z_surch - F_w_ext*z_wext)
        z_arm = M_total / F_H
    else:
        z_arm = P.H / 3   # fallback

    return F_H, z_arm


# %%
def lsf_bearing(X):
    """
    X[0] : su    [kPa]   – undrained shear strength of clay
    X[1] : tan_phi [-]    – tangent of the backfill friction angle
    X[2] : hgwl  [m+NAP] – groundwater level inside structure

    Evaluated for the operational case (governing for bearing).
    Returns [Z] where Z < 0 means bearing failure.
    """
    s_u      = X[0]
    tan_phi = X[1]
    phi_deg = np.degrees(np.arctan(tan_phi))
    hgwl    = X[2]
    q_surcharge = X[3]

    Vd = _vertical_load(hgwl, q_surcharge, factored=False)
    F_H, z_arm = _horizontal_load(phi_deg, hgwl, case="operational",
                                  q_surcharge=q_surcharge)

    e = abs(F_H * z_arm) / Vd
    A_prime = max(P.B - 2.0 * e, 0.0)   # effective foundation width [m/m]

    # Simplified horizontal-load reduction multiplier.
    # If the ISO-based penalty term is
    #   i_c = 0.5 * (1 - sqrt(1 - F_H / (A_prime * s_u))),
    # this adopted multiplier is eta_H = 1 - i_c. This is not the full
    # ISO 19901-4 bearing-capacity formulation.

    denom = A_prime * s_u
    ratio = min(F_H / denom, 1.0) if denom > 0 else 1.0
    eta_H = 0.5 * (1.0 + np.sqrt(max(1.0 - ratio, 0.0)))





    q0 = max(P.gamma_clay - P.gamma_w, 0.0) * P.d_emb
    q_ult     = P.N_c * s_u * eta_H  + q0   # simplified ultimate bearing pressure [kPa]
    R_bearing = A_prime * q_ult             # bearing resistance [kN/m]

    Z = (R_bearing - Vd) / Vd
    return [Z]


# %%





















































def lsf_sliding(X, case: str = "operational"):
    """
    X[0] : tan_phi     [-]        – tan(phi') of the sand, load AND resistance
    X[1] : hgwl        [m+NAP]    – groundwater level inside structure
    X[2] : q_surcharge [kN/m2]    – external traffic load
    X[3] : theta_R     [-]        – resistance model uncertainty
    """
    tan_phi, hgwl, q_surcharge, theta_R = X
    phi_deg = np.degrees(np.arctan(tan_phi))

    V   = _vertical_load(hgwl, q_surcharge, factored=False)
    F_H = _horizontal_load(phi_deg, hgwl, case=case, q_surcharge=q_surcharge)[0]
    R_H = theta_R * V * P.f_ds * tan_phi

    return [(R_H - F_H) / P.H]




def lsf_sliding_construction(X):
    """Sliding LSF for the construction case (no external water)."""
    return lsf_sliding(X, case="construction")


# %%
def lsf_rotation(X, case: str = "operational"):
    """
    X[0] : tan_phi [-]   – tangent of the backfill friction angle
    X[1] : hgwl  [m+NAP] – groundwater level inside structure

    Returns [Z] where Z < 0 means eccentricity exceeds B/6.
    """
    tan_phi = X[0]
    phi_deg = np.degrees(np.arctan(tan_phi))
    hgwl    = X[1]
    q_surcharge = X[2]

    Vd         = _vertical_load(hgwl, q_surcharge, factored=False)
    F_H, z_arm = _horizontal_load(phi_deg, hgwl, case=case, q_surcharge=q_surcharge)

    M_dest = F_H * z_arm                # destabilising moment [kNm/m]
    M_stab = Vd * (P.B / 2.0)           # stabilising moment [kNm/m]
    
    x_R    = (M_stab - M_dest) / Vd

    e     = abs(P.B / 2 - x_R)           # eccentricity [m]
    e_lim = P.B / 6                      # middle-third limit [m]

    Z = (e_lim - e) / (P.B / 2)
    return [Z]


def lsf_rotation_construction(X):
    """Rotation LSF for the construction case (no external water)."""
    return lsf_rotation(X, case="construction")

# %%
def unity_checks_external(phi_deg:  float = 32.5,
                           su_kPa:   float = 12.0,
                           hgwl_val: float = 0.10) -> dict:
    """
    Deterministic unity checks for external stability.
    Uses mean parameter values and calls the same helpers as the LSFs.

    Parameters
    ----------
    phi_deg  : characteristic friction angle [degrees]
    su_kPa   : characteristic undrained shear strength [kPa]
    hgwl_val : mean groundwater level [m+NAP]
    """
    # Design values via partial factors (CUR198, CC2)
    gamma_su  = 1.40
    gamma_phi = 1.25

    # Design friction angle: tan(phi_d) = tan(phi_k) / gamma_phi
    tan_phi_d = np.tan(np.radians(phi_deg)) / gamma_phi
    phi_d_deg = np.degrees(np.arctan(tan_phi_d))

    # Vertical load: unfavourable for bearing, favourable for sliding/rotation
    Vd_unfav = _vertical_load(hgwl_val, q_surcharge=P.q_det, factored=True)
    h1 = max(P.z_crest - hgwl_val, 0.0)
    h2 = max(hgwl_val - P.z_bed, 0.0)
    Vd_fav = (P.gamma_fill * h1 + P.gamma_fill_sat * h2) * P.B   # gamma_G,fav = 1.0

    # Shared derived quantities
    # f_d   = P.f_k / gamma_phi
    f_d   = P.f_ds * tan_phi_d
    su_d  = su_kPa / gamma_su
    e_lim = P.B / 6.0

    # ===========================================================================
    # OPERATIONAL CASE  (external lake water at h_winter opposes internal pressure)
    # ===========================================================================
    F_H_op, z_arm_op = _horizontal_load(phi_d_deg, hgwl_val, case="operational", q_surcharge=P.q_det)

    # --- Bearing capacity (operational only — Vd is maximum here) ---
    e_br       = abs(F_H_op * z_arm_op) / Vd_unfav
    A_prime_d  = max(P.B - 2.0 * e_br, 0.0)
    denom      = A_prime_d * su_d
    ratio      = min(F_H_op / denom, 1.0) if denom > 0 else 1.0
    eta_H_d    = 0.5 * (1.0 + np.sqrt(max(1.0 - ratio, 0.0)))
    R_d        = A_prime_d * P.N_c * su_d * eta_H_d
    # UC_br      = Vd / R_d if R_d > 0 else np.inf
    UC_br      = Vd_unfav / R_d if R_d > 0 else np.inf

    # --- Sliding (operational) ---
    # R_H_d_op   = Vd * f_d
    # UC_sl_op   = F_H_op / R_H_d_op if R_H_d_op > 0 else np.inf
    R_H_d_op   = Vd_fav * f_d
    UC_sl_op   = F_H_op / R_H_d_op if R_H_d_op > 0 else np.inf

    # --- Rotation (operational) ---
    # e_d_op     = abs(F_H_op * z_arm_op) / Vd
    # UC_rot_op  = e_d_op / e_lim if e_lim > 0 else np.inf
    e_d_op     = abs(F_H_op * z_arm_op) / Vd_fav
    UC_rot_op  = e_d_op / e_lim if e_lim > 0 else np.inf

    # ===========================================================================
    # CONSTRUCTION CASE  (no external water — cofferdam in place, no lake pressure)
    # ===========================================================================
    F_H_con, z_arm_con = _horizontal_load(phi_d_deg, hgwl_val, case="construction", q_surcharge=P.q_det)

    # --- Sliding (construction) ---
    # R_H_d_con  = Vd * f_d                  # same Vd, same friction coefficient
    # UC_sl_con  = F_H_con / R_H_d_con if R_H_d_con > 0 else np.inf
    R_H_d_con  = Vd_fav * f_d
    UC_sl_con  = F_H_con / R_H_d_con if R_H_d_con > 0 else np.inf

    # --- Rotation (construction) ---
    # e_d_con    = abs(F_H_con * z_arm_con) / Vd
    # UC_rot_con = e_d_con / e_lim if e_lim > 0 else np.inf
    e_d_con    = abs(F_H_con * z_arm_con) / Vd_fav
    UC_rot_con = e_d_con / e_lim if e_lim > 0 else np.inf

    return {
        "operational case": {
            "V_unfav [kN/m]" : round(Vd_unfav, 2),
            "V_fav [kN/m]"   : round(Vd_fav,   2),
            "F_H [kN/m]"  : round(F_H_op,    2),
            "z_arm [m]"   : round(z_arm_op,   3),
            "bearing" : {"UC": round(UC_br,     3), "passes": UC_br     <= 1.0},
            "sliding" : {"UC": round(UC_sl_op,  3), "passes": UC_sl_op  <= 1.0},
            "rotation": {"UC": round(UC_rot_op, 3), "passes": UC_rot_op <= 1.0},
        },
        "construction case": {
            "V_unfav [kN/m]" : round(Vd_unfav, 2), # identical — no water buoyancy difference
            "V_fav [kN/m]"   : round(Vd_fav,   2),  # identical — no water buoyancy difference
            "F_H [kN/m]"  : round(F_H_con,    2),  # larger — no opposing lake pressure
            "z_arm [m]"   : round(z_arm_con,   3),
            "bearing" : {"UC": "N/A",               "passes": True},   # not critical in construction
            "sliding" : {"UC": round(UC_sl_con,  3), "passes": UC_sl_con  <= 1.0},
            "rotation": {"UC": round(UC_rot_con, 3), "passes": UC_rot_con <= 1.0},
        },
    }


