# %%
import numpy as np
import Parameters as P

# %%
#Depth of GWL below crest  [m]  (positive downwards)

def _gwl_depth(hgwl: float) -> float:
    """
    Convert groundwater level in [m+NAP] to depth below crest [m].
    Clamped to [0, H] — GWL cannot be above crest or below foundation.
    """
    return float(np.clip(P.z_crest - hgwl, 0.0, P.H))

# %%
#effective vertical stress at depth z_i [m below crest]

def _sigma_v(z_i: float, hgwl: float, q_surcharge: float) -> float:
    """
    Effective vertical stress at reinforcement layer depth z_i [kPa].
 
    Two-layer model:
      - dry zone   (0 to d_gwl):  gamma_fill
      - saturated  (d_gwl to z_i): gamma_fill_eff  (= gamma_fill - gamma_w)
 
    Parameters
    ----------
    z_i   : depth of layer below crest [m]  (positive downward)
    hgwl  : groundwater level [m+NAP]
    """
    d_gwl = _gwl_depth(hgwl)
 
    if z_i <= d_gwl:
        # layer is fully above GWL — dry unit weight throughout
        sigma_v = P.gamma_fill * z_i + q_surcharge
    else:
        # layer is below GWL — dry above, effective below
        sigma_v = (P.gamma_fill     * d_gwl
                 + P.gamma_fill_eff * (z_i - d_gwl)
                 + q_surcharge)
 
    return sigma_v

# %%
#Earth pressure coefficient

def _Ka(phi_deg: float) -> float:
    """Rankine active earth pressure coefficient."""
    phi_rad = np.radians(phi_deg)
    return (1 - np.sin(phi_rad)) / (1 + np.sin(phi_rad))


def _Kp(phi_deg: float) -> float:
    """Rankine passive earth pressure coefficient."""
    phi_rad = np.radians(phi_deg)
    return (1 + np.sin(phi_rad)) / (1 - np.sin(phi_rad))

# %%
# Anchorage length L_e,i  [m]
#
# The failure wedge (45° + phi/2 from horizontal) defines the boundary between
# the active zone and the anchorage zone. From CUR198 geometry:

# def _Le(z_i: float, phi_deg: float) -> float:
#     """
#     Effective anchorage length for reinforcement layer at depth z_i [m].
 
#     Parameters
#     ----------
#     z_i     : depth below crest [m]
#     phi_deg : friction angle [degrees]
#     """
#     phi_rad    = np.radians(phi_deg)
#     wedge_width = z_i * np.tan(np.radians(45.0) - phi_rad / 2.0)
#     # Le          = max(P.B - wedge_width,0) #should maybe change the max to diffrent length to 1m ? 
#     Le          = 0.5 * ((P.B - wedge_width) + np.sqrt((P.B - wedge_width)**2 + 1e-4))
#     return Le 

def _Le(z_i: float, phi_deg: float, L_i: float | None = None) -> float:
    """
    Anchorage length outside the active tie-back wedge.

    z_i : depth below crest [m]
    L_i : actual reinforcement length at layer i [m]
    """
    if L_i is None:
        L_i = P.B

    theta_v = np.radians(45.0 - phi_deg / 2.0)

    # Vertical distance from the layer to the toe
    height_above_toe = np.clip(P.H - z_i, 0.0, P.H)

    active_width = height_above_toe * np.tan(theta_v)
    raw_anchorage = L_i - active_width

    # Smooth approximation of max(raw_anchorage, 0) for FORM
    eps = 1e-6
    Le = 0.5 * (raw_anchorage + np.sqrt(raw_anchorage**2 + eps**2))
    return Le

# %%
# LSF for reinforcement RUPTURE at depth z_i
#   Z = Rt - T_i
#   X = [tan_phi [-],  Rt [kN/m],  hgwl [m+NAP]]

def make_lsf_rupture(z_i: float):
    """
    Returns lsf(X) -> [Z] for tensile rupture check at depth z_i.
 
    Parameters
    ----------
    z_i : depth of the reinforcement layer below the crest [m]
    """
    def lsf_rupture(X):
        """
        X[0] : tan_phi [-]   – tangent of the backfill friction angle
        X[1] : Rt   [kN/m]  – tensile resistance of reinforcement
        X[2] : hgwl [m+NAP] – groundwater level inside structure
 
        Returns [Z] where Z < 0 means tensile rupture.
        CUR198 eq. 3.43 / 3.44 — T_{h,i}=0, T_{v,i}=0 (no concentrated loads)
        """
        tan_phi = X[0]
        phi_deg = np.degrees(np.arctan(tan_phi))
        Rt_val  = X[1]
        hgwl    = X[2]
        q_surcharge = X[3]
 
        Ka       = _Ka(phi_deg)
        sigma_v  = _sigma_v(z_i, hgwl, q_surcharge)
        T_i      = Ka * sigma_v * P.S_v        # CUR198 eq. 3.44
 
        Z = Rt_val - T_i
        return [Z]
 
    lsf_rupture.__name__ = f"lsf_rupture_z{z_i:.2f}"
    return lsf_rupture
 

# %%
# LSF for PULL-OUT failure at depth z_i                       

#   Z = R_a,i - T_i
#   X = [tan_phi [-],  fb [-],  hgwl [m+NAP]]

def make_lsf_pullout(z_i: float):
    """
    Returns lsf(X) -> [Z] for pull-out check at depth z_i.
 
    Parameters
    ----------
    z_i : depth of the reinforcement layer below the crest [m]
    """
    def lsf_pullout(X):
        """
        X[0] : tan_phi [-]  – tangent of the backfill friction angle
        X[1] : fb   [-]     – bond coefficient for strips (from BBA/pullout test)
        X[2] : hgwl [m+NAP] – groundwater level inside structure

        Pull-out resistance for STRIP reinforcement (CUR198 eq. 3.63, c=0):
            R_a,i = mu* · f_dg · 2 · sigma'_v,i · L_e,i
            mu*   = fb · tan(phi)   [capped at 1.0]   (CUR198 eq. 3.62 / 4.5.3)

        Note: factor 2 applies for strips (both sides contribute, no group effect).
            For geogrids the factor would be 1 — do NOT use this LSF for geogrids.

        Returns [Z] where Z < 0 means pull-out failure.
        """
        tan_phi = X[0]
        phi_deg = np.degrees(np.arctan(tan_phi))
        fb_val  = X[1]
        hgwl    = X[2]
        q_surcharge = X[3]

        Ka      = _Ka(phi_deg)
        sigma_v = _sigma_v(z_i, hgwl, q_surcharge)
        Le_i    = _Le(z_i, phi_deg)

        # Interaction coefficient for strips (CUR198 eq. 3.62 / section 4.5.3)
        mu_star = min(fb_val * tan_phi, 1.0)

        # Pull-out resistance — strips, eq. 3.63 (factor 2, scaled by coverage ratio)
        R_a = mu_star * P.f_dg * 2.0 * sigma_v * Le_i

        # Tensile force (CUR198 eq. 3.44)
        T_i = Ka * sigma_v * P.S_v

        Z = R_a - T_i
        return [Z]
 
    lsf_pullout.__name__ = f"lsf_pullout_z{z_i:.2f}"
    return lsf_pullout

# %%
# Deterministic unity checks — all layers, both mechanisms, CUR198 design values
# ---------------------------------------------------------------------------
def unity_checks_internal(phi_deg:  float = 32.5,
                           fb_val:   float = 0.80,
                           Rt_k:     float = 100.0,
                           hgwl_val: float = 0.10) -> dict:
    """
    Return unity checks for all reinforcement layers using CUR198 design values.
 
    Partial factors from CUR198 (CC2):
      gamma_phi = 1.25   friction angle
      gamma_T   = 1.35   tensile force (unfavourable permanent load)
      gamma_R   = 1.00   tensile resistance
      gamma_mu  = 1.20   bond factor (pull-out)
 
    Parameters
    ----------
    phi_deg  : characteristic friction angle [degrees]
    fb_val   : characteristic bond factor [-]
    Rt_k     : characteristic tensile resistance [kN/m]
    hgwl_val : characteristic GWL [m+NAP]  (mean value)
    """
    gamma_phi = 1.25
    gamma_T   = 1.35
    gamma_R   = 1.00
    gamma_mu  = 1.20
 
    # Design friction angle: tan(phi_d) = tan(phi_k) / gamma_phi
    tan_phi_d  = np.tan(np.radians(phi_deg)) / gamma_phi
    phi_d_deg  = np.degrees(np.arctan(tan_phi_d))
    Ka_d       = _Ka(phi_d_deg)
 
    results    = {}
    governing_rupture_UC = 0.0
    governing_pullout_UC = 0.0
    governing_rupture_z  = None
    governing_pullout_z  = None
 
    # Reinforcement depths: S_v, 2*S_v, ..., H  (from top to bottom)
    layer_depths = np.arange(P.S_v, P.H + P.S_v / 2, P.S_v)
 
    for z_i in layer_depths:
        sigma_v   = _sigma_v(z_i, hgwl_val, q_surcharge=P.q_det)
 
        # --- Tensile force (design value) ---
        T_i       = Ka_d * sigma_v * P.S_v
        T_i_d     = gamma_T * T_i
 
        # --- Rupture UC ---
        Rt_d      = Rt_k / gamma_R
        UC_rt     = T_i_d / Rt_d if Rt_d > 0 else np.inf
 
        # --- Pull-out UC ---
        Le_i      = _Le(z_i, phi_d_deg)
        mu_star      = min(fb_val * tan_phi_d, 1.0)
        R_po_k    = mu_star * P.f_dg * 2.0 * sigma_v * Le_i
        R_po_d    = R_po_k / gamma_mu
        UC_po     = T_i_d / R_po_d if R_po_d > 0 else np.inf
 
        results[f"z={z_i:.2f}m"] = {
            "sigma_v [kPa]"  : round(sigma_v, 2),
            "T_i_d [kN/m]"   : round(T_i_d,  2),
            "Le_i [m]"       : round(Le_i,   3),
            "UC_rupture"     : round(UC_rt,   3),
            "UC_pullout"     : round(UC_po,   3),
            "rupture OK"     : UC_rt <= 1.0,
            "pullout OK"     : UC_po <= 1.0,
        }
 
        if UC_rt > governing_rupture_UC:
            governing_rupture_UC = UC_rt
            governing_rupture_z  = z_i
        if UC_po > governing_pullout_UC:
            governing_pullout_UC = UC_po
            governing_pullout_z  = z_i
 
    results["governing_rupture"] = {
        "depth [m]": governing_rupture_z,
        "UC"       : round(governing_rupture_UC, 3),
    }
    results["governing_pullout"] = {
        "depth [m]": governing_pullout_z,
        "UC"       : round(governing_pullout_UC, 3),
    }
 
    return results


