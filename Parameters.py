# %%
import numpy as np
import openturns as ot

# %%
Pf_system = 1 / 100

#failure probability budget
mechanism = np.array(['Overtopping', 'Toe_erosion', 'External_stability', 'Internal_stability', 'Local_failure', 'Other'])
budget = np.array([0.24,0.08,0.18,0.12,0.08,0.30])
N = np.array([2,2,3,3,1,1])

"""
Where:
    N     = length-effect factor (Table 5.4, given from OI2014/WOWK)
    n_sub = number of OR sub-mechanisms in the 2nd fault tree level
"""

FAILURE_TARGETS = {
    # mechanism       : (omega,  N,  n_sub)
    "overtopping"     : (0.24,   2,  1),   # overtopping, overflow
    "overflow"        : (0.24,   2,  1),
    "toe_stone"       : (0.08,   2,  2),   # top layer, width
    "toe_width"       : (0.08,   2,  2),
    "bearing"         : (0.18,   3,  3),   # bearing, sliding, overturning
    "sliding"         : (0.18,   3,  3),   # (macro-instability = NOT separate LSF)
    "overturning"     : (0.18,   3,  3),
    "rupture"         : (0.12,   3,  2),   # rupture, pull-out
    "pullout"         : (0.12,   3,  2),
}


# %%
def pf_target(mechanism: str) -> float:
    omega, N, n_sub = FAILURE_TARGETS[mechanism]
    return (omega * Pf_system) / (N * n_sub)

# %%
#input design variables 
z_bed   = -3.00                # lakebed level [m+NAP]
z_crest = 4.50                 # crest level [m+NAP] TODO
H       = z_crest - z_bed

B       = 10.00            # width [m] TODO
d_emb   = H / 20                # embedded depth [m] only for reinforcement TODO

D50_d   = 0.20                  # design median stone diameter of toe protection  [m] TODO
B_toe   = 10                     # design width of toe protection  [m] TODO

# Reinforcement properties — Geostrap5 synthetic strips TODO
S_v              = 0.50          # vertical strip spacing [m] TODO
S_h              = 0.70          # horizontal strip spacing [m] TODO
strip_width      = 0.05          # strip width [m]
f_dg             = strip_width / S_h    # coverage ratio [-]  (CUR198 eq. 3.63)

R_t_mean_strip   = 29.6          # mean tensile resistance per strip [kN/strip] TODO
R_t_cov          = 0.15          # CoV of tensile resistance [-]
R_t_mean         = R_t_mean_strip / S_h  # = 42.3 kN/m wall

Rt = ot.LogNormal()
Rt.setParameter(ot.LogNormalMuSigmaOverMu()([R_t_mean, R_t_cov, 0.0]))
Rt.setName("R_t")
Rt.setDescription(["R_t [kN/m]"])

# %%
# ---------------------------------------------------------------------------
#Material and soil properties
# ---------------------------------------------------------------------------
gamma_fill      = 17            # unit weight of backfill unsaturated [kN/m³]
gamma_fill_sat  = 19           # unit weight of backfill saturated [kN/m³]
gamma_w         = 9.81
gamma_fill_eff  = gamma_fill_sat - gamma_w # unit weight of water  [kN/m
gamma_clay      = 14.6          # unit weight of clay  [kN/m³] from table

N_c             = 5.14          # Terzaghi bearing capacity factor
f_ds             = 0.80          # Base interface efficiency --> f_k = 0.3 is now f_k = f_ds * tan phi * theta (model uncertainty)

mu_k_steel_shallow  = 1.50      # interaction coefficient steel, z ≤ 6 m  (CUR198)
m_k_steel_deep      = 0.64      # interaction coefficient steel, z > 6 m  (CUR198)

# %%
#External Loads
# q_surcharge     = 20            # Traffic load  [kN/m²]
q_det     = 20            # Traffic load  [kN/m²]

# %%
# Hydraulic input T = 833 years.
# These are deterministic design values for unity checks. The stochastic
# crest-level FORM distribution for X=[SWL, Hm0] is fitted in
# Lsf_overtopping.py from the Hydra-NL return-level tables.
h_d833          = 1.86          # design still water level  [m+NAP]
H_s833          = 1.32          # significant wave height HydraNL  [m]
T_m833          = 4.63          # spectral wave period [s]
q_allow         = 0.01          # allowable mean overtopping discharge  [m³/s/m]

#Hydraulic input for erosion 
h_winter          = -0.40       # design still water level  [m+NAP]
H_winter          = 1.27        # significant wave height HydraNL  [m] (chosen conservatively)
T_winter          = 5.40        # spectral wave period [s] (chosen conservatively)
gamma_b           = 0.55        # breaker parameter 


# %%
#Fixed Constants
g               = 9.81              # gravitational accelleration

Theta_cr        = 0.03              # Shields stability parameter
rho_w           = 1000              # density of fresh water [kg/m³]
rho_s           = 2650              # density of sand/gravel [kg/m³]
Delta = (rho_s - rho_w) / rho_w     # relative density of stone in fresh water

K_v             = 1.50              # coefficient for non-uniform flow
alpha_toe = np.radians(14.04)       # bed slope angle of toe (approx 1:4)  [rad]
z_toe           = -0.60
h_ref           = h_winter - z_bed  # reference scour depth h0 (= 0.70 * h0 → h_max)  [m] TODO
gamma_sf        = 1.1               # safety factor in width formula
cot_eps         = 2                 # cot(epsilon) scour hole upper slope
cot_beta        = 15                # cot(beta) potential sliding plane


# %%
# ---------------------------------------------------------------------------
#input stochastic variables
# ---------------------------------------------------------------------------
su = ot.LogNormal()                           # undrained shear strength [kPa]
su.setParameter(ot.LogNormalMuSigmaOverMu()([12.0, 0.25, 0.0]))
su.setName("s_u")
su.setDescription(["s_u [kPa]"])

# Resistance model uncertainty (JCSS 3.7.5.2, Table 3.7.5.1)
theta_R = ot.LogNormal()
theta_R.setParameter(ot.LogNormalMuSigmaOverMu()([1.0, 0.10, 0.0]))
theta_R.setName("theta_R"); theta_R.setDescription(["theta_R [-]"])

# Backfill friction is modelled through the friction factor tan(phi'), as
# recommended by JCSS.  LogNormalMuSigmaOverMu is parameterised with the
# physical mean and coefficient of variation sigma/mu.
phi_mean_deg = 32.5
tan_phi_mean = np.tan(np.radians(phi_mean_deg))
tan_phi_cov  = 0.10
tan_phi_std  = tan_phi_cov * tan_phi_mean

tan_phi = ot.LogNormal()
tan_phi.setParameter(
    ot.LogNormalMuSigmaOverMu()([tan_phi_mean, tan_phi_cov, 0.0])
)
tan_phi.setName("tan_phi")
tan_phi.setDescription(["tan(phi) [-]"])

# The armour-stone angle belongs to the toe-erosion model and is a different
# material variable.  Keep its existing model separate from the backfill.
phi_stone_normal = ot.Normal(40.0, 3.0)
phi_stone = ot.TruncatedDistribution(
    phi_stone_normal, 30.0, ot.TruncatedDistribution.LOWER
)
phi_stone.setName("phi_stone")
phi_stone.setDescription(["phi_stone [deg]"])

q_sur    = ot.Triangular(8, 12, 15)            # traffic load [kN/m²]
q_sur.setName("q_surcharge")
q_sur.setDescription(["q_surcharge [kN/m²]"])

fb    = ot.Triangular(0.60, 0.80, 1.00)       # bond factor
fb.setName("f_b")
fb.setDescription(["f_b [-]"])

hgwl  = ot.Triangular(-0.20, 0.10, 0.60)      # groundwater level [m+NAP]
hgwl.setName("h_gwl")
hgwl.setDescription(["h_gwl [m+NAP]"])

# %%
tan_phi.drawPDF()
phi_stone.drawPDF()
su.drawPDF()

# %%



