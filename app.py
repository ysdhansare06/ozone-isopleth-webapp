import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from numba import njit, prange

# ============================================================
# Streamlit page setup
# ============================================================
st.set_page_config(page_title="Ozone isopleths", layout="wide")
st.title("Ozone isopleths")
st.caption(
    "Simplified VOC/NOx ozone isopleths using Table 6.3 reactions from Seinfield and Pandis"
)

# ============================================================
# Physical constants
# ============================================================
KB = 1.380649e-23  # J/K

# ============================================================
# Helper conversions
# ============================================================
def air_number_density_cm3(T_K: float, P_Pa: float) -> float:
    """Air number density in molecules/cm^3 from ideal gas law."""
    return P_Pa / (KB * T_K) / 1.0e6


def ppb_to_molec_cm3(x_ppb: float, M_air: float) -> float:
    return x_ppb * 1.0e-9 * M_air


def ppt_to_molec_cm3(x_ppt: float, M_air: float) -> float:
    return x_ppt * 1.0e-12 * M_air


# ============================================================
# Sidebar controls
# ============================================================
st.sidebar.header("Simulation controls")

T_K = st.sidebar.number_input("Temperature (K)", 250.0, 330.0, 298.0, 1.0)
P_atm = st.sidebar.number_input("Pressure (atm)", 0.5, 1.5, 1.0, 0.01)
P_Pa = P_atm * 101325.0
M_air = air_number_density_cm3(T_K, P_Pa)

sim_hours = st.sidebar.number_input("Simulation duration (hours)", 1.0, 24.0, 10.0, 0.5)
dt_s = st.sidebar.number_input("Time step (s)", 0.5, 120.0, 10.0, 0.5)

st.sidebar.markdown("### Grid for isopleths (log-scale axes)")
RH_min = st.sidebar.number_input("RH min (ppb)", 0.1, 1000.0, 50.0, 0.1)
RH_max = st.sidebar.number_input("RH max (ppb)", 1.0, 5000.0, 1000.0, 1.0)
n_RH = st.sidebar.slider("Number of RH grid points", 10, 150, 55)

NOx_min = st.sidebar.number_input("NOx min (ppb)", 0.01, 100.0, 1.0, 0.01)
NOx_max = st.sidebar.number_input("NOx max (ppb)", 0.1, 500.0, 100.0, 0.1)
n_NOx = st.sidebar.slider("Number of NOx grid points", 10, 150, 55)

st.sidebar.markdown("### Chemistry inputs from Table 6.3")
P_HOx_ppt_s = st.sidebar.number_input("P_HOx (ppt s^-1)", 0.001, 10.0, 0.1, 0.001)
j_NO2 = st.sidebar.number_input("j_NO2 (s^-1)", 0.001, 0.1, 0.015, 0.001)

# Table 6.3 defaults at 298 K
k1_default = 26.3e-12
k2_default = 7.7e-12
k3_default = 8.1e-12
k4_default = 1.1e-11
k5_default = 2.9e-12
k6_default = 5.2e-12
k8_default = 1.9e-14

with st.sidebar.expander("Advanced: rate constants (cm^3 molecule^-1 s^-1)"):
    k1 = st.number_input("k1: RH + OH", value=k1_default, format="%.3e")
    k2 = st.number_input("k2: RO2 + NO", value=k2_default, format="%.3e")
    k3 = st.number_input("k3: HO2 + NO", value=k3_default, format="%.3e")
    k4 = st.number_input("k4: OH + NO2", value=k4_default, format="%.3e")
    k5 = st.number_input("k5: HO2 + HO2", value=k5_default, format="%.3e")
    k6 = st.number_input("k6: RO2 + HO2", value=k6_default, format="%.3e")
    k8 = st.number_input("k8: O3 + NO", value=k8_default, format="%.3e")

st.sidebar.markdown("### Plot settings")
contour_text = st.sidebar.text_input(
    "Contour levels (ppb, comma-separated)",
    "20,40,60,80,100,120,140,160,180,200,220,240"
)

show_filled = st.sidebar.checkbox("Show filled contours", value=False)

# ============================================================
# Parse contour levels
# ============================================================
def parse_levels(s: str):
    vals = []
    for part in s.split(","):
        part = part.strip()
        if part:
            vals.append(float(part))
    vals = sorted(set(vals))
    return np.array(vals, dtype=float)

levels = parse_levels(contour_text)
if len(levels) < 2:
    st.error("Please provide at least two increasing contour levels.")
    st.stop()

if RH_min <= 0 or RH_max <= 0 or NOx_min <= 0 or NOx_max <= 0:
    st.error("For log-scale axes, all axis limits must be positive.")
    st.stop()

if RH_min >= RH_max or NOx_min >= NOx_max:
    st.error("Minimum axis value must be smaller than maximum axis value.")
    st.stop()

# ============================================================
# Numba core
# ============================================================
@njit(cache=True)
def positive_quadratic_root(A: float, B: float, C: float) -> float:
    """
    Solve A x^2 + B x + C = 0 and return the nonnegative physical root.
    """
    if abs(A) < 1.0e-300:
        if abs(B) < 1.0e-300:
            return 0.0
        x = -C / B
        return x if x > 0.0 else 0.0

    disc = B * B - 4.0 * A * C
    if disc < 0.0:
        disc = 0.0

    root = (-B + math.sqrt(disc)) / (2.0 * A)
    if root < 0.0:
        root = 0.0
    return root


@njit(cache=True)
def radical_ss(RH, NO, NO2, P_HOx, k1, k2, k3, k4, k5, k6):
    """
    HOx steady state, with:
    - RO2 steady state from reactions 1 and 2
    - HO2 steady state from reactions 2 and 3
    """
    NO_eff = max(NO, 1.0e-30)

    # RO2 = (k1 RH OH)/(k2 NO)
    a = k1 * RH / (k2 * NO_eff)

    # HO2 = (k1 RH OH)/(k3 NO)
    b = k1 * RH / (k3 * NO_eff)

    # P_HOx = k4*OH*NO2 + 2*k5*HO2^2 + 2*k6*RO2*HO2
    #       = B*OH + A*OH^2
    A = 2.0 * k5 * b * b + 2.0 * k6 * a * b
    B = k4 * NO2

    OH = positive_quadratic_root(A, B, -P_HOx)
    RO2 = a * OH
    HO2 = b * OH

    return OH, HO2, RO2


@njit(cache=True)
def rhs(y, P_HOx, k1, k2, k3, k4, k5, k6, k8, jNO2):
    """
    ODE system for [RH], [NO], [NO2], [O3]
    """
    RH = max(y[0], 0.0)
    NO = max(y[1], 0.0)
    NO2 = max(y[2], 0.0)
    O3 = max(y[3], 0.0)

    OH, HO2, RO2 = radical_ss(RH, NO, NO2, P_HOx, k1, k2, k3, k4, k5, k6)

    r1 = k1 * RH * OH
    r2 = k2 * RO2 * NO
    r3 = k3 * HO2 * NO
    r4 = k4 * OH * NO2
    r8 = k8 * O3 * NO
    j7 = jNO2 * NO2

    dRH = -r1
    dNO = j7 - r2 - r3 - r8
    dNO2 = -j7 + r2 + r3 + r8 - r4
    dO3 = j7 - r8

    return np.array([dRH, dNO, dNO2, dO3], dtype=np.float64)


@njit(cache=True)
def rk4_step(y, dt, P_HOx, k1, k2, k3, k4, k5, k6, k8, jNO2):
    k_1 = rhs(y, P_HOx, k1, k2, k3, k4, k5, k6, k8, jNO2)
    k_2 = rhs(y + 0.5 * dt * k_1, P_HOx, k1, k2, k3, k4, k5, k6, k8, jNO2)
    k_3 = rhs(y + 0.5 * dt * k_2, P_HOx, k1, k2, k3, k4, k5, k6, k8, jNO2)
    k_4 = rhs(y + dt * k_3, P_HOx, k1, k2, k3, k4, k5, k6, k8, jNO2)

    y_new = y + (dt / 6.0) * (k_1 + 2.0 * k_2 + 2.0 * k_3 + k_4)

    for i in range(4):
        if y_new[i] < 0.0:
            y_new[i] = 0.0

    return y_new


@njit(cache=True)
def integrate_case(RH0_ppb, NOx0_ppb, M_air, t_end_s, dt_s,
                   P_HOx_ppt_s, jNO2,
                   k1, k2, k3, k4, k5, k6, k8):
    """
    Initial conditions from the text:
      [NO]/[NO2] = 2
      initial O3 from photostationary state
    Returns maximum O3 reached during the run, in ppb.
    """
    RH0 = RH0_ppb * 1.0e-9 * M_air
    NOx0 = NOx0_ppb * 1.0e-9 * M_air

    NO0 = (2.0 / 3.0) * NOx0
    NO20 = (1.0 / 3.0) * NOx0

    # Photostationary state:
    # jNO2 [NO2] = k8 [O3][NO]
    O30 = jNO2 * NO20 / max(k8 * NO0, 1.0e-30)

    P_HOx = P_HOx_ppt_s * 1.0e-12 * M_air

    y = np.array([RH0, NO0, NO20, O30], dtype=np.float64)
    max_O3 = y[3]

    n_steps = int(math.ceil(t_end_s / dt_s))
    for _ in range(n_steps):
        y = rk4_step(y, dt_s, P_HOx, k1, k2, k3, k4, k5, k6, k8, jNO2)
        if y[3] > max_O3:
            max_O3 = y[3]

    return max_O3 / M_air * 1.0e9


@njit(parallel=True, cache=True)
def compute_isopleths(RH_grid_ppb, NOx_grid_ppb, M_air, t_end_s, dt_s,
                      P_HOx_ppt_s, jNO2,
                      k1, k2, k3, k4, k5, k6, k8):
    ny = NOx_grid_ppb.size
    nx = RH_grid_ppb.size
    Z = np.empty((ny, nx), dtype=np.float64)

    for j in prange(ny):
        NOx0 = NOx_grid_ppb[j]
        for i in range(nx):
            RH0 = RH_grid_ppb[i]
            Z[j, i] = integrate_case(
                RH0, NOx0, M_air, t_end_s, dt_s,
                P_HOx_ppt_s, jNO2,
                k1, k2, k3, k4, k5, k6, k8
            )
    return Z


# ============================================================
# Build log-spaced grids
# ============================================================
RH_grid_ppb = np.logspace(np.log10(RH_min), np.log10(RH_max), n_RH)
NOx_grid_ppb = np.logspace(np.log10(NOx_min), np.log10(NOx_max), n_NOx)
t_end_s = sim_hours * 3600.0

# ============================================================
# Run button
# ============================================================
run = st.button("Generate Figure", type="primary")

if run:

    loading_message = st.info("Running simulation... please wait.")
    with st.spinner("Compiling Numba and running grid simulations..."):
        Z = compute_isopleths(
            RH_grid_ppb, NOx_grid_ppb, M_air, t_end_s, dt_s,
            P_HOx_ppt_s, j_NO2,
            k1, k2, k3, k4, k5, k6, k8
        )
    
    loading_message.empty()

    X, Y = np.meshgrid(RH_grid_ppb, NOx_grid_ppb)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    if show_filled:
        cf = ax.contourf(X, Y, Z, levels=levels, extend="both")
        plt.colorbar(cf, ax=ax, label="Maximum O$_3$ (ppb)")

    cs = ax.contour(X, Y, Z, levels=levels, colors="black", linewidths=1.0)
    ax.clabel(cs, inline=True, fontsize=10, fmt=lambda v: f"{int(round(v))}")

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("Initial RH (ppb)", fontsize=13)
    ax.set_ylabel(r"Initial NO$_x$ (ppb)", fontsize=13)
    ax.set_title(
        rf"Maximum O$_3$ isopleths over {sim_hours:g} h, "
        rf"$P_{{HOx}}={P_HOx_ppt_s:g}$ ppt s$^{{-1}}$",
        fontsize=14
    )

    ax.tick_params(direction="in", which="both", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    ax.grid(False)

    st.pyplot(fig, use_container_width=False)

    col1, col2, col3 = st.columns(3)
    col1.metric("Min max-O3 on grid (ppb)", f"{np.min(Z):.1f}")
    col2.metric("Mean max-O3 on grid (ppb)", f"{np.mean(Z):.1f}")
    col3.metric("Max max-O3 on grid (ppb)", f"{np.max(Z):.1f}")

    with st.expander("Raw computed field"):
        st.write("Rows correspond to NOx grid values, columns correspond to RH grid values.")
        st.dataframe(Z)

    with st.expander("Model assumptions used"):
        st.write(
            """
            - Simplified VOC/NOx chemistry from Table 6.3
            - HOx treated in steady state
            - RO2 in steady state from reactions 1 and 2
            - HO2 in steady state from reactions 2 and 3
            - Initial [NO]/[NO2] = 2
            - Initial O3 from the photostationary relation
            - RH, NO, NO2, and O3 are integrated for the selected simulation time
            - Plotted value = maximum O3 reached during the simulation
            """
        )

else:
    st.info("Set the parameters and click **Generate Figure**.")
