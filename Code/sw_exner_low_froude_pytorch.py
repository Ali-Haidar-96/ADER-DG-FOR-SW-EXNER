"""
PyTorch reference solver for the 1D low-Froude shallow-water--Exner system.

Unknowns
--------
    h     : water depth
    q=h*u : water discharge
    b     : moving bed
    eta=h+b : free-surface elevation

Dimensionless PDE (equivalent to the user's pre-balanced (eta,q,b) form)
----------------------------------------------------------------------------
    h_t + q_x = 0,

    q_t + [ q^2/h + h^2/(2 Fr^2) ]_x
        = -(h/Fr^2) b_x,

    b_t + (q_b)_x = 0,

with
    q_b = xi*A*u*|u|^(m-1),
    xi = 1/(1-rho0).

Since eta=h+b, this is equivalent to
    eta_t + (q+q_b)_x = 0,
which is the first equation in the user's (eta,q,b) formulation.

Numerical method
----------------
* finite volumes on a uniform 1D mesh;
* hydrostatic reconstruction (Audusse et al., 2004) for the shallow-water
  flux/source balance, with a local Lax--Friedrichs (Rusanov) flux;
* upwind Grass bed-load flux for the Exner equation;
* explicit CFL time step;
* float64 PyTorch tensors.

The hydrostatic reconstruction is included because it preserves a lake at rest
(q=0, eta=constant) to roundoff for a fixed discontinuous bed. A built-in test
checks this property before the benchmark run.

Literature benchmark
--------------------
Default data are the 1D Riemann Test 1 of:
    A. Siviglia, D. Vanzo, E.F. Toro,
    "A splitting scheme for the coupled Saint-Venant-Exner model",
    Advances in Water Resources 159 (2022), 104062.

Published dimensional data:
    left  : h=2 m, q=0.5 m^2/s, b=3 m
    right : h=2 m, q=4.34297 m^2/s, b=2.84751 m
    Grass : Ag=0.01 s^2/m, m=3
    mesh  : M=200 cells
    domain length L=30 m
    final time T=2 s
    CFL=0.9

Low-Froude nondimensionalization
--------------------------------
The literature benchmark is dimensional, whereas the user's equations contain
1/Fr^2. Therefore we NONDIMENSIONALIZE first; we do not simply replace g by
1/Fr^2 in dimensional equations.

Reference scales are chosen from the published left state:
    H_ref = 2 m,
    U_ref = q_L/h_L = 0.25 m/s,
    L_ref = 30 m,
so
    Fr = U_ref/sqrt(g H_ref) = 0.05644046.
This gives an actual low-Froude dimensionless system and maps the numerical
solution back to SI units for output.

Porosity convention
-------------------
Siviglia et al. write q_b = Ag*u^3. The user's notes write
q_b = xi*A*u^3, xi=1/(1-rho0). We use rho0=0.4 and set
A=(1-rho0)*Ag, so xi*A=Ag exactly. Thus the benchmark sediment flux is
unchanged while retaining the user's notation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math

import matplotlib.pyplot as plt
import torch


# =============================================================================
# GLOBAL PARAMETERS
# =============================================================================

DTYPE = torch.float64
DEVICE = torch.device("cpu")

# Physical gravity is used only to construct nondimensional Fr.
G_PHYS = 9.81  # m/s^2

# --- Published Riemann Test 1 data (Siviglia--Vanzo--Toro, 2022) ---
XMIN_PHYS = -15.0       # m; total length = 30 m, Riemann jump at x=0
XMAX_PHYS = 15.0        # m
N_ELEMS = 200
TFINAL_PHYS = 2.0       # s
CFL = 0.9

H_LEFT_PHYS = 2.0       # m
Q_LEFT_PHYS = 0.5       # m^2/s
B_LEFT_PHYS = 3.0       # m
H_RIGHT_PHYS = 2.0      # m
Q_RIGHT_PHYS = 4.34297  # m^2/s
B_RIGHT_PHYS = 2.84751  # m

M_GRASS = 3.0
AG_EFFECTIVE_PHYS = 0.01  # q_b = Ag*u^3, units s^2/m for m=3

# User's porosity notation. rho0=0.4 is a standard morphodynamic value.
RHO0 = 0.40
XI = 1.0 / (1.0 - RHO0)
A_GRASS_PHYS = (1.0 - RHO0) * AG_EFFECTIVE_PHYS

# --- Reference scales and global low-Froude parameter ---
H_REF = H_LEFT_PHYS
U_REF = Q_LEFT_PHYS / H_LEFT_PHYS
L_REF = XMAX_PHYS - XMIN_PHYS
T_REF = L_REF / U_REF
Q_REF = H_REF * U_REF

FR = U_REF / math.sqrt(G_PHYS * H_REF)
INV_FR2 = 1.0 / FR**2

# q_b scale is H_ref*U_ref. Therefore
# Ag_nd = Ag_dim * U_ref^(m-1) / H_ref.
AG_EFFECTIVE = AG_EFFECTIVE_PHYS * U_REF ** (M_GRASS - 1.0) / H_REF
A_GRASS = (1.0 - RHO0) * AG_EFFECTIVE  # XI*A_GRASS = AG_EFFECTIVE

H_FLOOR = 1.0e-12

OUTPUT_DIR = Path(__file__).resolve().parent
CSV_PATH = OUTPUT_DIR / "sw_exner_low_froude_solution.csv"
PT_PATH = OUTPUT_DIR / "sw_exner_low_froude_solution.pt"
ETA_PLOT_PATH = OUTPUT_DIR / "sw_exner_eta.png"
Q_PLOT_PATH = OUTPUT_DIR / "sw_exner_q.png"
B_PLOT_PATH = OUTPUT_DIR / "sw_exner_b.png"


@dataclass
class Solution:
    x_nd: torch.Tensor
    x_phys: torch.Tensor
    h0_nd: torch.Tensor
    q0_nd: torch.Tensor
    b0_nd: torch.Tensor
    h_nd: torch.Tensor
    q_nd: torch.Tensor
    b_nd: torch.Tensor
    eta_nd: torch.Tensor
    h_phys: torch.Tensor
    q_phys: torch.Tensor
    b_phys: torch.Tensor
    eta_phys: torch.Tensor
    time_nd: float
    time_phys: float
    nsteps: int
    dt_min_phys: float
    dt_max_phys: float
    dt_mean_phys: float


def grass_flux(u: torch.Tensor) -> torch.Tensor:
    """Dimensionless Grass bed-load flux in the user's porosity notation."""
    return XI * A_GRASS * u * torch.abs(u).pow(M_GRASS - 1.0)


def initial_condition(x_nd: torch.Tensor):
    """Published piecewise-constant Riemann Test 1 data, nondimensionalized."""
    left = x_nd < 0.0

    hL = H_LEFT_PHYS / H_REF
    qL = Q_LEFT_PHYS / Q_REF
    bL = B_LEFT_PHYS / H_REF
    hR = H_RIGHT_PHYS / H_REF
    qR = Q_RIGHT_PHYS / Q_REF
    bR = B_RIGHT_PHYS / H_REF

    h = torch.where(left, torch.full_like(x_nd, hL), torch.full_like(x_nd, hR))
    q = torch.where(left, torch.full_like(x_nd, qL), torch.full_like(x_nd, qR))
    b = torch.where(left, torch.full_like(x_nd, bL), torch.full_like(x_nd, bR))
    return h, q, b


def rusanov_hydro_flux(hL, qL, hR, qR):
    """Rusanov flux for the homogeneous nondimensional shallow-water system."""
    hLsafe = hL.clamp_min(H_FLOOR)
    hRsafe = hR.clamp_min(H_FLOOR)
    uL = qL / hLsafe
    uR = qR / hRsafe

    FL = torch.stack((qL, qL * uL + 0.5 * INV_FR2 * hL * hL), dim=0)
    FRv = torch.stack((qR, qR * uR + 0.5 * INV_FR2 * hR * hR), dim=0)

    aL = torch.abs(uL) + torch.sqrt(hLsafe) / FR
    aR = torch.abs(uR) + torch.sqrt(hRsafe) / FR
    a = torch.maximum(aL, aR)

    jump = torch.stack((hR - hL, qR - qL), dim=0)
    return 0.5 * (FL + FRv) - 0.5 * a * jump


def interface_fluxes(hL, qL, bL, hR, qR, bR):
    """
    Hydrostatic reconstruction at all interfaces.

    Returns
    -------
    F_for_left_cell  : shallow-water flux used by the cell to the left
    F_for_right_cell : shallow-water flux used by the cell to the right
    qb_interface     : Exner bed-load flux
    """
    # Audusse-type hydrostatic reconstruction.
    bstar = torch.maximum(bL, bR)
    hLstar = torch.clamp(hL + bL - bstar, min=0.0)
    hRstar = torch.clamp(hR + bR - bstar, min=0.0)

    # Preserve reconstructed velocity where the original cell is wet.
    ratioL = torch.where(hL > H_FLOOR, hLstar / hL, torch.zeros_like(hL))
    ratioR = torch.where(hR > H_FLOOR, hRstar / hR, torch.zeros_like(hR))
    qLstar = qL * ratioL
    qRstar = qR * ratioR

    Fstar = rusanov_hydro_flux(hLstar, qLstar, hRstar, qRstar)

    # Hydrostatic source corrections; these make lake-at-rest exactly balanced.
    corrL = torch.stack(
        (torch.zeros_like(hL), 0.5 * INV_FR2 * (hL * hL - hLstar * hLstar)),
        dim=0,
    )
    corrR = torch.stack(
        (torch.zeros_like(hR), 0.5 * INV_FR2 * (hR * hR - hRstar * hRstar)),
        dim=0,
    )
    F_left = Fstar + corrL
    F_right = Fstar + corrR

    # Bed-load flux. Use the direction of the interfacial water mass flux.
    uL = qL / hL.clamp_min(H_FLOOR)
    uR = qR / hR.clamp_min(H_FLOOR)
    qbL = grass_flux(uL)
    qbR = grass_flux(uR)
    qb_interface = torch.where(Fstar[0] >= 0.0, qbL, qbR)

    return F_left, F_right, qb_interface


def add_transmissive_ghosts(v: torch.Tensor) -> torch.Tensor:
    """One zero-gradient ghost cell at each boundary."""
    return torch.cat((v[:1], v, v[-1:]), dim=0)


def compute_dt(h: torch.Tensor, q: torch.Tensor, dx_nd: float, t_nd: float, tfinal_nd: float):
    """CFL condition used in the SVE benchmark, expressed nondimensionally."""
    hsafe = h.clamp_min(H_FLOOR)
    u = q / hsafe
    lam = torch.abs(u) + torch.sqrt(hsafe) / FR
    amax = float(lam.max().item())
    dt = CFL * dx_nd / max(amax, 1.0e-14)
    return min(dt, tfinal_nd - t_nd)


def lake_at_rest_check() -> tuple[float, float, float]:
    """Check well-balancedness over the same discontinuous bed."""
    xmin_nd = XMIN_PHYS / L_REF
    xmax_nd = XMAX_PHYS / L_REF
    dx_nd = (xmax_nd - xmin_nd) / N_ELEMS
    x = torch.linspace(
        xmin_nd + 0.5 * dx_nd,
        xmax_nd - 0.5 * dx_nd,
        N_ELEMS,
        dtype=DTYPE,
        device=DEVICE,
    )

    bL = B_LEFT_PHYS / H_REF
    bR = B_RIGHT_PHYS / H_REF
    b = torch.where(x < 0.0, torch.full_like(x, bL), torch.full_like(x, bR))
    eta = torch.full_like(x, 2.5)  # arbitrary constant nondimensional free surface
    h = eta - b
    q = torch.zeros_like(x)

    hg = add_transmissive_ghosts(h)
    qg = add_transmissive_ghosts(q)
    bg = add_transmissive_ghosts(b)

    FL, FR, qb = interface_fluxes(
        hg[:-1], qg[:-1], bg[:-1], hg[1:], qg[1:], bg[1:]
    )
    rhs_h = -(FL[0, 1:] - FR[0, :-1]) / dx_nd
    rhs_q = -(FL[1, 1:] - FR[1, :-1]) / dx_nd
    rhs_b = -(qb[1:] - qb[:-1]) / dx_nd

    return (
        float(rhs_h.abs().max().item()),
        float(rhs_q.abs().max().item()),
        float(rhs_b.abs().max().item()),
    )


def run_solver() -> Solution:
    xmin_nd = XMIN_PHYS / L_REF
    xmax_nd = XMAX_PHYS / L_REF
    dx_nd = (xmax_nd - xmin_nd) / N_ELEMS
    x_nd = torch.linspace(
        xmin_nd + 0.5 * dx_nd,
        xmax_nd - 0.5 * dx_nd,
        N_ELEMS,
        dtype=DTYPE,
        device=DEVICE,
    )

    h, q, b = initial_condition(x_nd)
    h0 = h.clone()
    q0 = q.clone()
    b0 = b.clone()

    tfinal_nd = TFINAL_PHYS / T_REF
    t_nd = 0.0
    nsteps = 0
    dts = []

    while t_nd < tfinal_nd - 1.0e-15:
        dt_nd = compute_dt(h, q, dx_nd, t_nd, tfinal_nd)

        hg = add_transmissive_ghosts(h)
        qg = add_transmissive_ghosts(q)
        bg = add_transmissive_ghosts(b)

        FL, FR, qb = interface_fluxes(
            hg[:-1], qg[:-1], bg[:-1], hg[1:], qg[1:], bg[1:]
        )

        # Each cell uses the left-cell corrected flux at its right interface and
        # the right-cell corrected flux at its left interface.
        h_new = h - (dt_nd / dx_nd) * (FL[0, 1:] - FR[0, :-1])
        q_new = q - (dt_nd / dx_nd) * (FL[1, 1:] - FR[1, :-1])
        b_new = b - (dt_nd / dx_nd) * (qb[1:] - qb[:-1])

        if not (torch.isfinite(h_new).all() and torch.isfinite(q_new).all() and torch.isfinite(b_new).all()):
            raise FloatingPointError(f"Non-finite state at step {nsteps}, t={t_nd*T_REF:.8e} s")
        if float(h_new.min().item()) <= 0.0:
            raise FloatingPointError(
                f"Non-positive depth at step {nsteps}, t={t_nd*T_REF:.8e} s; "
                f"min(h)={h_new.min().item()}"
            )

        h, q, b = h_new, q_new, b_new
        t_nd += dt_nd
        dts.append(dt_nd * T_REF)
        nsteps += 1

    eta = h + b
    x_phys = x_nd * L_REF
    h_phys = h * H_REF
    q_phys = q * Q_REF
    b_phys = b * H_REF
    eta_phys = eta * H_REF

    return Solution(
        x_nd=x_nd,
        x_phys=x_phys,
        h0_nd=h0,
        q0_nd=q0,
        b0_nd=b0,
        h_nd=h,
        q_nd=q,
        b_nd=b,
        eta_nd=eta,
        h_phys=h_phys,
        q_phys=q_phys,
        b_phys=b_phys,
        eta_phys=eta_phys,
        time_nd=t_nd,
        time_phys=t_nd * T_REF,
        nsteps=nsteps,
        dt_min_phys=min(dts),
        dt_max_phys=max(dts),
        dt_mean_phys=sum(dts) / len(dts),
    )


def save_outputs(sol: Solution) -> None:
    eta0_phys = (sol.h0_nd + sol.b0_nd) * H_REF
    q0_phys = sol.q0_nd * Q_REF
    b0_phys = sol.b0_nd * H_REF

    with CSV_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "x_m",
                "eta_initial_m",
                "eta_final_m",
                "q_initial_m2_s",
                "q_final_m2_s",
                "b_initial_m",
                "b_final_m",
                "h_final_m",
            ]
        )
        for row in zip(
            sol.x_phys.tolist(),
            eta0_phys.tolist(),
            sol.eta_phys.tolist(),
            q0_phys.tolist(),
            sol.q_phys.tolist(),
            b0_phys.tolist(),
            sol.b_phys.tolist(),
            sol.h_phys.tolist(),
        ):
            writer.writerow(row)

    torch.save(
        {
            "x_m": sol.x_phys,
            "eta_m": sol.eta_phys,
            "q_m2_s": sol.q_phys,
            "b_m": sol.b_phys,
            "h_m": sol.h_phys,
            "time_s": sol.time_phys,
            "Fr": FR,
            "1_over_Fr2": INV_FR2,
            "nsteps": sol.nsteps,
            "dt_min_s": sol.dt_min_phys,
            "dt_max_s": sol.dt_max_phys,
            "dt_mean_s": sol.dt_mean_phys,
        },
        PT_PATH,
    )

    x = sol.x_phys.cpu().numpy()

    plt.figure(figsize=(8, 4.8))
    plt.plot(x, eta0_phys.cpu().numpy(), linestyle="--", label="initial")
    plt.plot(x, sol.eta_phys.cpu().numpy(), label="final")
    plt.xlabel("x [m]")
    plt.ylabel(r"$\eta$ [m]")
    plt.title(r"Free surface $\eta=h+b$")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(ETA_PLOT_PATH, dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    plt.plot(x, q0_phys.cpu().numpy(), linestyle="--", label="initial")
    plt.plot(x, sol.q_phys.cpu().numpy(), label="final")
    plt.xlabel("x [m]")
    plt.ylabel(r"$q$ [m$^2$/s]")
    plt.title("Water discharge q")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(Q_PLOT_PATH, dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    plt.plot(x, b0_phys.cpu().numpy(), linestyle="--", label="initial")
    plt.plot(x, sol.b_phys.cpu().numpy(), label="final")
    plt.xlabel("x [m]")
    plt.ylabel(r"$b$ [m]")
    plt.title("Moving bed b")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(B_PLOT_PATH, dpi=180)
    plt.close()


def print_summary(sol: Solution, wb_residuals) -> None:
    print("=== PyTorch 1D low-Froude shallow-water--Exner reference solver ===")
    print(f"global Fr                 : {FR:.10f}")
    print(f"1/Fr^2                    : {INV_FR2:.10f}")
    print(f"mesh cells M              : {N_ELEMS}")
    print(f"physical dx               : {(XMAX_PHYS-XMIN_PHYS)/N_ELEMS:.8f} m")
    print(f"CFL                       : {CFL}")
    print(f"final time                : {sol.time_phys:.10f} s")
    print(f"number of time steps      : {sol.nsteps}")
    print(f"dt min / mean / max [s]   : {sol.dt_min_phys:.8e} / {sol.dt_mean_phys:.8e} / {sol.dt_max_phys:.8e}")
    print(f"lake-at-rest max RHS      : h={wb_residuals[0]:.3e}, q={wb_residuals[1]:.3e}, b={wb_residuals[2]:.3e}")
    print(f"final h range [m]         : {sol.h_phys.min().item():.8f} .. {sol.h_phys.max().item():.8f}")
    print(f"final eta range [m]       : {sol.eta_phys.min().item():.8f} .. {sol.eta_phys.max().item():.8f}")
    print(f"final q range [m^2/s]     : {sol.q_phys.min().item():.8f} .. {sol.q_phys.max().item():.8f}")
    print(f"final b range [m]         : {sol.b_phys.min().item():.8f} .. {sol.b_phys.max().item():.8f}")
    print(f"CSV                       : {CSV_PATH}")
    print(f"PyTorch data              : {PT_PATH}")
    print(f"eta plot                  : {ETA_PLOT_PATH}")
    print(f"q plot                    : {Q_PLOT_PATH}")
    print(f"b plot                    : {B_PLOT_PATH}")


if __name__ == "__main__":
    torch.set_default_dtype(DTYPE)
    wb = lake_at_rest_check()
    solution = run_solver()
    save_outputs(solution)
    print_summary(solution, wb)
