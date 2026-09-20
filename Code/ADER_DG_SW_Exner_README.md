# 1D low-Froude shallow-water--Exner ADER--DG implementation

## What is implemented

The PyTorch file `ader_dg_sw_exner_low_froude_pytorch.py` implements the formulation in `ADER_DG_Exner_Low_Froude.tex` with

\[
U=(\eta,q,b)^T,\qquad h=\eta-b,\qquad u=q/h,
\]

\[
U_t+F(U)_x=S,
\]

\[
F(U)=\begin{pmatrix}
q+q_b\\
qu+\frac{1}{2\mathrm{Fr}^2}(\eta^2-2\eta b)\\
q_b
\end{pmatrix},\qquad
S=\begin{pmatrix}0\\-\eta b_x/\mathrm{Fr}^2\\0\end{pmatrix},
\]

with the Grass law

\[
q_b=\xi A u^3,\qquad \xi=(1-\rho_0)^{-1}.
\]

The default polynomial degree is `k=2` (third-order local polynomial representation).

## Direct correspondence with the LaTeX notes

- `modal_to_spatial_taylor`: constructs the exact spatial coefficients
  \(\widetilde U(k_x,0)=\partial_x^{k_x}U/k_x!\) used in (10b).
- `series_mul`: implements the DT product convolution.
- `series_inverse`: implements the recursive DT coefficients of \(1/h\).
- `transformed_flux_source`: implements \(Q_1,Q_2,G_1,G_2,G_3,G_4\) and equation (12).
- `local_space_time_predictor`: applies recurrence (11) successively in increasing \(k_t\).
- `dg_one_stage_update`: performs the fully discrete one-stage DG correction corresponding to (7).
- `predictor_interface_states`: evaluates the time-dependent left/right traces of \(U_\tau\).
- `interface_prebalanced_flux`: computes the well-balanced interface flux.

The DG state itself is stored in a Legendre modal basis for numerical conditioning. This does **not** change the DT predictor: at each time step the Legendre polynomial is converted exactly to the centered Taylor coefficients required by the notes.

## Well-balanced interface treatment

The shallow-water interface flux uses hydrostatic reconstruction with a Rusanov homogeneous flux. Hydrostatic reconstruction is naturally formulated for the standard momentum flux

\[
q^2/h+\frac{h^2}{2\mathrm{Fr}^2}.
\]

The notes instead use the pre-balanced momentum flux

\[
q^2/h+\frac{\eta^2-2\eta b}{2\mathrm{Fr}^2}
= q^2/h+\frac{h^2}{2\mathrm{Fr}^2}-\frac{b^2}{2\mathrm{Fr}^2}.
\]

Therefore the code converts each hydrostatic cell-side momentum flux to the pre-balanced form by subtracting \(b_{\rm side}^2/(2\mathrm{Fr}^2)\). This is why a lake at rest with a discontinuous bed remains balanced to round-off.

This interface construction is a consistent extension of hydrostatic reconstruction to the pre-balanced `(eta,q,b)` form. The DT predictor and DG correction follow the supplied notes; the coupled Exner interface treatment itself should be regarded as the numerical closure chosen for this implementation, not as a claim that it is copied verbatim from the 2021 shallow-water-only ADER--DG paper.

## Stabilization

The Riemann benchmark contains discontinuities. Following the stabilization remark in the notes, the code includes:

1. a conservative troubled-cell detector and minmod limiter; troubled P2 cells are reduced locally to a limited P1 polynomial;
2. a positivity scaling limiter for \(h=\eta-b\), applied only to nonconstant modes and therefore preserving cell averages.

For the reported run the positivity limiter was never activated.

## Literature settings used

### ADER--DG discretization

Gang Li, Jiaojiao Li, Shouguo Qian, Jinmei Gao, *A well-balanced ADER discontinuous Galerkin method based on differential transformation procedure for shallow water equations*, Applied Mathematics and Computation 395 (2021), 125848. The paper uses `k=2` in its numerical examples and reports CFL `0.18` for the 1D computations.

### Moving-bed benchmark / FV reference

A. Siviglia, D. Vanzo, E. F. Toro, *A splitting scheme for the coupled Saint-Venant-Exner model*, Advances in Water Resources 159 (2022), 104062. Riemann Test 1 supplies

- left: `(h,q,b)=(2,0.5,3)`;
- right: `(h,q,b)=(2,4.34297,2.84751)`;
- Grass coefficient `Ag=0.01`, exponent `m=3`;
- 200 cells, domain length 30 m, final time 2 s.

The previously created `sw_exner_low_froude_pytorch.py` is run automatically and used as the robust first-order reference.

## Low-Froude scaling

The dimensional Riemann data are nondimensionalized before inserting `1/Fr^2`. With

\[
H_{ref}=2\;\mathrm m,\quad U_{ref}=0.25\;\mathrm{m/s},
\]

the global reference Froude number is

\[
\mathrm{Fr}=0.0564404551,\qquad 1/\mathrm{Fr}^2=313.92.
\]

A nuance: the right Riemann state has a physical local Froude number around 0.49, so this inherited Riemann benchmark is **not uniformly in the asymptotic low-Froude regime**, even though the governing equations are written with the correct low-Froude nondimensional scaling. A stricter low-Froude moving-bed benchmark is the smooth Exner test in Fernandez-Nieto et al., Journal of Scientific Computing 105 (2025), article 66, with `eta=10`, `q=10`, `Ag=0.1`, `m=3`, `rho0=0.2` and a smooth sediment hump.

## Validation obtained

Default P2 run:

- `N=200`, `k=2`, CFL `0.18`, final time `2 s`;
- 490 one-stage ADER--DG time steps;
- DT recurrence residual: `0` to floating-point precision;
- lake-at-rest residual: `(eta,q,b)=(0, 5.68e-12, 0)`;
- no positivity-scaling events;
- final cell-average depth remained positive: approximately `1.477 m <= h <= 2.00037 m`.

A particularly useful regression test is also included conceptually in the validation: with `k=0` and the FV CFL `0.9`, the pre-balanced DG update reduces to the reference FV method. The two final solutions agree at approximately machine precision:

- max eta difference: `3.55e-15 m`;
- max q difference: `1.60e-14 m^2/s`;
- max b difference: `1.33e-15 m`.

The P2 result is sharper than the first-order FV reference, as expected. The numerical comparison is saved in `ader_dg_vs_reference.png` and detailed norms are in `ader_dg_validation.txt`.

## Running

Place the ADER--DG file and the earlier FV reference file in the same directory, then run

```bash
python ader_dg_sw_exner_low_froude_pytorch.py
```

The reference solver is optional; if it is present, the code automatically computes the FV comparison.
