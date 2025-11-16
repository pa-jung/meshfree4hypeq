# Quantitative Analysis of Mass Loss in Linear Advection

## Introduction

This experiment investigates whether the mass loss associated with the `MOOD` mechanism, which was significant for the non-linear Burgers' equation, also occurs for the linear advection equation. By tracking the total mass for a discontinuous profile, we can compare the magnitude of the effect between the linear and non-linear cases. Furthermore, by using a smooth profile and "relaxing" the `MOOD` criterion, we can definitively diagnose the fallback mechanism as the source of the non-conservative behavior.

## Experimental Setup

The simulation solves the 1D linear advection equation, $\partial_t u + u = 0$, on a uniform, periodic domain. Two scenarios are considered.

### Scenario 1: Discontinuous Profile (Riemann Problem)
This test measures the mass loss for a propagating step function.
- **Initial Condition**: Step from 1.0 to 0 (`init_func: box`)
- **Domain**: `[-5, 5]`
- **Final Time (`tmax`)**: 5.0
- **Particles (`N`)**: 300
- **Methods**:
    - `ARS-MUSCL2-MOOD`: Meshfree MUSCL with `ARS222` (IMEX) timestepper.
    - `SS-MUSCL2-MOOD`: Meshfree MUSCL with `SimpleSplitting` timestepper.
    - `LW-MOOD`: A classical (non-meshfree) Lax-Wendroff scheme with MOOD.
    - `RK5-MOOD`: A 5th order scheme with MOOD.

### Scenario 2: Smooth Profile with MOOD Relaxation
This test uses a smooth Gaussian profile to show the effect of relaxing the `MOOD` criterion's sensitivity.
- **Initial Condition**: Gaussian (`init_func: gaussian`)
- **Domain**: `[-5, 5]`
- **Final Time (`tmax`)**: 5.0
- **Particles (`N`)**: 100
- **MOOD Settings**:
    - Standard: `delta_relax = 0.`
    - Relaxed: `delta_relax = 0.1`

---

## Observation of Plots

### Discontinuous Profile

![Mass vs. Time for Linear Advection Shock](./figures/LA_riemann_mass.png)

- All methods exhibit a very small, gradual loss of mass over time when advecting the discontinuity.
- The total mass loss is negligible for practical purposes (on the order of $10^{-4}$), and is orders of magnitude smaller than the loss observed for the Burgers' shock.
- The `ARS-MUSCL2-MOOD` scheme shows a slightly higher rate of mass loss compared to the other methods in this linear case.

### Smooth Profile with Standard vs. Relaxed MOOD

![Mass vs. Time with Standard MOOD](./figures/LA_smooth_mass_MOOD.png)
![Mass vs. Time with Relaxed MOOD](./figures/LA_smooth_mass_relaxed.png)

- With the standard, sensitive `MOOD` criterion, all methods show a small but steady loss of mass, even on the smooth Gaussian profile.
- When the `MOOD` criterion is relaxed (made less sensitive), the mass loss is almost entirely eliminated for all methods.
- Specifically, the `LW-MOOD` scheme becomes virtually perfectly conservative, and the meshfree schemes (`ARS` and `SS`) show only a tiny residual mass change, consistent with the baseline error of the discretization itself.

---

## Analysis

This set of experiments confirms that the `MOOD` fallback mechanism is the primary source of the mass loss, and that the effect, while present, is far less severe for linear problems than for non-linear ones.

- **Confirmation of MOOD-induced Mass Loss**: The comparison between the standard and relaxed `MOOD` criteria provides definitive proof that the fallback mechanism is the cause of the mass loss. By making the criterion less sensitive (`delta_relax = 0.1`), the dissipative first-order fallback is no longer triggered by the minor numerical ripples in the smooth Gaussian solution. This stops the mass loss, showing a direct causal link. The fact that the classical `LW-MOOD` scheme becomes perfectly conservative under the relaxed criterion isolates the effect to the `MOOD` logic itself, independent of the meshfree discretization.

- **Linear vs. Non-linear Effects**: The mass loss for the linear shock is present but dramatically smaller than for the Burgers' shock. This indicates that the severity of the non-conservative behavior is strongly linked to the non-linearity of the governing equation. For the Burgers' equation, the self-steepening nature of the shock creates a more challenging scenario for the numerical scheme, likely leading to more frequent or more aggressive `MOOD` interventions at the shock front. In the linear case, the discontinuity propagates without changing shape, resulting in a more predictable and less dissipative application of the fallback mechanism.

- **Performance of ARS Method in the Linear Case**: Interestingly, the `ARS` (IMEX) scheme, which was beneficial in the non-linear case, performs slightly worse here, showing the most mass loss (though the absolute difference is still negligible). This is likely an artifact of using a relaxation-based system of equations to approximate a simple scalar problem. The small errors introduced by the relaxation approximation may create minor numerical artifacts that trigger the sensitive `MOOD` criterion slightly more often than in a direct solve, leading to the marginally higher mass loss.