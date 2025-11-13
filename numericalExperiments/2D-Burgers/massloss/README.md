# Quantitative Analysis of Mass Conservation for Shock Problem

## Introduction

This document provides a quantitative analysis of mass conservation for the 2D Burgers' shock problem, complementing the previous visual analysis of 1D cuts. The experiment plots the **Relative Mass** over time, calculated as the numerical mass `M_num(t)` divided by the analytical mass `M_ana(t)`.

In these plots, a value of **1.0 (indicated by the red horizontal line) signifies perfect mass conservation**. Deviations from this line represent a loss (if < 1.0) or gain (if > 1.0) of mass.

## Experimental Setup

The setup is identical to the 100x100 2D Burgers' shock problem analyzed previously.
- **PDE**: 2D Burgers' (`burgers2d`)
- **Particles**: `Nx = 100`, `Ny = 100`
- **Grid**: Irregular (`randomness_factor = (0.2, 0.2)`)
- **Initial Condition**: Riemann Problem (`init_func = riemann`)
- **IC Parameters**: `(1.0, 0.0, (-3.0, -3.0), (1.0, 1.0))` (Step from 1 to 0)
- **Max Time (`tmax`)**: `10.0`

---

## Observation of Plots

### Direct (RK2) Solver Mass Conservation

![Direct (RK2) Solver Mass Error](./figures/burgers2d_mass_shock_direct.png)

- **`RK2Upwind`, `RK2MUSCL2`, `RK2MUSCL2Limiter`**: These schemes are **not perfectly conservative**. As expected for meshfree methods, they exhibit small errors, with their relative mass **oscillating symmetrically** around the 1.0 line. This deviation is minor and non-systematic.
- **`RK2MUSCL2MOOD`**: This scheme shows a clear and **systematic non-conservative behavior**. The relative mass rapidly drops and stabilizes around **0.965**, corresponding to a persistent **3.5% mass loss**.

### IMEX (ARS222) Solver Mass Conservation

![IMEX (ARS222) Solver Mass Error](./figures/burgers2d_mass_shock_imex.png)

- **`ARS222Upwind`, `ARS222MUSCL2`, `ARS222MUSCL2Limiter`**: Similar to the direct solver, these schemes show small, **symmetric oscillations** around the 1.0 line. Their error is not at machine precision but is not one-sided.
- **`ARS222MUSCL2MOOD`**: This scheme is also systematically non-conservative, but the mass loss is **significantly mitigated** by the IMEX solver. The relative mass drops and stabilizes around **0.994**, corresponding to a much smaller (but still present) **0.6% mass loss**.

---

## Analysis

These plots provide definitive, quantitative proof of the observations from the 1D cuts, with the new corrections incorporated.

1.  **Systematic vs. Oscillating Error**: This is the key finding. The `Upwind`, unlimited `MUSCL2`, and `MUSCL2Limiter` schemes all exhibit small errors that **oscillate around 1.0**. This behavior is typical for non-conservative (in the strict finite-volume sense) meshfree methods, and the symmetric nature means the error may not accumulate over long simulations.

2.  **`MOOD`'s Systematic Mass Loss**: In stark contrast, the `MOOD` scheme's error is **systematically negative** (always < 1.0). This one-sided error, even if small, **will accumulate over time**, leading to a continuous drift from the true solution.

3.  **IMEX Mitigates, But Doesn't Solve, the Flaw**: The `MOOD` scheme's mass loss is dramatically worse when paired with the direct `RK2` solver (~3.5% loss) compared to the IMEX `ARS222` solver (~.6% loss). This confirms the hypothesis that the IMEX solver's relaxation/stabilization mechanism helps *suppress* the non-conservative instability, but the underlying flaw in the `MOOD` formulation remains.