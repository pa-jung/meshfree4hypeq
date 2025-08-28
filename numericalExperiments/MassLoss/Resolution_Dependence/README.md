# Analysis of Grid Resolution on Mass Conservation and L2-Error

## Introduction

This final experiment analyzes the effect of grid resolution on the performance of the numerical schemes for a shock problem in the Burgers' equation. By systematically increasing the number of particles (`N`), we investigate whether the mass loss observed in previous tests is a systematic error that persists under grid refinement. Furthermore, by analyzing the L2-error, we directly assess how this non-conservative behavior impacts the overall convergence of the solution to the correct physical state.

## Experimental Setup

The simulation solves the 1D inviscid Burgers' equation with a box initial condition on a uniform, periodic grid. The final mass and L2-error are measured at `tmax = 10.0` for a range of particle numbers. The correct total mass for this initial condition is **1.0**.

### Shared Parameters
- **PDE**: Burgers' Equation
- **Initial Condition**: Step from 1.0 to 0 (`init_func: box`)
- **Domain**: `[-5, 5]` (periodic)
- **Final Time (`tmax`)**: 10.0
- **Grid**: Uniform
- **Particle Numbers (`N`)**: `[56, 100, 177, 316, 562, 1000]`
- **Methods**:
    - `ARS-MUSCL2-MOOD(U1)`: Meshfree MUSCL with ARS222 (IMEX) timestepper and U1 MOOD criterion.
    - `LW-MOOD(U1)`: Classical Lax-Wendroff with U1 MOOD criterion.
    - `RK2-MUSCL2-MOOD(U1)`: Meshfree MUSCL with RalstonRK2 timestepper and U1 MOOD criterion.
    - `RK2-MUSCL2-MOOD(U2)`: Meshfree MUSCL with RalstonRK2 timestepper and U2 MOOD criterion.
    - `RK2-MUSCL2`: Unlimited Meshfree MUSCL with RalstonRK2 timestepper.
    - `LW`: Classical Lax-Wendroff without MOOD.

---

## Observation of Plots

### Final Mass vs. Number of Particles

![Final Mass vs. N](./figures/burgers_shock_mass_Ndependence.svg)

- The `LW` and unlimited `RK2-MUSCL2` schemes demonstrate excellent mass conservation. Their final total mass remains very close to the correct value of **1.0** across all tested resolutions.
- All four `MOOD`-based schemes show a significant and nearly identical amount of mass loss, ending with a total mass of approximately 0.96.
- Critically, the amount of mass lost by the MOOD schemes is independent of the grid resolution; the final mass remains at a constant, incorrect value even as `N` increases.

### L2-Error vs. Number of Particles

![L2-Error vs. N](./figures/burgers_shock_L2error_Ndependence.png)

- The `LW` and `RK2-MUSCL2` schemes show a visible, albeit slow, decrease in L2-error as `N` increases, indicating that they are converging toward the correct solution. The error curves for these unlimited schemes are not perfectly smooth, exhibiting some non-monotonic behavior.
- All four `MOOD`-based schemes have a substantially higher L2-error, which remains nearly constant across the entire range of `N`. These schemes are not converging.

---

## Analysis

This experiment reveals a critical flaw in the non-conservative MOOD implementation for this problem: the mass loss is a zeroth-order error that prevents the solution from converging.

- **Systematic Mass Loss and Stalled Convergence**: The key finding is that the mass loss from the `MOOD` mechanism is a systematic error that does not diminish with grid refinement. Because the total mass is incorrect by a fixed amount regardless of `N` (as $\Delta x \to 0$), it represents a fundamental inconsistency in the scheme for this type of problem. The L2-error measures the difference between the numerical and exact solutions. Since the numerical solution has a systematically wrong total mass, this error cannot go to zero. The error becomes "stalled" or "saturated" by this constant mass error, which explains why the L2-error for all `MOOD`-based schemes fails to decrease with increasing `N`.

- **Convergence of More-Conservative Schemes**: In contrast, the `LW` and unlimited `RK2-MUSCL2` schemes are better at conserving mass. Because they converge to the correct total mass of 1.0, their L2-error is dominated by local discretization errors (like numerical oscillations or smearing near the shock) that *do* decrease with grid refinement. This allows their L2-error to trend downwards, demonstrating proper convergence.

- **Non-Monotonic Error in Unlimited Schemes**: The non-monotonic "wiggles" in the error curves for the `LW` and `RK2-MUSCL2` schemes are a well-known characteristic of applying unlimited, high-order methods to discontinuous problems. The L2-error is sensitive to the exact location and amplitude of the numerical oscillations relative to the true shock position. As `N` changes, the structure of these oscillations shifts, which can temporarily increase or decrease the L2-error while the overall trend of convergence continues.

- **MOOD Flavors (`U1` vs. `U2`)**: For this sharp shock problem, the performance of the `U1` and `U2` MOOD criteria are nearly identical. This is expected, as the `U2` criterion's main feature is the relaxation of the DMP for smooth extrema. At a sharp discontinuity, this relaxation is inactive, and both criteria behave like the simpler and more restrictive DMP check (`U1`).