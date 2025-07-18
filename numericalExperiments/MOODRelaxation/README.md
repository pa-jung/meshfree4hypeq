### Numerical Experiment: Analysis of MOOD Switching Parameters and Conservation

A critical aspect of the Multi-dimensional Optimal Order Detection (MOOD) framework is the logic that determines when to switch from the high-order base scheme to the robust, low-order fallback method. This experiment investigates how two different switching parameters—`delta_relax` and `switch_tol`—affect the mass conservation of the scheme when solving the inviscid Burgers' equation with a shock wave.

A key insight for this analysis is that both the high-order base scheme (`RK2MUSCL2`) and the low-order fallback scheme (`Upwind`) are mass-conservative when run on their own. Therefore, any observed mass loss is not a property of either individual scheme, but rather an artifact of the **non-conservative mixing** of fluxes that occurs when the MOOD framework applies different spatial discretizations to adjacent cells within the same time step.

#### Experimental Setup

The test problem is the inviscid Burgers' equation, $u_t + (\frac{1}{2}u^2)_x = 0$, with a shock wave initial condition. The base numerical method is `RK2MUSCL2`, a second-order Runge-Kutta timestepper combined with a second-order MUSCL spatial reconstruction. This high-order scheme is stabilized using different MOOD criteria. We vary the switching parameters and plot the resulting `relative_mass` of the solution at a fixed final time. An ideal result would be a relative mass of 1.0, indicating perfect conservation.

#### Plot 1 & 2: `delta_relax` with U1 and U2 MOOD Criteria

The first two plots show the effect of the `delta_relax` parameter on the final relative mass when using the `U1` and `U2` MOOD criteria. This parameter acts as a hard threshold for the MOOD detector.

* **Observations:**
    * In both plots, the behavior is that of a sharp, binary switch.
    * For the `U1` criterion (first plot), the relative mass is poor (around 0.90) for all `delta_relax` values below a critical threshold of approximately 1.0. At this threshold, the mass conservation abruptly jumps to nearly perfect (1.0).
    * For the `U2` criterion (second plot), the switch is even more pronounced. A `delta_relax` of 0.0 results in poor mass conservation, but any infinitesimally small positive value causes the relative mass to jump immediately to 1.0.

* **Analysis:**
    The `delta_relax` parameter controls a trade-off between **stability and mass conservation**. For small values, the MOOD criterion is active, frequently flagging cells at the shock front. This triggers the use of the fallback scheme in those cells, leading to non-conservative mixing and thus mass loss, but results in a stable, non-oscillatory solution. Once `delta_relax` is large enough, the MOOD check is always satisfied; the purely high-order (and conservative) `RK2MUSCL2` scheme is used everywhere. This perfectly conserves mass but would be unstable and produce oscillations at the shock. The sharpness of the transition shows that `delta_relax` acts as an on/off switch for this trade-off.

#### Plot 3: `switch_tol` with the Smooth Switching Method

The third plot shows the effect of the `switch_tol` parameter, used by the `RK2MUSCL2Smooth` method. This method is designed to provide a more gradual transition between the high-order and low-order schemes.

* **Observation:**
    Unlike `delta_relax`, the `switch_tol` parameter exhibits a smooth, gradual transition. For very small tolerances (e.g., $10^{-5}$), the relative mass is nearly perfect at 1.0. As the tolerance is increased, the mass conservation smoothly degrades, with a significant drop-off occurring for tolerances greater than $10^{-3}$.

* **Analysis:**
    This behavior highlights a trade-off between **accuracy (order) and mass conservation**. A small tolerance forces the scheme to be almost purely second-order and conservative, as very little mixing occurs. As the tolerance is increased, the scheme is allowed to switch to the first-order fallback method more liberally. This switch to a stable, non-oscillatory first-order state is desirable for stability, but the *process of mixing* the two conservative schemes is what causes the gradual loss of mass. The limit for a very large `switch_tol` is a stable, first-order method, but the path to get there involves this non-conservative blending.

#### Conclusion

This set of experiments reveals that the observed mass loss in the MOOD-stabilized schemes is a direct consequence of the **non-conservative mixing of fluxes** at the interface between high-order and low-order cells. The switching parameters `delta_relax` and `switch_tol` are fundamentally controls for this mixing process. The `delta_relax` parameter provides a sharp, binary control over the stability-vs-conservation trade-off, while `switch_tol` offers a more nuanced, continuous control over the accuracy-vs-conservation trade-off.