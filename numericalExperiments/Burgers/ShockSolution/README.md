### Numerical Experiment: Scheme Performance for the Burgers' Equation Shock Wave

This experiment evaluates the performance of various high-order schemes and stabilization techniques when applied to the inviscid Burgers' equation, $u_t + (\frac{1}{2}u^2)_x = 0$, with a discontinuous initial condition that forms a shock wave. The primary goals are to assess each method's ability to capture the shock sharply and without oscillations, and to analyze their mass conservation properties on both uniform and irregular grids.

#### Experimental Setup

The initial condition is a Riemann problem that evolves into a single traveling shock wave. We investigate the performance on both uniform (`randomness_factor = 0.0`) and irregular (`randomness_factor > 0`) grids to test the methods' robustness. Two different shock strengths are considered: a large jump from a left state of $u_L=1.0$ to a right state of $u_R=0.0$, and a smaller jump from $u_L=1.0$ to $u_R=0.5$.

#### Observations and Analysis

##### Comparison on Uniform vs. Irregular Grids
![Solution profile on a uniform grid](./figures/burgers_shock_allmethods_uniform.svg)
![Solution profile on an irregular grid](./figures/burgers_shock_allmethods_irregular.svg)

A comparison of the solution profiles on uniform and irregular grids reveals the fundamental robustness of the stabilized meshfree methods for this nonlinear problem.
* **Observation:** The results on the irregular grid are remarkably similar to those on the uniform grid. All the stabilized methods (`MOOD`, `WENO`, and both `Superbee` and `VKLimiter`) successfully capture a sharp, non-oscillatory shock front.
* **Analysis:** This is a significant result. Unlike in the linear advection case where the `Superbee` limiter produced overshoots on irregular grids, for the nonlinear Burgers' equation it remains stable and non-oscillatory. The inherent dissipation of the nonlinear problem, combined with the limiter's compressive nature, is sufficient to control oscillations. This demonstrates that the implemented stabilization techniques are robust to grid perturbations for this problem.

##### Effect of Shock Strength on Mass Conservation
![Mass conservation for the small shock (1.0 -> 0.5)](./figures/burgers_shock_small_step.svg)

The mass conservation of the MOOD schemes is highly dependent on the properties of the shock.
* **Observation:** For the smaller shock (a jump from 1.0 to 0.5), the mass loss of the MOOD-stabilized schemes is minimal, with the relative mass staying very close to 1.0. However, for the stronger shock (a jump from 1.0 to 0.0), the mass loss is significantly more pronounced.
* **Analysis:** This confirms that the mass loss is not simply a function of the jump height ($\Delta u$), but is also sensitive to the absolute values of the states. The presence of a zero state ($u_R=0$) appears to make the problem more challenging for the MOOD detector, likely causing more frequent or wider application of the non-conservative fallback scheme compared to the case where both states are positive.

##### Direct vs. Relaxation Methods
The comparison between direct and relaxation-based schemes reveals a clear advantage for the latter in terms of conservation.
* **Observation:** In the mass conservation plots for the shock problem, the relaxation-based MOOD methods (e.g., `ARS233MUSCL2MOOD`) consistently exhibit less mass loss than their direct counterparts (e.g., `RK2MUSCL2MOOD`).
* **Analysis:** This reinforces a key finding: the relaxation framework improves the overall conservation of the stabilized scheme. The MOOD criteria are applied to the individual, linearly advected kinetic variables. Since these variables represent smoother components of the underlying physics, the MOOD detector is likely triggered less often or less severely than when applied directly to the highly nonlinear macroscopic solution. This results in less non-conservative mixing and better overall mass conservation.

#### Conclusion

This series of experiments on the Burgers' shock problem validates the robustness of the implemented stabilization techniques on both regular and irregular grids. The results highlight that the mass loss associated with the MOOD framework is a complex phenomenon tied to the non-conservative mixing of fluxes at the shock front. The severity of this mass loss depends on the properties of the shock itself, with stronger jumps to a zero state being particularly challenging. Finally, the superior conservation properties of the relaxation schemes suggest that applying stabilization to the simpler kinetic variables is a highly effective strategy for mitigating the conservation errors inherent in the MOOD switching process.