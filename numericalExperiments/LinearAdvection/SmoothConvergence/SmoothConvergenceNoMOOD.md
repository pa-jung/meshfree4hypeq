### Numerical Experiment: Convergence Analysis for a Smooth Profile

A fundamental validation for any numerical scheme is to verify that it achieves its theoretical order of convergence for a problem with a smooth solution. This experiment is designed to measure the convergence rates of the various implemented methods by solving the linear advection equation on a uniform grid with a smooth initial condition.

#### Experimental Setup

The test problem is the one-dimensional linear advection equation, $u_t + a u_x = 0$, with an advection speed of $a=1.0$. The initial condition is a smooth Gaussian pulse, $u(x, 0) = \exp(-x^2)$, centered at the origin. The simulation is performed on a periodic domain, `xmin` = -5.0 to `xmax` = 5.0. The final time is set to `tmax` = 10.0, which corresponds to exactly one full period of the wave traversing the domain and returning to its initial position. This setup allows for a direct measurement of the accumulated numerical error over a long integration time.

The convergence study is performed by running a series of simulations for each numerical method, systematically increasing the number of particles `N` (from approximately $10^{1.5} \approx 32$ to $10^3 = 1000$). The time step $\Delta t$ is coupled to the spatial resolution via a fixed Courant-Friedrichs-Lewy number of `CFL` = 0.2 to ensure that the temporal error from the Runge-Kutta schemes does not dominate the spatial error. For each simulation, the L2 error of the numerical solution is computed at the final time. The results are presented in a log-log plot of the L2 error versus the number of particles `N`, where the slope of the line corresponds to the observed order of convergence.

#### Rationale for Method Selection

The purpose of this study is to confirm that the implemented spatial discretizations and their coupling with appropriate time integrators yield the expected order of accuracy. We compare:
1.  **First-Order Schemes:** `EulerUpwind` and the classical Lax-Friedrichs (`LLF`) method are expected to show first-order convergence.
2.  **Second-Order Schemes:** The classical Lax-Wendroff (`LW`) and meshfree methods combining a second-order Runge-Kutta timestepper with a second-order MUSCL reconstruction (`RK2MUSCL2`) are expected to be second-order accurate.
3.  **High-Order Schemes:** Combinations of high-order timesteppers (`RK4`, `ARS233`) and high-order MUSCL reconstructions (`MUSCL5`, which corresponds to a 4th-order reconstruction) are tested to verify if they achieve the higher design accuracy.

#### Observations

The figure presents the L2 error as a function of the number of particles `N` on a log-log scale. Dashed grey lines are included for reference, indicating slopes corresponding to first, second, third, and fourth-order convergence.

* **First-Order Methods:** The `EulerUpwind` (green dash-dot) and `LLF` (orange dash-dot) methods produce lines that are parallel to the reference line with a slope of -1. This confirms their expected first-order accuracy.
* **Second-Order Methods:** The `LW` (purple dotted), `RK2MUSCL2` (cyan dotted), `ARS233MUSCL2` (red solid), and `RK4MUSCL2` (yellow dashed) methods all produce lines that are parallel to the reference line with a slope of -2. This is the expected second-order convergence. Notably, pairing a fourth-order timestepper (`RK4`) with a second-order spatial scheme (`MUSCL2`) still results in a second-order accurate method, as the overall accuracy is limited by the lowest-order component, which in this case is the spatial discretization.
* **High-Order Methods:** The `ARS233MUSCL5` (blue dashed) and `RK4MUSCL5` (grey dashed) methods show a significantly steeper slope. Their convergence lines are parallel to the reference line with a slope of -4, demonstrating clear fourth-order convergence.

#### Analysis and Conclusion

The results of this convergence study successfully validate the implementation of the numerical schemes. Each method achieves its theoretically expected order of accuracy for a smooth problem on a uniform grid.

The key conclusion is that the **meshfree `MUSCL` implementation correctly provides higher-order spatial accuracy when requested**. Increasing the reconstruction order from linear (`MUSCL2`) to quartic (`MUSCL5`) and pairing it with a sufficiently high-order timestepper (`RK4` or `ARS233`) successfully increases the overall order of the scheme from second to fourth. This demonstrates that the framework is capable of high-order accuracy and that the MLS-based reconstruction of higher-order derivatives is working as intended under these ideal conditions.