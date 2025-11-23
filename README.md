## Operator-Based Feedback Control for the Vlasov–Poisson System
This repository contains the implementation of the operator-based feedback control framework developed in the paper: "Dynamical feedback control with operator learning for the Vlasov-Poisson system" by Jingchehng Lu,
Li Wang, and Jeff Calder.

The project focuses on long-time stabilization of the nonlinear Vlasov–Poisson system using linear feedback operators, constructed via:

$\bullet$ Neural-operator-based feedback (from PDE-constrained optimization)

$\bullet$  Cancellation-based analytical feedback (with infinite-horizon decay guarantee)

The codes integrate semi-Lagrangian kinetic solvers, adjoint-state optimization, neural operator training, control testing and visualization. Experiments can be run through

$\bullet$ VPfeedback.py (neural operator-based control for 1D-1V, including training and testing)

$\bullet$ VPfeedback_cancellation_based.py (cancellation-based control for 1D-1V)

$\bullet$ VP4D.py (cancellation-based control for 2D-2V)
