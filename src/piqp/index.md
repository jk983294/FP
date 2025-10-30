---
title: Home
layout: default
nav_order: 1
---

PIQP is a Proximal Interior Point Quadratic Programming solver, which can solve dense and sparse quadratic programs of the form

$$
\begin{aligned}
\min_{x} \quad & \frac{1}{2} x^\top P x + c^\top x \\
\text {s.t.}\quad & Ax=b, \\
& h_l \leq Gx \leq h_u, \\
& x_l \leq x \leq x_u,
\end{aligned}
$$

with primal decision variables $$x \in \mathbb{R}^n$$, matrices $$P\in \mathbb{S}_+^n$$, $$A \in \mathbb{R}^{p \times n}$$,  $$G \in \mathbb{R}^{m \times n}$$, and vectors $$c \in \mathbb{R}^n$$, $$b \in \mathbb{R}^p$$, $$h \in \mathbb{R}^m$$, $$x_{lb} \in \mathbb{R}^n$$, and $$x_{ub} \in \mathbb{R}^n$$. Combining an infeasible interior point method with the proximal method of multipliers, the algorithm can handle ill-conditioned convex QP problems without the need for linear independence of the constraints.

For more detailed technical results see our pre-print:

[**PIQP: A Proximal Interior-Point Quadratic Programming Solver**](https://arxiv.org/abs/2304.00290)