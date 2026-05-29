# DG Credit-Rating Migration Solver

A 1D Discontinuous Galerkin (DG) solver for the regularized credit-rating migration PDE,
using a frozen-coefficient backward Euler scheme with SIPG diffusion and upwind drift.

## Target PDE (divergence form)

The solver targets the divergence-form equation:

    u_t - d/dx [ a_eps(x,t) u_x ] + d/dx [ beta_eps(x,t) u ] + gamma_w(x,t) u = f(x,t)

where `a_eps`, `beta_eps` (effective drift with correction), and `gamma_w` are regularized
coefficients depending on the smoothing parameter eps > 0.

**Do NOT discretize the nondivergence form** `-a_eps u_xx - b_eps u_x` directly.
**Do NOT start with Newton** before the frozen scheme and Picard both work.

## Implementation Phases

| Phase | What |
|-------|------|
| 0 | Project structure, YAML configs (this file) |
| 1 | Mesh, Legendre basis, quadrature, mass matrix |
| 2 | SIPG diffusion operator, heat equation MMS |
| 3 | Upwind drift, reaction term, sign tests (±β) |
| 4 | Regularized coefficients a_eps, beta_eps, gamma_w |
| 5 | Full MMS convergence tables for paper (p=1,2,3) |
| 6–7 | Financial reference runs, boundary tracker |
| 8 | Optional Picard/Newton nonlinear solve |

## First Milestone

Clean MMS convergence tables for p=1, 2, 3:
- Spatial EOC ≈ p+1
- Temporal EOC ≈ 1 (backward Euler)

## Config Files

- `config/mms_p1.yaml` — MMS runs with p=1
- `config/financial_base.yaml` — financial reference run with p=2
- `config/eps_sweep.yaml` — sweep over eps, enforcing h ≤ 0.5·eps

## Warnings

- Always use the **divergence-form PDE** with effective drift correction β_ε.
- Test upwind sign for **both β > 0 and β < 0** before touching the financial model.
- Do **not** omit γ_w from the drift form.
- Report **h/ε** in every financial run.
