# PlasticityFEM.jl

A small, fast, modular **3D finite element solver for small-strain
elastoplasticity** in Julia. It is built to make it *easy to build test models
and change loading / boundary conditions*, while keeping the numerical core
clean and allocation-free.

| | |
|---|---|
| **Element** | 8-node hexahedron (Hex8), trilinear, 2×2×2 Gauss |
| **Material** | J2 / von Mises, rate-independent, **combined linear isotropic + kinematic hardening** |
| **Integration** | radial-return mapping with the **consistent algorithmic tangent** (quadratic Newton) |
| **Global solve** | incremental Newton–Raphson with load stepping, sparse tangent |
| **Mesh** | structured box generator + predicate-based node/face selection |
| **BCs / loads** | Dirichlet (fixed & prescribed) and nodal/face forces, one line each |

The full specification — governing equations, the return-mapping algorithm, the
consistent tangent, Voigt conventions, and the verification plan — is in
[`docs/DESIGN.md`](docs/DESIGN.md).

---

## Quick start

```julia
using PlasticityFEM

# 1) Mesh — a 1×1×1 cube as a single Hex8 element (refine with box_mesh(...,n,n,n))
mesh = box_mesh(1.0, 1.0, 1.0, 1, 1, 1)

# 2) Material — steel-like, with linear isotropic hardening (consistent units: N, mm, MPa)
steel = J2Material(E = 210e3, ν = 0.3, σy0 = 250.0, Hiso = 1000.0)

# 3) Model
model = Model(mesh, steel)

# 4) Boundary conditions by face predicates — easy to read and change
fix!(model, on_face(mesh, :xmin), :x)    # roller on x = 0 face (fix x)
fix!(model, on_face(mesh, :ymin), :y)    # roller on y = 0 face (fix y)
fix!(model, on_face(mesh, :zmin), :z)    # roller on z = 0 face (fix z)

# 5) Loading — pull the x = L face to 1% nominal strain (displacement control)
prescribe!(model, on_face(mesh, :xmax), :x, 0.01)

# 6) Solve with load stepping
result = solve!(model; nsteps = 20, tol = 1e-8)

# 7) Postprocess
σ = gauss_stress(model)                  # 6 × ngp Voigt stresses [xx,yy,zz,xy,yz,zx]
println("σ_xx = ", σ[1, 1])              # ≈ 258.77 MPa (matches the analytical curve)
println("eq. plastic strain = ", maximum(equivalent_plastic_strain(model)))
```

This exact model is in [`examples/tension_cube.jl`](examples/tension_cube.jl); a
force-controlled cantilever (elastic vs partially-plastic) is in
[`examples/cantilever.jl`](examples/cantilever.jl).

---

## Building models, BCs, and loads

Everything is driven by **node selection predicates**, so changing the setup is a
one-line edit.

```julia
mesh = box_mesh(lx, ly, lz, nx, ny, nz)   # nx·ny·nz Hex8 elements in [0,lx]×[0,ly]×[0,lz]

# Select nodes…
on_face(mesh, :xmax)                       # nodes on a bounding-box face
                                           # :xmin/:xmax/:ymin/:ymax/:zmin/:zmax
select_nodes(mesh, (x,y,z) -> x ≈ 0 && z > 0.5)   # arbitrary coordinate predicate
```

| Call | Meaning |
|---|---|
| `fix!(model, nodes, comp=:all)` | homogeneous Dirichlet `u = 0` on `comp ∈ {:x,:y,:z,:all}` |
| `prescribe!(model, nodes, comp, value; ramp=true)` | inhomogeneous Dirichlet (prescribed displacement) |
| `load!(model, nodes, comp, value; distribute=false)` | nodal force; `distribute=true` splits a **total** face load across the nodes |
| `solve!(model; nsteps, tol=1e-8, maxiter=25, verbose=false)` | load-stepped Newton; returns a `SolveResult` (per-step iterations & residual histories) |
| `reset!(model)` | clear the solution and history (called automatically by `solve!`) |

**Postprocessing:** `nodal_displacements(model)` (3 × nnodes),
`gauss_stress(model)` (6 × ngp), `gauss_strain(model)` (6 × ngp total strain),
`equivalent_plastic_strain(model)` (ngp), `von_mises(σ)` (scalar from a Voigt
6-vector).

### Visualizing stress/strain distributions

Export a ParaView/VisIt file (dependency-free `.vtu` writer) and open it to see
the full 3D field distribution:

```julia
solve!(model; nsteps = 20)
write_vtu("result", model)          # -> result.vtu
```

The file carries per-node **Displacement** (use *Warp By Vector* for the deformed
shape) and per-element (Gauss-point-averaged) **Stress** and **Strain** (Voigt
tensors), **VonMises**, **MeanStress**, and **EqPlasticStrain** — color by any of
these in ParaView. Both `examples/` scripts write `.vtu` files; the plastic
cantilever shows the yielded zone near the clamp.

`solve!` is idempotent — it resets the model on entry, so you can change loads or
BCs and re-solve the same `model` object safely. Loading is ramped over `nsteps`
load steps because plasticity is path dependent.

---

## Installation

Requires Julia ≥ 1.10. The only dependency is
[StaticArrays](https://github.com/JuliaArrays/StaticArrays) (plus the
`SparseArrays`/`LinearAlgebra` standard libraries).

```bash
git clone <this-repo>
cd experimental
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Run an example or the test suite:

```bash
julia --project=. examples/tension_cube.jl
julia --project=. test/runtests.jl          # or: julia --project=. -e 'using Pkg; Pkg.test()'
```

---

## Conventions

- **Voigt ordering** (everywhere): `[xx, yy, zz, xy, yz, zx]`.
- **Strain** uses engineering shear (`γ_xy = 2 ε_xy`); **stress** uses physical
  shear components.
- **DOF numbering**: node-major, global DOF of (node `n`, component `c`) is
  `3(n−1)+c`, with `c = 1,2,3` ↦ `x,y,z`.
- **Units** are unit-agnostic; supply a consistent set (examples use N, mm, MPa).
- Sign convention: tension positive.

See [`docs/DESIGN.md` §9](docs/DESIGN.md) for the complete list.

---

## Project layout

```
src/
  PlasticityFEM.jl       top module + curated public API
  Materials.jl           J2Material, return_map (radial return + consistent tangent)
  Elements.jl            Hex8 shape functions, B-matrix, Gauss rule, cached geometry,
                         element internal force & tangent
  Mesh.jl                Mesh, box_mesh, select_nodes / on_face, dof, GaussState
  BoundaryConditions.jl  Dirichlet/Neumann structs, symmetric BC imposition
  Assembly.jl            cached sparsity pattern, scatter-into-nzval assembly
  Solver.jl              Newton–Raphson, load stepping, history commit
  Model.jl               Model + high-level fix!/prescribe!/load!/reset! + postprocessing
  Visualization.jl       dependency-free VTK (.vtu) export for ParaView/VisIt
examples/                runnable tension-cube and cantilever demos
test/                    unit, system, hard-validation, and performance tests
docs/DESIGN.md           full design & verification specification
```

---

## Performance

The package is written for high performance and validated to stay that way:

- **Type-stable, allocation-free hot kernels.** The constitutive `return_map`
  and the element force/tangent kernel allocate **0 bytes** per call (verified by
  tests). Element-level math uses `StaticArrays`.
- **Cached geometry & sparsity.** Element B-matrices and `detJ·w` are precomputed
  once; the global sparsity pattern is built once and assembly scatters directly
  into the `nzval` array, so `assemble!` allocation is **O(1)** (independent of
  the number of elements).
- **Consistent tangent ⇒ quadratic Newton.** Plastic steps converge in a handful
  of iterations, roughly independent of mesh size.

---

## Verification

The test suite implements the verification plan in `docs/DESIGN.md §8` and an
additional hard-validation/performance suite, including:

- consistent tangent checked against finite differences across many plastic
  states (guards quadratic convergence);
- uniaxial post-yield stress–strain vs the closed form
  `σ = σy0 + (E·Hiso/(E+Hiso))(ε − σy0/E)` (matched to machine precision);
- Bauschinger effect (`2σy0` elastic span on reversal) for kinematic hardening;
- single- and multi-element patch tests, rigid-body modes, plastic
  incompressibility, reaction/global-force balance;
- cantilever tip deflection vs Euler–Bernoulli beam theory;
- zero-allocation and O(1)-assembly performance gates.

---

## Finite strain (large deformation)

A finite-strain J2 element family is available alongside the small-strain path,
selected by one keyword at model construction:

```julia
model = Model(mesh, steel)                        # small strain (default)
model = Model(mesh, steel; element = :finite)      # finite strain, standard F
model = Model(mesh, steel; element = :finite_fbar) # finite strain, F-bar
```

Everything else is identical — `fix!`, `prescribe!`, `load!`, `solve!`,
`reset!`, `nodal_displacements`, `gauss_stress`, `equivalent_plastic_strain`,
`write_vtu` all work unchanged. The finite path uses Hencky (logarithmic)
hyperelasticity with an exponential-map multiplicative split `F = Fᵉ·Fᵖ`
(`det Fᵖ = 1` exactly), reusing the verified small-strain `return_map` in
log-strain space. The consistent tangent is the first-Piola/`F` form
`K = ∫ Gᵀ (∂P/∂F) G dV`, which contains both the material and geometric
(initial-stress) parts and is finite-difference-verified to ~1e-9.

- `:finite_fbar` applies the F-bar volumetric correction (de Souza Neto et al.
  1996) for the near-incompressible plastic limit, markedly relieving the
  volumetric locking of the trilinear Hex8.
- For finite-strain models `gauss_stress` reports the **Cauchy** stress
  `σ = τ/J`; `gauss_kirchhoff` returns the raw Kirchhoff stress `τ`. VTK output
  warps by the displacement vector to show the true deformed shape.
- Notes: with **kinematic** hardening (and for **F-bar**) the finite-strain
  algorithmic tangent is slightly non-symmetric (~1e-4 / ~1e-2 respectively),
  an inherent property of those formulations; use `linsolve=:direct` for those
  cases. Standard finite strain with isotropic/perfect plasticity yields a
  symmetric tangent. See `docs/FINITE_STRAIN.md`.

## Scope

Hex8 elements, J2 plasticity with linear (isotropic + kinematic) hardening,
quasi-static loading, in **both** small-strain and finite-strain kinematics.
Intentionally **not** included: contact/dynamics, nonlinear hardening laws,
other element types, anisotropic plasticity, thermomechanics. See
`docs/DESIGN.md §0, §10` and `docs/FINITE_STRAIN.md §0`.
