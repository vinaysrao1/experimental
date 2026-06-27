"""
    Solver

Newton–Raphson with load stepping and history commit. See DESIGN.md §1.5, §5.1.

Linear solve (SCALING.md §1): the default path is preconditioned Conjugate
Gradient (`Krylov.cg!`) with an Algebraic Multigrid preconditioner
(`AlgebraicMultigrid`), driven by an inexact-Newton (Eisenstat–Walker) forcing
term. A `:direct` (UMFPACK `\\`) path is kept for tiny/degenerate systems and as a
correctness reference. Preconditioner reuse policy: rebuild the AMG hierarchy
once per load step and reuse it across that step's Newton iterations (always
correctness-safe — CG always multiplies by the true current K).
"""
module Solver

using SparseArrays
using LinearAlgebra
using Krylov: CgWorkspace, cg!
using AlgebraicMultigrid: smoothed_aggregation, ruge_stuben, aspreconditioner, Jacobi
using ..MeshMod: GaussState
using ..BoundaryConditions: DirichletBC, NeumannBC, impose_dirichlet!, assemble_neumann!
using ..Assembly: assemble!, assemble_threaded!
using ..ModelMod: Model, reset!

export solve!, newton!, SolveResult, LinearSolveState

# Row-parallel SpMV for a SYMMETRIC CSC matrix (SCALING.md §3.2). Because K = Kᵀ,
# CSC column i holds exactly the entries of row i, so y[i] = Σ_{k in col i}
# nzval[k]·x[rowval[k]] lets each thread write a disjoint y[i] — race-free, no
# atomics. Used as the CG operator when threading is on; CG still applies the AMG
# preconditioner built from the underlying K. Falls back to a serial loop when
# single-threaded.
struct SymThreadedK{Tv,Ti}
    K::SparseMatrixCSC{Tv,Ti}
end
Base.eltype(::SymThreadedK{Tv}) where {Tv} = Tv
Base.size(A::SymThreadedK, d::Integer) = size(A.K, d)
Base.size(A::SymThreadedK) = size(A.K)
function LinearAlgebra.mul!(y::AbstractVector, A::SymThreadedK, x::AbstractVector)
    K = A.K; cp = K.colptr; rv = K.rowval; nz = K.nzval
    Threads.@threads for i in 1:size(K, 2)
        s = zero(eltype(y))
        @inbounds for k in cp[i]:(cp[i+1] - 1)
            s += nz[k] * x[rv[k]]
        end
        @inbounds y[i] = s
    end
    return y
end
Base.:*(A::SymThreadedK, x::AbstractVector) = mul!(similar(x, promote_type(eltype(A), eltype(x))), A, x)

"""
    SolveResult

Per-load-step iteration counts and residual histories (DESIGN §6.3). Used by
tests to assert Newton convergence rate (T15). `cg_iters` records, per load step,
the inner CG iteration count of every Newton iteration (empty for the `:direct`
path) — used by the scaling sweep to assert mesh-independent CG counts.
"""
struct SolveResult
    converged::Bool
    iters::Vector{Int}
    residuals::Vector{Vector{Float64}}
    cg_iters::Vector{Vector{Int}}
end

# Backwards-compatible constructor (older call sites / tests that omit cg_iters).
SolveResult(c, i, r) = SolveResult(c, i, r, Vector{Int}[])

"""
    LinearSolveState

Persistent workspace for the iterative linear solve (SCALING.md §1.2): the Krylov
CG workspace (so the inner solve is ~O(1) alloc, not O(N), per Newton iteration)
and the cached AMG preconditioner for the current load step. Created once for a
`solve!` call and reused across all load steps / Newton iterations.

Fields:
- `method` — `:cg` (CG+AMG, default) or `:direct` (UMFPACK `\\`).
- `amg` — `:sa` (smoothed aggregation, default) or `:rs` (Ruge–Stüben).
- `smoother` — `:gs` (Gauss–Seidel, AMG default) or `:jacobi` (thread-parallel).
- `ws` — Krylov `CgWorkspace` sized to ndof (allocated lazily).
- `Pl` — the cached AMG preconditioner (one V-cycle), rebuilt per load step.
"""
mutable struct LinearSolveState
    method::Symbol
    amg::Symbol
    smoother::Symbol
    cg_itmax::Int
    η_min::Float64
    η_max::Float64
    ew_γ::Float64
    ew_α::Float64
    ndof::Int
    threaded::Bool   # use row-parallel SpMV + 8-color threaded assembly
    ws::Any          # CgWorkspace{Float64,Float64,Vector{Float64}} | Nothing
    Pl::Any          # AlgebraicMultigrid.Preconditioner | Nothing
    cg_iters_hist::Vector{Int}   # rolling history for the refresh trigger
end

function LinearSolveState(ndof::Int; method::Symbol=:cg, amg::Symbol=:sa,
                          smoother::Symbol=:gs, cg_itmax::Int=200,
                          η_min::Float64=1e-8, η_max::Float64=0.1,
                          ew_γ::Float64=0.9, ew_α::Float64=1.5,
                          threaded::Bool=(Threads.nthreads() > 1))
    return LinearSolveState(method, amg, smoother, cg_itmax, η_min, η_max,
                            ew_γ, ew_α, ndof, threaded, nothing, nothing, Int[])
end

# Build an AMG hierarchy from K and wrap it as a (one V-cycle) preconditioner.
# Smoothed aggregation is the default for 3D vector elasticity; Ruge–Stüben is
# the documented fallback switch (SCALING.md §1.2). A Jacobi smoother (fully
# thread-parallel) is selectable; the AMG default is symmetric Gauss–Seidel.
function _build_amg(K::SparseMatrixCSC, amg::Symbol, smoother::Symbol)
    # AlgebraicMultigrid's classical Ruge–Stüben coarsening currently mishandles
    # Int32-indexed matrices (rs_direct_interpolation_pass2 assumes Int work
    # vectors), so build the RS hierarchy from an Int64-indexed copy. This is
    # safe: the preconditioner only maps residual→correction *vectors*, so its
    # internal index type is independent of the operator K that CG multiplies by.
    # Smoothed aggregation handles Int32 directly (no copy needed).
    Krs = amg === :rs ? SparseMatrixCSC{eltype(K),Int}(K) : K
    if smoother === :jacobi
        sm = Jacobi(2.0 / 3.0; iter=1)
        ml = amg === :rs ? ruge_stuben(Krs; presmoother=sm, postsmoother=sm) :
                           smoothed_aggregation(K; presmoother=sm, postsmoother=sm)
    else
        ml = amg === :rs ? ruge_stuben(Krs) : smoothed_aggregation(K)
    end
    return aspreconditioner(ml)
end

# Eisenstat–Walker choice 2 forcing term (SCALING.md §1.4). `rk`,`rkm1` are the
# current/previous Newton residual norms; `η_prev` the previous forcing term.
# Returns the CG rtol for this Newton iteration.
function _forcing_term(ls::LinearSolveState, it::Int, rk::Float64,
                       rkm1::Float64, η_prev::Float64)
    if it == 1 || rkm1 <= 0
        return ls.η_max
    end
    η = ls.ew_γ * (rk / rkm1)^ls.ew_α
    # safeguard against oversolving oscillation
    safe = ls.ew_γ * η_prev^ls.ew_α
    if safe > 0.1
        η = max(η, safe)
    end
    return clamp(η, ls.η_min, ls.η_max)
end

# Build the frozen BC objects from the model accumulators.
# Dirichlet is split into ramped (prescribed) and non-ramped (fixed) sets so a
# single `ramp` bool per DirichletBC suffices (DESIGN §4.5).
function _freeze_bcs(model::Model)
    # Deduplicate Dirichlet constraints per DOF (last assignment wins) so a DOF
    # that was constrained more than once does not appear twice in the imposed
    # system. Warn if two *conflicting* values are prescribed on the same DOF.
    last = Dict{Int,Tuple{Float64,Bool}}()   # dof => (value, ramp)
    @inbounds for i in eachindex(model.dir_dofs)
        d = model.dir_dofs[i]
        v = model.dir_vals[i]
        prev = get(last, d, nothing)
        if prev !== nothing && prev[1] != v
            @warn "conflicting Dirichlet values on dof $d ($(prev[1]) vs $v); using the last" maxlog=5
        end
        last[d] = (v, model.dir_ramp[i])
    end
    rd = Int[]; rv = Float64[]; fd = Int[]; fv = Float64[]
    for (d, (v, ramp)) in last
        if ramp
            push!(rd, d); push!(rv, v)
        else
            push!(fd, d); push!(fv, v)
        end
    end
    dir_ramp = DirichletBC(rd, rv, true)
    dir_fix = DirichletBC(fd, fv, false)
    neu = NeumannBC(copy(model.neu_dofs), copy(model.neu_vals))
    return dir_ramp, dir_fix, neu
end

# Solve K δU = b into δU. CG+AMG by default, with a refresh-on-stall fallback;
# `:direct` uses UMFPACK. Returns the number of CG iterations (0 for direct).
# `rebuild_pc` forces an AMG rebuild before solving (start of a load step).
function _linear_solve!(δU::Vector{Float64}, K::SparseMatrixCSC, b::Vector{Float64},
                        ls::LinearSolveState, rtol::Float64, rebuild_pc::Bool)
    if ls.method === :direct
        δU .= K \ b
        return 0
    end

    # (re)build AMG preconditioner per the reuse policy
    if rebuild_pc || ls.Pl === nothing
        ls.Pl = _build_amg(K, ls.amg, ls.smoother)
    end
    if ls.ws === nothing || ls.ws.n != length(b)
        ls.ws = CgWorkspace(length(b), length(b), Vector{Float64})
    end

    # CG operator: the row-parallel symmetric SpMV wrapper when threading is on
    # (K is symmetric after `impose_dirichlet!`), else the plain CSC matrix. The
    # AMG preconditioner is always built from the underlying sparse K.
    A = ls.threaded ? SymThreadedK(K) : K
    cg!(ls.ws, A, b; M=ls.Pl, ldiv=true, atol=0.0, rtol=rtol, itmax=ls.cg_itmax)
    stats = ls.ws.stats
    niter = stats.niter

    # Fallback 1 (SCALING.md §1.7): refresh the preconditioner from the current K
    # and retry once if CG failed to converge (stall / itmax hit).
    if !stats.solved
        ls.Pl = _build_amg(K, ls.amg, ls.smoother)
        cg!(ls.ws, A, b; M=ls.Pl, ldiv=true, atol=0.0, rtol=rtol, itmax=ls.cg_itmax)
        stats = ls.ws.stats
        niter += stats.niter
        if !stats.solved
            @warn "CG failed to converge after preconditioner refresh ($(stats.status)); falling back to direct solve" maxlog=5
            δU .= K \ b
            return niter
        end
    end
    copyto!(δU, ls.ws.x)
    return niter
end

# Assemble K and F_int, choosing the race-free 8-color threaded path when enabled
# (SCALING.md §3.1); otherwise the serial assembler. Same result either way.
@inline function _assemble_KR!(model::Model, U::Vector{Float64}, st::GaussState,
                               R::Vector{Float64}, threaded::Bool, commit::Bool)
    if threaded
        return assemble_threaded!(model.sparsity, model.material, model.cache, U,
                                  st.εp, st.β, st.ᾱ, st.σ, R; commit=commit)
    else
        return assemble!(model.sparsity, model.material, model.cache, U,
                         st.εp, st.β, st.ᾱ, st.σ, R; commit=commit)
    end
end

"""
    newton!(model, dir_ramp, dir_fix, neu, λ, Fext, ls; tol, maxiter, verbose)
        -> (converged, residual_history, cg_history)

One load step at load factor λ. Iterates K_T δU = −R to equilibrium using the
consistent tangent (quadratic convergence, DESIGN §1.5). The inner linear solve
is CG+AMG with an inexact-Newton (Eisenstat–Walker) forcing term via `ls`
(SCALING.md §1.4). Writes the trial per-GP state; the caller commits on
convergence. The AMG preconditioner is rebuilt once at the start of the step.
"""
function newton!(model::Model, dir_ramp::DirichletBC, dir_fix::DirichletBC,
                 neu::NeumannBC, λ::Float64, Fext::Vector{Float64},
                 ls::LinearSolveState;
                 tol::Float64=1e-8, maxiter::Int=25, verbose::Bool=false)
    st = model.state_trial
    R = model.Rbuf
    U = model.U

    # external force for this step
    fill!(Fext, 0.0)
    assemble_neumann!(Fext, neu, λ)

    res_hist = Float64[]
    cg_hist = Int[]
    converged = false
    # Reference scale for a *relative* convergence test (DESIGN §6.3). See the v1
    # rationale: set from the first-iteration residual + external load norm with a
    # floor of 1. UNCHANGED by Phase 1 (outer Newton test is preserved).
    ref = 1.0
    fext_norm = sqrt(sum(abs2, Fext))

    rkm1 = 0.0      # previous Newton residual norm (for Eisenstat–Walker)
    η_prev = ls.η_max
    first_step = true

    for it in 1:maxiter
        # reset trial state from committed before recomputing (path-dependent)
        copyto!(model.state_trial, model.state_committed)

        # assemble F_int (into R) and K (into sparsity.nzval)
        K, _ = _assemble_KR!(model, U, st, R, ls.threaded, false)
        # residual R = F_int − F_ext
        @inbounds @. R = R - Fext

        # impose Dirichlet symmetrically (modifies K, R): on free rows this
        # carries the known-column contributions; on constrained rows R holds
        # the constraint violation −δg = −(g − U[d]) (DESIGN §5).
        impose_dirichlet!(K, R, dir_ramp, λ, U)
        impose_dirichlet!(K, R, dir_fix, λ, U)

        # convergence: full residual of the imposed system (UNCHANGED from v1).
        rnorm = sqrt(sum(abs2, R))
        push!(res_hist, rnorm)
        if it == 1
            ref = max(rnorm, fext_norm, 1.0)
        end

        if rnorm <= tol * ref
            converged = true
            verbose && println("    iter $it  |R| = $rnorm  -> converged")
            break
        end

        # inexact-Newton forcing term = inner CG rtol (SCALING.md §1.4)
        η = _forcing_term(ls, it, rnorm, rkm1, η_prev)
        # rebuild the AMG preconditioner once at the start of the load step
        # (reuse policy SCALING.md §1.3); reuse it for the rest of the iterations.
        rebuild = first_step
        first_step = false
        # RHS b = −R (the convergence test above is already done, and R is fully
        # re-assembled next iteration, so negating it in place is safe & alloc-free)
        @inbounds @. R = -R
        niter = _linear_solve!(model.δU, K, R, ls, η, rebuild)
        push!(cg_hist, niter)
        verbose && println("    iter $it  |R| = $rnorm  η = $η  cg_iters = $niter")

        @inbounds @. U = U + model.δU
        rkm1 = rnorm
        η_prev = η
    end

    return converged, res_hist, cg_hist
end

"""
    solve!(model; nsteps=10, tol=1e-8, maxiter=25, verbose=false,
           linsolve=:cg, amg=:sa, smoother=:gs, cg_itmax=200) -> SolveResult

Load-stepped Newton driver (DESIGN §1.5, §6.3). Ramps λ = 1/N … 1, Newton-
iterates each step, and commits the per-GP state on convergence (plasticity is
path dependent).

Linear-solver keywords (SCALING.md §1):
- `linsolve` — `:cg` (PCG+AMG, default) or `:direct` (UMFPACK `\\`).
- `amg` — `:sa` smoothed aggregation (default) or `:rs` Ruge–Stüben.
- `smoother` — `:gs` Gauss–Seidel (AMG default) or `:jacobi` (thread-parallel).
- `cg_itmax` — inner CG iteration cap (default 200).
- `threaded` — 8-color threaded assembly + row-parallel SpMV (default: on when
  `Threads.nthreads() > 1`). Results are identical to the serial path.
"""
function solve!(model::Model; nsteps::Int=10, tol::Float64=1e-8,
                maxiter::Int=25, verbose::Bool=false,
                linsolve::Symbol=:cg, amg::Symbol=:sa, smoother::Symbol=:gs,
                cg_itmax::Int=200, threaded::Bool=(Threads.nthreads() > 1))
    reset!(model)   # idempotent: start from the undeformed, unhardened state
    dir_ramp, dir_fix, neu = _freeze_bcs(model)
    Fext = zeros(length(model.U))
    ls = LinearSolveState(length(model.U); method=linsolve, amg=amg,
                          smoother=smoother, cg_itmax=cg_itmax, threaded=threaded)

    iters = Int[]
    residuals = Vector{Float64}[]
    cg_iters = Vector{Int}[]
    allconv = true

    for n in 1:nsteps
        λ = n / nsteps
        verbose && println("Load step $n/$nsteps  (λ=$λ)")
        conv, hist, chist = newton!(model, dir_ramp, dir_fix, neu, λ, Fext, ls;
                                    tol=tol, maxiter=maxiter, verbose=verbose)
        push!(iters, length(hist))
        push!(residuals, hist)
        push!(cg_iters, chist)
        if !conv
            allconv = false
            @warn "load step $n did not converge in $maxiter iterations"
            break
        end
        # commit: re-run assembly once with commit=true to write GP state,
        # then copy trial → committed (DESIGN §9 commit semantics).
        _assemble_KR!(model, model.U, model.state_trial, model.Rbuf, ls.threaded, true)
        copyto!(model.state_committed, model.state_trial)
    end

    return SolveResult(allconv, iters, residuals, cg_iters)
end

end # module
