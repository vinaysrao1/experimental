"""
    Solver

Newton–Raphson with load stepping and history commit. See DESIGN.md §1.5, §5.1.
"""
module Solver

using SparseArrays
using LinearAlgebra
using ..MeshMod: GaussState
using ..BoundaryConditions: DirichletBC, NeumannBC, impose_dirichlet!, assemble_neumann!
using ..Assembly: assemble!
using ..ModelMod: Model, reset!

export solve!, newton!, SolveResult

"""
    SolveResult

Per-load-step iteration counts and residual histories (DESIGN §6.3). Used by
tests to assert Newton convergence rate (T15).
"""
struct SolveResult
    converged::Bool
    iters::Vector{Int}
    residuals::Vector{Vector{Float64}}
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

"""
    newton!(model, dir_ramp, dir_fix, neu, λ, Fext; tol, maxiter, verbose)
        -> (converged, residual_history)

One load step at load factor λ. Iterates K_T δU = −R to equilibrium using the
consistent tangent (quadratic convergence, DESIGN §1.5). Writes the trial
per-GP state; the caller commits on convergence.
"""
function newton!(model::Model, dir_ramp::DirichletBC, dir_fix::DirichletBC,
                 neu::NeumannBC, λ::Float64, Fext::Vector{Float64};
                 tol::Float64=1e-8, maxiter::Int=25, verbose::Bool=false)
    st = model.state_trial
    R = model.Rbuf
    U = model.U

    # external force for this step
    fill!(Fext, 0.0)
    assemble_neumann!(Fext, neu, λ)

    res_hist = Float64[]
    converged = false
    # Reference scale for a *relative* convergence test (DESIGN §6.3). The
    # residual carries force units, so an absolute tolerance is not scale
    # invariant — a force-controlled problem with large applied forces would
    # never reach an absolute 1e-8 (it bottoms out at the round-off floor of the
    # force magnitude). We set the reference from the first-iteration residual
    # (which captures the load-increment magnitude, including the prescribed-
    # displacement constraint contribution) and the external load norm, with a
    # floor of 1 so a near-zero step still uses an absolute test.
    ref = 1.0
    fext_norm = sqrt(sum(abs2, Fext))

    for it in 1:maxiter
        # reset trial state from committed before recomputing (path-dependent)
        copyto!(model.state_trial, model.state_committed)

        # assemble F_int (into R) and K (into sparsity.nzval)
        K, _ = assemble!(model.sparsity, model.material, model.cache, U,
                         st.εp, st.β, st.ᾱ, st.σ, R; commit=false)
        # residual R = F_int − F_ext
        @inbounds @. R = R - Fext

        # impose Dirichlet symmetrically (modifies K, R): on free rows this
        # carries the known-column contributions; on constrained rows R holds
        # the constraint violation −δg = −(g − U[d]) (DESIGN §5).
        impose_dirichlet!(K, R, dir_ramp, λ, U)
        impose_dirichlet!(K, R, dir_fix, λ, U)

        # convergence: full residual of the imposed system. This includes both
        # the free-dof equilibrium residual and any unsatisfied Dirichlet
        # constraint (so a step that must apply a prescribed displacement does
        # not falsely "converge" at U=0). Equivalent to ‖[R_free; g−U_bc]‖.
        rnorm = sqrt(sum(abs2, R))
        push!(res_hist, rnorm)
        if it == 1
            ref = max(rnorm, fext_norm, 1.0)
        end
        verbose && println("    iter $it  |R| = $rnorm  (tol = $(tol*ref))")

        if rnorm <= tol * ref
            converged = true
            break
        end

        δU = K \ (-R)
        @inbounds @. U = U + δU
    end

    return converged, res_hist
end

"""
    solve!(model; nsteps=10, tol=1e-8, maxiter=25, verbose=false) -> SolveResult

Load-stepped Newton driver (DESIGN §1.5, §6.3). Ramps λ = 1/N … 1, Newton-
iterates each step, and commits the per-GP state on convergence (plasticity is
path dependent).
"""
function solve!(model::Model; nsteps::Int=10, tol::Float64=1e-8,
                maxiter::Int=25, verbose::Bool=false)
    reset!(model)   # idempotent: start from the undeformed, unhardened state
    dir_ramp, dir_fix, neu = _freeze_bcs(model)
    Fext = zeros(length(model.U))

    iters = Int[]
    residuals = Vector{Float64}[]
    allconv = true

    for n in 1:nsteps
        λ = n / nsteps
        verbose && println("Load step $n/$nsteps  (λ=$λ)")
        conv, hist = newton!(model, dir_ramp, dir_fix, neu, λ, Fext;
                             tol=tol, maxiter=maxiter, verbose=verbose)
        push!(iters, length(hist))
        push!(residuals, hist)
        if !conv
            allconv = false
            @warn "load step $n did not converge in $maxiter iterations"
            break
        end
        # commit: re-run assembly once with commit=true to write GP state,
        # then copy trial → committed (DESIGN §9 commit semantics).
        assemble!(model.sparsity, model.material, model.cache, model.U,
                  model.state_trial.εp, model.state_trial.β, model.state_trial.ᾱ,
                  model.state_trial.σ, model.Rbuf; commit=true)
        copyto!(model.state_committed, model.state_trial)
    end

    return SolveResult(allconv, iters, residuals)
end

end # module
