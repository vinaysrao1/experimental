# Scaling-redesign tests (SCALING.md §6). Phase-gated:
#   Phase 1 — C1 (CG+AMG == direct), C3 (inexact Newton keeps the Newton tail).
#   Phase 2 — allocation gate, nnz/N flat, Int32 indices, no COO transient.
#   Phase 3 — 1-vs-4-thread result equality (race-free), threaded SpMV correctness.
#
# These tests use moderate meshes that run in the 15 GB sandbox; the 10M / 64 GB
# target is validated separately by extrapolation.

using PlasticityFEM
using PlasticityFEM.Materials
using PlasticityFEM.Assembly
using SparseArrays
using LinearAlgebra
using Test

# Standard uniaxial roller cube (mirrors test_solver.jl) with a chosen linsolve.
function _scaling_cube(nx; E=210e3, ν=0.3, σy0=250.0, Hiso=1000.0, Hkin=0.0,
                       εtarget=0.01)
    mesh = box_mesh(1.0, 1.0, 1.0, nx, nx, nx)
    mat = J2Material(E=E, ν=ν, σy0=σy0, Hiso=Hiso, Hkin=Hkin)
    model = Model(mesh, mat)
    fix!(model, on_face(mesh, :xmin), :x)
    fix!(model, on_face(mesh, :ymin), :y)
    fix!(model, on_face(mesh, :zmin), :z)
    prescribe!(model, on_face(mesh, :xmax), :x, εtarget)
    return model
end

@testset "C1 CG+AMG matches direct solve (displacements & stresses)" begin
    # Solve the SAME plastic problem with the direct (UMFPACK) and CG+AMG paths
    # on a ~16³ mesh and assert the answers agree to ~10·rtol (SCALING.md §6.1).
    rtol = 1e-8
    nx = 16

    md = _scaling_cube(nx)
    solve!(md; nsteps=12, tol=1e-9, linsolve=:direct)
    Ud = copy(md.U)
    σd = copy(gauss_stress(md))

    mc = _scaling_cube(nx)
    solve!(mc; nsteps=12, tol=1e-9, linsolve=:cg, amg=:sa)
    Uc = copy(mc.U)
    σc = copy(gauss_stress(mc))

    uerr = norm(Uc - Ud) / max(norm(Ud), eps())
    serr = norm(σc - σd) / max(norm(σd), eps())
    @info "C1 match" uerr serr
    @test uerr <= 10 * rtol
    @test serr <= 10 * rtol

    # Ruge–Stüben fallback also matches.
    mr = _scaling_cube(nx)
    solve!(mr; nsteps=12, tol=1e-9, linsolve=:cg, amg=:rs)
    @test norm(mr.U - Ud) / norm(Ud) <= 10 * rtol
end

@testset "C3 inexact Newton preserves the quadratic Newton tail (analog of T15)" begin
    # Same as T15 but through the CG+AMG inexact-Newton path: the last residuals
    # of a plastic step must still drop quadratically and the step must converge
    # in a small number of Newton iterations (SCALING.md §1.4, §6.1).
    model = _scaling_cube(2; σy0=250.0, Hiso=1000.0, Hkin=500.0, εtarget=0.01)
    res = solve!(model; nsteps=10, tol=1e-9, maxiter=15, linsolve=:cg)
    @test res.converged

    plastic_step = 0
    for (i, h) in enumerate(res.residuals)
        if length(h) >= 4
            plastic_step = i; break
        end
    end
    @test plastic_step > 0
    h = res.residuals[plastic_step]
    # quadratic tail: ‖R^{k+1}‖ ≤ C ‖R^k‖² for some tail iteration
    ok = false
    for k in 2:(length(h) - 1)
        if h[k] < 1.0 && h[k+1] <= 50.0 * h[k]^2 + 1e-12
            ok = true
        end
    end
    @info "C3 plastic residual tail" plastic_step h
    @test ok
    # small Newton iteration count preserved under inexact CG
    @test all(length(hh) <= 8 for hh in res.residuals)
end
