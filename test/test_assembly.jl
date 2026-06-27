# Assembly tests (DESIGN §8.3 T16, §7.1 T20 allocation/scaling).

using PlasticityFEM
using PlasticityFEM.Assembly
using PlasticityFEM.Materials
using SparseArrays
using LinearAlgebra
using Test

@testset "sparsity pattern correctness" begin
    mesh = box_mesh(1.0, 1.0, 1.0, 2, 2, 2)
    sp = build_sparsity(mesh)
    ndof = 3 * mesh.nnodes
    @test size(sp.K) == (ndof, ndof)
    # map indices all resolved (no zeros)
    for e in 1:mesh.nelem
        @test all(sp.map[e] .> 0)
    end
end

@testset "T16 stiffness symmetry & SPD (elastic)" begin
    mesh = box_mesh(1.0, 1.0, 1.0, 3, 2, 2)
    mat = J2Material(E=210e3, ν=0.3, σy0=1e9)   # elastic
    model = Model(mesh, mat)
    U = zeros(3 * mesh.nnodes)
    st = model.state_trial
    K, R = assemble!(model.sparsity, mat, model.cache, U,
                     st.εp, st.β, st.ᾱ, st.σ, model.Rbuf)
    Kd = Matrix(K)
    @test norm(Kd - Kd') <= 1e-10 * norm(Kd)
    # after fixing enough DOFs to remove rigid body modes → SPD
    # clamp xmin face fully
    fixnodes = on_face(mesh, :xmin)
    free = trues(3 * mesh.nnodes)
    for n in fixnodes, c in 1:3
        free[3*(n-1)+c] = false
    end
    idx = findall(free)
    Kff = Kd[idx, idx]
    @test isposdef(Symmetric(Kff))
end

@testset "T20 assemble allocation bounded constant" begin
    function assemble_alloc(nx)
        mesh = box_mesh(1.0, 1.0, 1.0, nx, nx, nx)
        mat = J2Material(E=210e3, ν=0.3, σy0=250.0, Hiso=1000.0)
        model = Model(mesh, mat)
        U = zeros(3 * mesh.nnodes)
        st = model.state_trial
        # warmup
        assemble!(model.sparsity, mat, model.cache, U, st.εp, st.β, st.ᾱ, st.σ, model.Rbuf)
        return @allocated assemble!(model.sparsity, mat, model.cache, U,
                                    st.εp, st.β, st.ᾱ, st.σ, model.Rbuf)
    end
    a2 = assemble_alloc(2)
    a4 = assemble_alloc(4)
    # bounded by a small constant, independent of nelem (8× more elements)
    @test a2 < 4096
    @test a4 < 4096
    @test a4 <= a2 + 64    # essentially constant
end

@testset "T20 nnz scales ~linearly with nelem" begin
    nnzs = Int[]
    nelems = Int[]
    for nx in (2, 4, 6)
        mesh = box_mesh(1.0, 1.0, 1.0, nx, nx, nx)
        sp = build_sparsity(mesh)
        push!(nnzs, nnz(sp.K))
        push!(nelems, mesh.nelem)
    end
    # ratio nnz/nelem should be roughly bounded (not growing like nelem)
    r = nnzs ./ nelems
    @test maximum(r) / minimum(r) < 2.0
end
