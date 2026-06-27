"""
    Elements

Hex8 trilinear hexahedron: shape functions, isoparametric Jacobian, B-matrix,
2×2×2 Gauss quadrature, geometry caching, and the allocation-free element
force/tangent kernel. See DESIGN.md §3.
"""
module Elements

using StaticArrays
using LinearAlgebra
using ..Materials: J2Material, return_map

export hex8_shape, hex8_dshape, jacobian, bmatrix, precompute_cache,
       element_force_tangent!, ElementCache, GAUSS_PTS, GAUSS_WTS,
       NODE_NAT

# Natural coordinates of the 8 nodes (DESIGN §3.1, VTK_HEXAHEDRON ordering)
const NODE_NAT = SMatrix{8,3,Float64,24}(
    # ξ                          η                          ζ
    -1, 1, 1, -1, -1, 1, 1, -1,
    -1, -1, 1, 1, -1, -1, 1, 1,
    -1, -1, -1, -1, 1, 1, 1, 1)

# 2×2×2 Gauss points & weights (DESIGN §3.5): coords ±1/√3, weight 1 each.
const _g = 1 / sqrt(3.0)
const GAUSS_PTS = SVector{8,SVector{3,Float64}}(
    SVector(-_g, -_g, -_g),
    SVector(_g, -_g, -_g),
    SVector(_g, _g, -_g),
    SVector(-_g, _g, -_g),
    SVector(-_g, -_g, _g),
    SVector(_g, -_g, _g),
    SVector(_g, _g, _g),
    SVector(-_g, _g, _g))
const GAUSS_WTS = SVector{8,Float64}(1, 1, 1, 1, 1, 1, 1, 1)

"""
    hex8_shape(ξ) -> SVector{8}

Trilinear shape functions N_a at natural coordinate ξ=(ξ,η,ζ) (DESIGN §3.2).
"""
@inline function hex8_shape(ξ::SVector{3,Float64})
    return SVector{8,Float64}(ntuple(a -> 0.125 *
        (1 + NODE_NAT[a, 1] * ξ[1]) *
        (1 + NODE_NAT[a, 2] * ξ[2]) *
        (1 + NODE_NAT[a, 3] * ξ[3]), 8))
end

"""
    hex8_dshape(ξ) -> SMatrix{8,3}

Shape-function derivatives ∂N_a/∂ξ_j w.r.t. natural coords (DESIGN §3.2).
"""
@inline function hex8_dshape(ξ::SVector{3,Float64})
    return SMatrix{8,3,Float64,24}(ntuple(8 * 3) do k
        a = (k - 1) % 8 + 1          # row (node)
        j = (k - 1) ÷ 8 + 1          # col (direction)
        ξa = NODE_NAT[a, 1]; ηa = NODE_NAT[a, 2]; ζa = NODE_NAT[a, 3]
        if j == 1
            0.125 * ξa * (1 + ηa * ξ[2]) * (1 + ζa * ξ[3])
        elseif j == 2
            0.125 * ηa * (1 + ξa * ξ[1]) * (1 + ζa * ξ[3])
        else
            0.125 * ζa * (1 + ξa * ξ[1]) * (1 + ηa * ξ[2])
        end
    end)
end

"""
    jacobian(Xe, dN) -> SMatrix{3,3}

Isoparametric Jacobian J_ij = Σ_a ∂N_a/∂ξ_i X_a,j = (dN)ᵀ Xe (DESIGN §3.3).
`Xe` is the 8×3 matrix of element node coordinates.
"""
@inline jacobian(Xe::SMatrix{8,3,Float64,24}, dN::SMatrix{8,3,Float64,24}) = dN' * Xe

"""
    bmatrix(dNdx) -> SMatrix{6,24}

Strain-displacement matrix B (Voigt [xx,yy,zz,xy,yz,zx], engineering shear)
from spatial shape-function gradients dN/dx (8×3) (DESIGN §3.4).
"""
@inline function bmatrix(dNdx::SMatrix{8,3,Float64,24})
    return SMatrix{6,24,Float64,144}(ntuple(6 * 24) do k
        r = (k - 1) % 6 + 1          # Voigt row
        c = (k - 1) ÷ 6 + 1          # local DOF column (1..24)
        a = (c - 1) ÷ 3 + 1          # node
        comp = (c - 1) % 3 + 1       # component x/y/z
        Nx = dNdx[a, 1]; Ny = dNdx[a, 2]; Nz = dNdx[a, 3]
        # B_a block (6×3) per DESIGN §3.4
        if r == 1
            comp == 1 ? Nx : 0.0
        elseif r == 2
            comp == 2 ? Ny : 0.0
        elseif r == 3
            comp == 3 ? Nz : 0.0
        elseif r == 4          # xy
            comp == 1 ? Ny : (comp == 2 ? Nx : 0.0)
        elseif r == 5          # yz
            comp == 2 ? Nz : (comp == 3 ? Ny : 0.0)
        else                   # zx
            comp == 1 ? Nz : (comp == 3 ? Nx : 0.0)
        end
    end)
end

"""
    ElementCache

Per-element cached geometry: B-matrices and detJ·w at every Gauss point
(DESIGN §4.4). Geometry is independent of U, so these are computed once and
reused every Newton iteration.
"""
struct ElementCache
    B::Vector{SVector{8,SMatrix{6,24,Float64,144}}}   # [elem] -> (B per gp)
    detJw::Vector{SVector{8,Float64}}                  # [elem] -> (detJ*w per gp)
end

"""
    precompute_cache(nodes, elements) -> ElementCache

Compute and cache B and detJ·w for every (element, gp). `nodes` is 3×nnodes,
`elements` is 8×nelem (DESIGN §3.6, §4.4).
"""
function precompute_cache(nodes::Matrix{Float64}, elements::Matrix{Int})
    nelem = size(elements, 2)
    Bvec = Vector{SVector{8,SMatrix{6,24,Float64,144}}}(undef, nelem)
    Jwvec = Vector{SVector{8,Float64}}(undef, nelem)
    for e in 1:nelem
        # element node coordinates as 8×3 SMatrix
        Xe = SMatrix{8,3,Float64,24}(ntuple(24) do k
            a = (k - 1) % 8 + 1
            j = (k - 1) ÷ 8 + 1
            nodes[j, elements[a, e]]
        end)
        # compute B and detJ·w together (one Jacobian per Gauss point)
        geo = ntuple(8) do g
            ξ = GAUSS_PTS[g]
            dN = hex8_dshape(ξ)
            J = jacobian(Xe, dN)
            detJ = det(J)
            @assert detJ > 0 "non-positive detJ=$detJ in element $e (check node ordering)"
            dNdx = dN * inv(J)
            (bmatrix(dNdx), detJ * GAUSS_WTS[g])
        end
        Bvec[e] = SVector{8,SMatrix{6,24,Float64,144}}(ntuple(g -> geo[g][1], 8))
        Jwvec[e] = SVector{8,Float64}(ntuple(g -> geo[g][2], 8))
    end
    return ElementCache(Bvec, Jwvec)
end

"""
    element_force_tangent!(mat, Bs, detJw, ue, εp, β, ᾱ, e, σout, commit=Val(false))
        -> (Fe, Ke)

Element internal force (SVector{24}) and consistent tangent (SMatrix{24,24})
by looping the 8 Gauss points (DESIGN §3.6). Allocation-free.

`εp`, `β` (6×ngp), `ᾱ` (ngp), `σout` (6×ngp) are the working SoA state arrays;
`e` is the element index (gp global index = (e-1)*8 + g). When `commit=Val(true)`
the updated per-GP plastic state is written back; otherwise only stresses are
recorded (for assembly). `commit` is a `Val` (not a keyword) so the branch is
resolved at compile time and the kernel stays allocation-free.
"""
@inline function element_force_tangent!(mat::J2Material,
                                        Bs::SVector{8,SMatrix{6,24,Float64,144}},
                                        detJw::SVector{8,Float64},
                                        ue::SVector{24,Float64},
                                        εp::Matrix{Float64},
                                        β::Matrix{Float64},
                                        ᾱ::Vector{Float64},
                                        e::Int,
                                        σout::Matrix{Float64},
                                        ::Val{COMMIT}=Val(false)) where {COMMIT}
    Fe = zero(SVector{24,Float64})
    Ke = zero(SMatrix{24,24,Float64,576})
    @inbounds for g in 1:8
        B = Bs[g]
        w = detJw[g]
        idx = (e - 1) * 8 + g
        ε = B * ue
        εp_n = SVector{6,Float64}(εp[1, idx], εp[2, idx], εp[3, idx],
                                  εp[4, idx], εp[5, idx], εp[6, idx])
        β_n = SVector{6,Float64}(β[1, idx], β[2, idx], β[3, idx],
                                 β[4, idx], β[5, idx], β[6, idx])
        ᾱ_n = ᾱ[idx]
        σ, εp_new, β_new, ᾱ_new, D = return_map(mat, ε, εp_n, β_n, ᾱ_n)
        Fe += (B' * σ) * w
        Ke += (B' * (D * B)) * w
        # record stress for output / postprocessing
        σout[1, idx] = σ[1]; σout[2, idx] = σ[2]; σout[3, idx] = σ[3]
        σout[4, idx] = σ[4]; σout[5, idx] = σ[5]; σout[6, idx] = σ[6]
        if COMMIT
            εp[1, idx] = εp_new[1]; εp[2, idx] = εp_new[2]; εp[3, idx] = εp_new[3]
            εp[4, idx] = εp_new[4]; εp[5, idx] = εp_new[5]; εp[6, idx] = εp_new[6]
            β[1, idx] = β_new[1]; β[2, idx] = β_new[2]; β[3, idx] = β_new[3]
            β[4, idx] = β_new[4]; β[5, idx] = β_new[5]; β[6, idx] = β_new[6]
            ᾱ[idx] = ᾱ_new
        end
    end
    return Fe, Ke
end

end # module
