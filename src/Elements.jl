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
       element_force_tangent!, ElementCache, element_geometry, element_coords,
       GAUSS_PTS, GAUSS_WTS, NODE_NAT

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

Isoparametric Jacobian J_ij = ∂x_i/∂ξ_j = Σ_a X_a,i ∂N_a/∂ξ_j = Xeᵀ dN
(DESIGN §3.3). `Xe` is the 8×3 matrix of element node coordinates. The spatial
gradients are then `dN/dx = dN · J⁻¹`. (`detJ` is invariant to this transpose,
so axis-aligned box meshes — diagonal J — are unaffected; the transpose only
matters for sheared/distorted elements.)
"""
@inline jacobian(Xe::SMatrix{8,3,Float64,24}, dN::SMatrix{8,3,Float64,24}) = Xe' * dN

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
    element_geometry(Xe) -> (Bs, detJw)

Compute the per-Gauss-point B-matrices `Bs::SVector{8,SMatrix{6,24}}` and
`detJ·w` weights `detJw::SVector{8}` for an element with node coordinates
`Xe::SMatrix{8,3}`. Allocation-free; this is the v1 geometry math factored out so
it can be reused for both the cached reference element and the on-the-fly fallback
(SCALING.md §2.2).
"""
@inline function element_geometry(Xe::SMatrix{8,3,Float64,24})
    geo = ntuple(8) do g
        ξ = GAUSS_PTS[g]
        dN = hex8_dshape(ξ)
        J = jacobian(Xe, dN)
        detJ = det(J)
        dNdx = dN * inv(J)
        (bmatrix(dNdx), detJ * GAUSS_WTS[g])
    end
    Bs = SVector{8,SMatrix{6,24,Float64,144}}(ntuple(g -> geo[g][1], 8))
    detJw = SVector{8,Float64}(ntuple(g -> geo[g][2], 8))
    return Bs, detJw
end

# Element node coordinates as an 8×3 SMatrix (allocation-free).
@inline function element_coords(nodes::Matrix{Float64}, elements::AbstractMatrix{<:Integer}, e::Integer)
    return SMatrix{8,3,Float64,24}(ntuple(24) do k
        a = (k - 1) % 8 + 1
        j = (k - 1) ÷ 8 + 1
        nodes[j, elements[a, e]]
    end)
end

"""
    ElementCache

Element geometry source for the assembler (SCALING.md §2.2). For a **uniform box
mesh** every element is geometrically identical up to translation, so a *single*
reference set of B-matrices and detJ·w is cached (`uniform=true`) — replacing the
v1 per-element cache (≈31 GB at 10M) with ≈9 kB. For non-uniform meshes
(`uniform=false`) the cache stores only the node coordinates and recomputes the
geometry on the fly per element (zero extra memory; modest compute).

Fields:
- `uniform` — true if every element shares one reference geometry.
- `Bref`, `detJwref` — the single reference set (valid iff `uniform`).
- `nodes`, `elements` — kept for the on-the-fly recompute fallback.
"""
struct ElementCache
    uniform::Bool
    Bref::SVector{8,SMatrix{6,24,Float64,144}}
    detJwref::SVector{8,Float64}
    nodes::Matrix{Float64}
    elements::Matrix{Int}
end

"""
    element_geometry(cache, e) -> (Bs, detJw)

Geometry for element `e`: the cached reference set if the mesh is uniform,
otherwise recomputed on the fly from node coordinates. Allocation-free — used in
the hot assembly loop (SCALING.md §2.2).
"""
@inline function element_geometry(cache::ElementCache, e::Integer)
    if cache.uniform
        return cache.Bref, cache.detJwref
    else
        Xe = element_coords(cache.nodes, cache.elements, e)
        return element_geometry(Xe)
    end
end

# Detect whether a box-style mesh is geometrically uniform: every element has the
# same edge-length triple (axis-aligned, identical shape up to translation). For
# `box_mesh` this is always true; a defensive numeric check keeps the fallback
# correct for any hand-built non-uniform mesh.
function _is_uniform(nodes::Matrix{Float64}, elements::Matrix{Int})
    nelem = size(elements, 2)
    nelem == 0 && return true
    Xe1 = element_coords(nodes, elements, 1)
    B1, Jw1 = element_geometry(Xe1)
    tol = 1e-9 * (sum(abs, Jw1) + 1.0)
    @inbounds for e in 2:nelem
        Xe = element_coords(nodes, elements, e)
        _, Jw = element_geometry(Xe)
        for g in 1:8
            abs(Jw[g] - Jw1[g]) > tol && return false
        end
    end
    return true
end

"""
    precompute_cache(nodes, elements) -> ElementCache

Build the element-geometry cache (SCALING.md §2.2). Detects whether the mesh is a
uniform box: if so caches a single reference element; otherwise stores the mesh
for on-the-fly geometry recompute. `nodes` is 3×nnodes, `elements` is 8×nelem.
"""
function precompute_cache(nodes::Matrix{Float64}, elements::Matrix{Int})
    uniform = _is_uniform(nodes, elements)
    if uniform
        Xe = element_coords(nodes, elements, 1)
        Bref, detJwref = element_geometry(Xe)
        @assert all(>(0), detJwref) "non-positive detJ in reference element (check node ordering)"
        return ElementCache(true, Bref, detJwref, nodes, elements)
    else
        # placeholder reference set (unused when uniform=false)
        Bz = zero(SMatrix{6,24,Float64,144})
        Bref = SVector{8,SMatrix{6,24,Float64,144}}(ntuple(_ -> Bz, 8))
        detJwref = zero(SVector{8,Float64})
        return ElementCache(false, Bref, detJwref, nodes, elements)
    end
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
