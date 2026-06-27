"""
    Assembly

Global sparsity pattern (cached COO → CSC mapping) and assembly of the global
tangent and residual by scattering element contributions into `nzval`.
See DESIGN.md §5.1, §7.
"""
module Assembly

using SparseArrays
using StaticArrays
using ..MeshMod: Mesh, dof
using ..Elements: ElementCache, element_force_tangent!
using ..Materials: J2Material

export SparsityPattern, build_sparsity, assemble!

"""
    SparsityPattern

Cached mapping from each element's 24×24 local stiffness entry to its position
in the global CSC `nzval` array. Built once; reused every Newton iteration so
assembly is a pure scatter into `nzval` (no `sparse()` rebuild) (DESIGN §7).

`K` is the prebuilt CSC skeleton (correct structure, zero values). `map[e]` is a
length-576 vector of nzval indices, column-major over the 24×24 local block.
"""
struct SparsityPattern
    K::SparseMatrixCSC{Float64,Int}
    edofs::Matrix{Int}          # 24 × nelem element DOF maps
    map::Vector{Vector{Int}}    # [elem] -> 576 nzval indices
end

"""
    element_dofs(elements, e) -> NTuple{24,Int}

Global DOFs of element e in local order (node-major: u1x,u1y,u1z,...,u8z).
"""
@inline function element_dofs(elements::Matrix{Int}, e::Int)
    return ntuple(24) do c
        a = (c - 1) ÷ 3 + 1
        comp = (c - 1) % 3 + 1
        dof(elements[a, e], comp)
    end
end

"""
    build_sparsity(mesh) -> SparsityPattern

Build the global CSC skeleton from element connectivity and cache, for every
element, the nzval index of each of its 576 local entries (DESIGN §7).
"""
function build_sparsity(mesh::Mesh)
    ndof = 3 * mesh.nnodes
    nelem = mesh.nelem
    edofs = Matrix{Int}(undef, 24, nelem)
    for e in 1:nelem
        ed = element_dofs(mesh.elements, e)
        @inbounds for i in 1:24
            edofs[i, e] = ed[i]
        end
    end

    # COO triplets (values 0) to define structure; sparse() coalesces duplicates.
    nnz_est = nelem * 576
    I = Vector{Int}(undef, nnz_est)
    J = Vector{Int}(undef, nnz_est)
    p = 0
    @inbounds for e in 1:nelem
        for c in 1:24, r in 1:24
            p += 1
            I[p] = edofs[r, e]
            J[p] = edofs[c, e]
        end
    end
    V = zeros(Float64, nnz_est)
    K = sparse(I, J, V, ndof, ndof)   # coalesced CSC skeleton

    # For each element local entry (r,c) find its position in K.nzval.
    colptr = K.colptr; rowval = K.rowval
    map = Vector{Vector{Int}}(undef, nelem)
    @inbounds for e in 1:nelem
        idxs = Vector{Int}(undef, 576)
        local_p = 0
        for c in 1:24
            gcol = edofs[c, e]
            colstart = colptr[gcol]; colend = colptr[gcol+1] - 1
            for r in 1:24
                local_p += 1
                grow = edofs[r, e]
                # binary search within column (rowval sorted ascending)
                lo = colstart; hi = colend; pos = 0
                while lo <= hi
                    mid = (lo + hi) >>> 1
                    rv = rowval[mid]
                    if rv == grow
                        pos = mid; break
                    elseif rv < grow
                        lo = mid + 1
                    else
                        hi = mid - 1
                    end
                end
                idxs[local_p] = pos
            end
        end
        map[e] = idxs
    end
    return SparsityPattern(K, edofs, map)
end

"""
    assemble!(sp, mat, cache, U, εp, β, ᾱ, σ, R; commit=false)
        -> (K, R)

Assemble global tangent (into `sp.K.nzval`) and internal-force residual `R`
from element contributions + per-GP return maps (DESIGN §5.1, §7). O(1) heap
allocation independent of nelem; element kernel is allocation-free.

`R` is filled with F_int (the caller subtracts F_ext to form the residual).
"""
function assemble!(sp::SparsityPattern, mat::J2Material, cache::ElementCache,
                   U::Vector{Float64},
                   εp::Matrix{Float64}, β::Matrix{Float64}, ᾱ::Vector{Float64},
                   σ::Matrix{Float64}, R::Vector{Float64};
                   commit::Bool=false)
    return _assemble!(sp, mat, cache, U, εp, β, ᾱ, σ, R, Val(commit))
end

# Parametric on the Val so element_force_tangent! specializes on COMMIT and the
# loop stays allocation-free (no dynamic dispatch / boxing of the SVector/SMatrix
# return). DESIGN §7.6.
function _assemble!(sp::SparsityPattern, mat::J2Material, cache::ElementCache,
                    U::Vector{Float64},
                    εp::Matrix{Float64}, β::Matrix{Float64}, ᾱ::Vector{Float64},
                    σ::Matrix{Float64}, R::Vector{Float64},
                    commit::Val{COMMIT}) where {COMMIT}
    nzval = sp.K.nzval
    fill!(nzval, 0.0)
    fill!(R, 0.0)
    nelem = size(sp.edofs, 2)
    @inbounds for e in 1:nelem
        # gather element displacement (Val(24) so ntuple unrolls allocation-free)
        ue = SVector{24,Float64}(ntuple(i -> U[sp.edofs[i, e]], Val(24)))
        Fe, Ke = element_force_tangent!(mat, cache.B[e], cache.detJw[e], ue,
                                        εp, β, ᾱ, e, σ, commit)
        # scatter F_int
        for i in 1:24
            R[sp.edofs[i, e]] += Fe[i]
        end
        # scatter K into nzval via cached map (column-major over 24×24)
        idxs = sp.map[e]
        lp = 0
        for c in 1:24, r in 1:24
            lp += 1
            nzval[idxs[lp]] += Ke[r, c]
        end
    end
    return sp.K, R
end

end # module
