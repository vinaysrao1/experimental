"""
    ModelMod

The assembled problem (`Model`) and the high-level UX builders
(`fix!`, `prescribe!`, `load!`) plus postprocessing. See DESIGN.md §4.6, §6.
"""
module ModelMod

using SparseArrays
using StaticArrays
using ..MeshMod: Mesh, GaussState, dof
using ..Materials: J2Material
using ..Elements: ElementCache, precompute_cache
using ..BoundaryConditions: DirichletBC, NeumannBC
using ..Assembly: SparsityPattern, build_sparsity

export Model, fix!, prescribe!, load!,
       nodal_displacements, gauss_stress, equivalent_plastic_strain

"""
    Model(mesh, material)

Assembled elastoplastic problem. Heavy arrays (cache, sparsity, state) are
preallocated once; `U` and buffers are reassignable (DESIGN §4.6).

BCs/loads are accumulated in growable lists by the builder API and frozen into
`DirichletBC`/`NeumannBC` at solve time.
"""
mutable struct Model
    mesh::Mesh
    material::J2Material
    cache::ElementCache
    sparsity::SparsityPattern
    state_committed::GaussState
    state_trial::GaussState
    Rbuf::Vector{Float64}
    U::Vector{Float64}
    # BC accumulators (DOF-indexed; deduplicated at solve)
    dir_dofs::Vector{Int}
    dir_vals::Vector{Float64}
    dir_ramp::Vector{Bool}
    neu_dofs::Vector{Int}
    neu_vals::Vector{Float64}
end

function Model(mesh::Mesh, material::J2Material)
    ndof = 3 * mesh.nnodes
    ngp = mesh.nelem * 8
    cache = precompute_cache(mesh.nodes, mesh.elements)
    sparsity = build_sparsity(mesh)
    return Model(mesh, material, cache, sparsity,
                 GaussState(ngp), GaussState(ngp),
                 zeros(ndof), zeros(ndof),
                 Int[], Float64[], Bool[], Int[], Float64[])
end

# component symbol -> list of component indices
function _comps(comp::Symbol)
    comp === :x && return (1,)
    comp === :y && return (2,)
    comp === :z && return (3,)
    comp === :all && return (1, 2, 3)
    error("unknown component $comp (use :x,:y,:z,:all)")
end

"""
    fix!(model, nodes, comp=:all)

Homogeneous Dirichlet (u = 0) on selected nodes/components (DESIGN §6.3).
"""
function fix!(model::Model, nodes::AbstractVector{<:Integer}, comp::Symbol=:all)
    for n in nodes, c in _comps(comp)
        push!(model.dir_dofs, dof(n, c))
        push!(model.dir_vals, 0.0)
        push!(model.dir_ramp, false)
    end
    return model
end

"""
    prescribe!(model, nodes, comp, value; ramp=true)

Inhomogeneous Dirichlet: prescribe `value` on selected nodes/component
(DESIGN §6.3).
"""
function prescribe!(model::Model, nodes::AbstractVector{<:Integer}, comp::Symbol,
                    value::Real; ramp::Bool=true)
    for n in nodes, c in _comps(comp)
        push!(model.dir_dofs, dof(n, c))
        push!(model.dir_vals, Float64(value))
        push!(model.dir_ramp, ramp)
    end
    return model
end

"""
    load!(model, nodes, comp, value; distribute=false, ramp=true)

Nodal force on selected nodes/component. `distribute=true` splits `value`
equally across the nodes (user gives a *total* face load) (DESIGN §6.3).
`ramp` is honored via the Neumann load factor at solve time.
"""
function load!(model::Model, nodes::AbstractVector{<:Integer}, comp::Symbol,
               value::Real; distribute::Bool=false, ramp::Bool=true)
    nn = length(nodes)
    per = distribute ? Float64(value) / nn : Float64(value)
    for n in nodes, c in _comps(comp)
        push!(model.neu_dofs, dof(n, c))
        push!(model.neu_vals, per)
    end
    # `ramp` always true in v1 Neumann; kept in signature for API completeness.
    ramp || @warn "load! ramp=false not supported in v1; load is ramped" maxlog=1
    return model
end

# Freeze accumulators into BC structs. Dirichlet ramp must be uniform per the
# simple struct; we honor per-dof ramp by splitting into two impose passes if
# needed — but in practice all-fixed (ramp=false) and prescribed (ramp=true)
# coexist, so we build one combined BC and carry the ramp flag per dof.
function _dirichlet(model::Model)
    return (model.dir_dofs, model.dir_vals, model.dir_ramp)
end

# --- postprocessing (DESIGN §6.3) ---

"""
    nodal_displacements(model) -> Matrix (3 × nnodes)
"""
function nodal_displacements(model::Model)
    U = model.U
    out = Matrix{Float64}(undef, 3, model.mesh.nnodes)
    @inbounds for n in 1:model.mesh.nnodes
        out[1, n] = U[dof(n, 1)]
        out[2, n] = U[dof(n, 2)]
        out[3, n] = U[dof(n, 3)]
    end
    return out
end

"""
    gauss_stress(model) -> Matrix (6 × ngp)  committed stresses
"""
gauss_stress(model::Model) = model.state_committed.σ

"""
    equivalent_plastic_strain(model) -> Vector (ngp)  committed ᾱ
"""
equivalent_plastic_strain(model::Model) = model.state_committed.ᾱ

end # module
