"""
    PlasticityFEM

A 3D finite element solver for small-strain elastoplasticity (J2 / von Mises,
combined linear isotropic + kinematic hardening, Hex8 elements). See docs/DESIGN.md.

Public API (curated, small):
    box_mesh, on_face, select_nodes, J2Material, Model, fix!, prescribe!, load!,
    solve!, nodal_displacements, gauss_stress, equivalent_plastic_strain
"""
module PlasticityFEM

include("Materials.jl")
include("Mesh.jl")
include("Elements.jl")
include("BoundaryConditions.jl")
include("Assembly.jl")
include("Model.jl")
include("Solver.jl")

using .Materials
using .MeshMod
using .Elements
using .BoundaryConditions
using .Assembly
using .ModelMod
using .Solver

# --- curated public exports (DESIGN §5.1) ---
export box_mesh, on_face, select_nodes
export J2Material
export Model, fix!, prescribe!, load!, solve!, reset!
export nodal_displacements, gauss_stress, equivalent_plastic_strain
export SolveResult

# Re-export selected internals useful for tests / advanced use.
export return_map, precompute_cache, element_force_tangent!,
       build_sparsity, assemble!, GaussState, DirichletBC, NeumannBC,
       impose_dirichlet!, dof, Mesh

end # module
