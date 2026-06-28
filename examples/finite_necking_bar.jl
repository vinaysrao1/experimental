# Finite-strain necking of a tension bar (3D, F-bar).
#
# A classic large-deformation J2 benchmark (de Souza Neto / Simo): a bar pulled
# in tension develops a localized neck once plastic flow dominates. The F-bar
# element relieves the volumetric locking of the trilinear Hex8 so the neck can
# form. Run:  julia --project=. examples/finite_necking_bar.jl
#
# Output: finite_necking_bar.vtu — open in ParaView, Warp By Vector on
# `Displacement` to see the necked shape; color by `EqPlasticStrain`.

using PlasticityFEM

# slender bar [0,L]×[0,w]×[0,w]; refine along the axis so a neck can localize
L, w = 10.0, 1.0
mesh = box_mesh(L, w, w, 16, 3, 3)

# steel-like, modest hardening (lower hardening ⇒ more pronounced necking)
steel = J2Material(E = 210e3, ν = 0.3, σy0 = 250.0, Hiso = 600.0)

model = Model(mesh, steel; element = :finite_fbar)

# symmetry rollers on the three min faces + axial pull on xmax (displacement control)
fix!(model, on_face(mesh, :xmin), :x)
fix!(model, on_face(mesh, :ymin), :y)
fix!(model, on_face(mesh, :zmin), :z)
prescribe!(model, on_face(mesh, :xmax), :x, 0.6)     # 6% nominal elongation

# many small steps keep the F-bar Newton iteration in its (near-symmetric) basin
res = solve!(model; nsteps = 30, tol = 1e-7, maxiter = 40,
             linsolve = :direct, verbose = false)

println("converged: ", res.converged, "  iters/step: ", res.iters)
u = nodal_displacements(model)
ᾱ = equivalent_plastic_strain(model)
println("max axial displacement: ", maximum(u[1, :]))
println("max equivalent plastic strain: ", maximum(ᾱ))

out = write_vtu("finite_necking_bar", model)
println("wrote ", out)
