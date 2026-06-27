# End-loaded cantilever beam (DESIGN §6.2).
#
# A slender beam clamped at x=0 with a distributed tip load (total Fz = −P) on
# the x=L face. For a small (elastic) load the tip deflection matches Euler–
# Bernoulli beam theory (DESIGN T14):
#
#   δ = P L³ / (3 E I),   I = b h³ / 12.
#
# Hex8 with shear adds a small Timoshenko correction, so the FEM value is a few
# percent larger than the Euler–Bernoulli target.

using PlasticityFEM

L = 10.0; b = 1.0; h = 1.0
mesh  = box_mesh(L, b, h, 40, 4, 4)
# NOTE: with the tip load P=100 below the max bending stress is
# σ_max = (P L)(h/2)/I ≈ 6000 MPa, far above σy0=250 of DESIGN §6.2 — that
# combination yields/collapses the beam and cannot match the *elastic* beam-
# theory deflection of T14. We keep DESIGN's geometry and P=100 but raise the
# yield stress so the response stays elastic, which is the regime T14 targets
# (δ = P L³/(3 E I) ≈ 1.905). Lowering σy0 simply drives it plastic.
mat   = J2Material(E = 210e3, ν = 0.3, σy0 = 1.0e6, Hkin = 2000.0)
model = Model(mesh, mat)

# Clamp the x=0 face (all three components)
fix!(model, on_face(mesh, :xmin))

# Distributed tip load: total Fz = −P split across the xmax-face nodes
P = 100.0
load!(model, on_face(mesh, :xmax), :z, -P; distribute = true)

result = solve!(model; nsteps = 10, tol = 1e-8, maxiter = 25)

uz = nodal_displacements(model)[3, :]
δtip = maximum(abs, uz)

E = mat.E
I = b * h^3 / 12
δ_beam = P * L^3 / (3 * E * I)

println("converged       : ", result.converged)
println("iters per step  : ", result.iters)
println("δtip (FEM)      : ", δtip)
println("δ  (Euler-Bern) : ", δ_beam)
println("rel diff        : ", abs(δtip - δ_beam) / δ_beam)
