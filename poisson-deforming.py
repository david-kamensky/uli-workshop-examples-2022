from fenics import *
import math
N = 32
mesh = UnitSquareMesh(N,N)

# Define physical coordinates through
# a `Function` in a vector-valued space
# with a displacement at each mesh
# vertex.
xi = SpatialCoordinate(mesh)
V_vec = VectorFunctionSpace(mesh,"CG",1)
u_hat = Function(V_vec)
x = xi + u_hat

# Define change of variables in UFL:
grad_xi = grad
dx_dxi = grad_xi(x)
dxi = dx
def grad_x(f):
    df_dxi = grad_xi(f)
    dxi_dx = inv(dx_dxi)
    return dot(df_dxi,dxi_dx)
 
# Define residual of Poisson problem:
kappa = Constant(1)
f = Constant(1)
V = FunctionSpace(mesh,"CG",1)
u = Function(V)
E = (0.5*kappa*dot(grad_x(u),grad_x(u))
     - f*u)*det(dx_dxi)*dxi
R = derivative(E,u) # (Leave `TestFunction` anonymous.)

# Symbolic definition of a displacement field
# to project onto `u_hat`.
A = Constant(0)
u_hat_sym = as_vector([A*sin(2*pi*xi[0])*sin(2*pi*xi[1]),
                       A*sin(2*pi*xi[0])*sin(2*pi*xi[1])])

# Solve with a series of different amplitudes
# of deformation, accumulating solutions
# in a single Paraview file.
bc = DirichletBC(V,Constant(0),"on_boundary")
u_file = File("u.pvd")
u_hat_file = File("u_hat.pvd")
for i in range(0,32):
    A.assign(2e-3*i)
    u_hat.assign(project(u_hat_sym,V_vec))
    solve(R==0,u,bc)
    u.rename("u","u")
    u_hat.rename("u_hat","u_hat")
    u_file << u
    u_hat_file << u_hat

# Note:  To visualize the solution in Paraview
# on a deformed mesh, load both "u.pvd" and
# "u_hat.pvd", then use the "Append Attributes"
# filter to combine them and a "Warp by Vector"
# filter to deform the mesh by `u_hat`.  

# Symbolic functional derivative of the problem
# residual w.r.t. the displacement field `u_hat`,
# in the direction of an arbitrary (anonymous)
# `TestFunction`.
dR_du_hat = derivative(R,u_hat)

# Matrix of partial derivatives of the discrete
# residual vector w.r.t. nodal displacements:
dR_du_hat_mat = assemble(dR_du_hat)

# Print out entries as a NumPy array.
print(dR_du_hat_mat.array())
