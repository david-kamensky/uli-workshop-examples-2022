from fenics import *
import math
M = 16; N = 32
mesh = RectangleMesh(Point(1,0),
                     Point(2,math.pi/2),
                     M,N)

# Introduce a symbolic variable corresponding
# to the polar coordinates.
xi = SpatialCoordinate(mesh)
r = xi[0]
theta = xi[1]

# Define the vector of Cartesian coordinates
# in terms of polar coordinates.
x = as_vector([r*cos(theta),
               r*sin(theta)])

# Jacobian of mapping; UFL `grad` is w.r.t. `xi`.
grad_xi = grad
dx_dxi = grad_xi(x)

# Pre-defined UFL integration measure `dx` is
# interpreted as integration w.r.t. `xi`.
dxi = dx

# Extend UFL to take the gradient with respect
# to physical coordinates, using the
# multivariate chain rule.
def grad_x(f):
    df_dxi = grad_xi(f)
    dxi_dx = inv(dx_dxi)
    return dot(df_dxi,dxi_dx)
 
# Set up weak form of Poisson equation, using
# change of variables in gradient and integration
# measure.
kappa = Constant(1)
f = sin(2*theta)
V = FunctionSpace(mesh,"CG",1)
u = TrialFunction(V)
v = TestFunction(V)
a = kappa*dot(grad_x(u),grad_x(v))*det(dx_dxi)*dxi
L = f*v*det(dx_dxi)*dxi

# Solve with boundary conditions.
bc = DirichletBC(V,Constant(0),"on_boundary")
u = Function(V)
solve(a==L,u,bc)

# Output the solution to a Paraview file.
u.rename("u","u")
File("u.pvd") << u

# Also write a projection of the displacement
# from `xi` to `x`, for visualization in
# physical space.
V_vec = VectorFunctionSpace(mesh,"CG",1)
u_hat = project(x-xi,V_vec)
u_hat.rename("u_hat","u_hat")
File("u_hat.pvd") << u_hat

# Note:  To visualize the solution in Paraview
# on a deformed mesh, load both "u.pvd" and
# "u_hat.pvd", then use the "Append Attributes"
# filter to combine them and a "Warp by Vector"
# filter to deform the mesh by `u_hat`.  
