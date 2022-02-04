from fenics import *
import math

# Setup from "poisson-basic.py".
M = 16; N = 32
mesh = RectangleMesh(Point(1,0),
                     Point(2,math.pi/2),
                     M,N)
x = SpatialCoordinate(mesh)
kappa = Constant(1)
f = sin(2*x[1])
V = FunctionSpace(mesh,"CG",1)
bc = DirichletBC(V,Constant(0),"on_boundary")

# Define the Dirichlet energy associated
# with the Poisson problem, in terms of a
# `Function` object `u`.
u = Function(V)
E = (kappa*dot(grad(u),grad(u)) - f*u)*dx

# The residual of this problem is its
# directional derivative w.r.t. `u`, in the
# direction of some arbitrary `TestFunction`.
v = TestFunction(V)
R = derivative(E,u,v)

# Syntax for solving a nonlinear problem
# (which handles this linear problem as
# a special case):
solve(R==0,u,bcs=[bc,])

# Alternative: For a linear problem, we can
# manually set up a single iteration of
# Newton's method, using another application
# of `derivative` and the linear problem
# `solve` syntax from earlier.

#Delta_u = TrialFunction(V)
#a = derivative(R,u,Delta_u)
#L = -R
#Delta_u = Function(V)
#solve(a==L,Delta_u,bc)
#u.assign(u + Delta_u)

# Postprocessing:
from matplotlib import pyplot
plot(u)
pyplot.show()
u.rename("u","u")
File("u.pvd") << u
