from fenics import *
import math

# Use a built-in function to define a
# structured $M\times N$ mesh.  It is possible
# import meshes from external mesh generators,
# but that is outside the scope of this
# introduction.
M = 16; N = 32
mesh = RectangleMesh(Point(1,0),
                     Point(2,math.pi/2),
                     M,N)

# Introduce a symbolic variable corresponding
# to coordinates in the mesh.
x = SpatialCoordinate(mesh)

# Coefficient and source function for the
# Poisson equation.
kappa = Constant(1)
f = sin(2*x[1])

# Define a function space of continuous Galerkin
# (CG) functions that are polynomials of degree
# one on each element of the mesh.
V = FunctionSpace(mesh,"CG",1)

# `TrialFunction` and `TestFunction` objects
# are symbolic placeholders used as form
# arguments in the definition of linear and
# bilinear forms.
u = TrialFunction(V)
v = TestFunction(V)

# Define $a$ and $L$ from the weak form of the
# Poisson equation as the UFL objects
# `a` and `L`, using math-like syntax.
a = kappa*dot(grad(u),grad(v))*dx
L = f*v*dx

# Define a boundary condition on the space `V`,
# enforcing that functions are zero on the
# entire boundary.
bc = DirichletBC(V,Constant(0),"on_boundary")

# Solve the variational problem, storing the
# solution in a `Function` `u`, which has an
# associated vector of basis function
# coefficients.
u = Function(V)
solve(a==L,u,bc)

# Interactive visualization with Matplotlib:
from matplotlib import pyplot
plot(u)
pyplot.show()

# Write the solution to a Paraview file,
# with a fixed name to re-use saved
# states.
u.rename("u","u")
File("u.pvd") << u
