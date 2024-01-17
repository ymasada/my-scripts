"""
To run and plot using e.g. 4 processes:
    $ mpiexec -n 4 python3 shear_dynamo.py
    $ mpiexec -n 4 python3 plot_snapshots.py snapshots/*.h5
"""    
import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)


# Parameters
Lx,Ly,Lz  = 1,1,1
#Nx,Ny,Nz  = 128,128,128
#Nx,Ny,Nz  = 64,64,64
Nx,Ny,Nz  = 32,32,32 
Rossby    = 0.1 # control parameter: change from 1.0 to 0.01 (1.0, 0.1, 0.01) 
Elsasser  = 1e-2
Ekman     = 2e-4
mReynolds = 5e2

dealias = 3/2
stop_sim_time = 50
timestepper   = d3.RK222
max_timestep  = 2e-2
dtype = np.float64

# Bases
coords = d3.CartesianCoordinates('x','y','z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=dealias)
zbasis = d3.ChebyshevT( coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

# Fields
p      = dist.Field(name='p', bases=(xbasis,ybasis,zbasis))                # gauge pressure (not absolute pressure)
u      = dist.VectorField(coords, name='u',  bases=(xbasis,ybasis,zbasis)) # velocity (perturbed)
a      = dist.VectorField(coords, name='a',  bases=(xbasis,ybasis,zbasis)) # b-field  (perturbed)
b      = dist.VectorField(coords, name='b',  bases=(xbasis,ybasis,zbasis)) # b-field  (perturbed)
U0     = dist.VectorField(coords, name='U0', bases=(xbasis,ybasis,zbasis)) # background velocity 
B0     = dist.VectorField(coords, name='B0', bases=(xbasis,ybasis,zbasis)) # background B-field
A0     = dist.VectorField(coords, name='A0', bases=(xbasis,ybasis,zbasis)) # background vector potential
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=(xbasis,ybasis))
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=(xbasis,ybasis))
tau_b1 = dist.VectorField(coords, name='tau_b1', bases=(xbasis,ybasis))
tau_b2 = dist.VectorField(coords, name='tau_b2', bases=(xbasis,ybasis))
tau_p  = dist.Field(name='tau_p')

# Substitutions
Ro = Rossby
La = Elsasser
E  = Ekman 
Rm = mReynolds
Re = Ro/E
nu = 1/Re
beta_x = 0
beta_y = 1E-2
Bx0    = 0
By0    = 1E-2
x, y, z = dist.local_grids(xbasis, ybasis, zbasis) 
ex,ey,ez = coords.unit_vector_fields(dist)
dz = lambda A: d3.Differentiate(A, coords['z'])
lift_basis = zbasis.derivative_basis(1) # Chebyshev U basis
lift = lambda A: d3.Lift(A, lift_basis, -1) # Shortcut for multiplying by U_{N-1}(y)
grad_u = d3.grad(u) - ez*lift(tau_u1) # Operator representing Gu
grad_b = d3.grad(b) - ez*lift(tau_b1) # Operator representing Gb
ux = u@ex; uy = u@ey; uz = u@ez
bx = b@ex; by = b@ey; bz = b@ez

# Background shear flow & B-field (via vector potential A)
U0['g'][0] =  1.0 - 2.0*z
U0['g'][1] =  0.0
U0['g'][2] =  0.0
A0['g'][0] =  beta_y*z**2/2.0 + By0*z
A0['g'][1] = -beta_x*z**2/2.0 - Bx0*z
A0['g'][2] =  0.0
B0 = d3.curl(A0)
#b  = d3.curl(a)

# Problem
problem = d3.IVP([u, b, p, tau_p, tau_u1, tau_u2, tau_b1, tau_b2], namespace=locals())
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("trace(grad_b) = 0")
problem.add_equation("dt(u) +grad(p)/Ro -E/Ro*div(grad_u) + lift(tau_u2) = - grad(b@b)/Ro +2*(u@ez)*ex -U0@grad(u) -u@grad(u) -cross(ez,u)/Ro +La*b@grad(B0)/Ro +La*B0@grad(b)/Ro +La*b@grad(b)/Ro ")
problem.add_equation("dt(b) - (1.0/Rm)*div(grad_b) + lift(tau_b2)  =  curl(cross((U0 + u),(B0+b)))")
problem.add_equation("integ(p) = 0")                              # Pressure gauge
problem.add_equation("dz(ux)(z= 0) = 0")
problem.add_equation("dz(ux)(z=Lz) = 0")
problem.add_equation("dz(uy)(z= 0) = 0")
problem.add_equation("dz(uy)(z=Lz) = 0")
problem.add_equation("uz(z= 0)     = 0")
problem.add_equation("uz(z=Lz)     = 0")
problem.add_equation("dz(bx)(z= 0) = 0")
problem.add_equation("bx(z=Lz)     = 0")
problem.add_equation("dz(by)(z= 0) = 0")
problem.add_equation("by(z=Lz)     = 0")
problem.add_equation("bz(z= 0)     = 0")
problem.add_equation("dz(bz)(z=Lz) = 0")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
# Add small velocity perturbations
u.fill_random('g', distribution='normal', scale=1e-4)
#b.fill_random('g', distribution='normal', scale=1e-4)

# Analysis
slices_x = solver.evaluator.add_file_handler('slices_x', sim_dt=0.02, max_writes=10)
slices_x.add_task(d3.Interpolate(u@ex,'x',0.5), layout='g', name='vx_yz') 
slices_x.add_task(d3.Interpolate(b@ex,'x',0.5), layout='g', name='bx_yz')
slices_x.add_task(d3.Interpolate(b@ez,'x',0.5), layout='g', name='by_yz')

slices_y = solver.evaluator.add_file_handler('slices_y', sim_dt=0.02, max_writes=10)
slices_y.add_task(d3.Interpolate(u@ex,'y',0.5), layout='g', name='vx_xz') 
slices_y.add_task(d3.Interpolate(b@ex,'y',0.5), layout='g', name='bx_xz')
slices_y.add_task(d3.Interpolate(b@ez,'y',0.5), layout='g', name='by_xz')

slices_z = solver.evaluator.add_file_handler('slices_z', sim_dt=0.02, max_writes=10)
slices_z.add_task(d3.Interpolate(u@ex,'z',0.5), layout='g', name='vx_xy') 
slices_z.add_task(d3.Interpolate(b@ex,'z',0.5), layout='g', name='bx_xy')
slices_z.add_task(d3.Interpolate(b@ez,'z',0.5), layout='g', name='by_xy')

mean1 = solver.evaluator.add_file_handler('mean1', sim_dt=0.02)
mean2 = solver.evaluator.add_file_handler('mean2', sim_dt=0.02)
mean1.add_task(d3.integ(0.5*u@u), name='KE')
mean2.add_task(d3.integ(0.5*b@b), name='ME')

# CFL
dt = 1e-4
CFL = d3.CFL(solver, initial_dt=dt, cadence=1, safety=0.2, threshold=0.05,
             max_change=1.5, max_dt=max_timestep)#, min_change=0.5)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=1)
flow.add_property(np.sqrt(u@u)/nu, name='Re')

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_Re = flow.max('Re')
            logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f' %(solver.iteration, solver.sim_time, timestep, max_Re))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
