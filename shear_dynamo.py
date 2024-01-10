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
#Nx,Ny,Nz  = 64,64,64
#Nx,Ny,Nz  = 32,32,32
Nx,Ny,Nz  = 128,32,128 
Rossby    = 0.1
Elsasser  = 0.002
Ekman     = 0.1
mReynolds = 5e1

dealias = 3/2
stop_sim_time = 50#5
timestepper = d3.RK222
max_timestep = 2e-2
dtype = np.float64

# Bases
coords = d3.CartesianCoordinates('x','y','z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=dealias)
zbasis = d3.RealFourier(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

# Fields
p  = dist.Field(name='p', bases=(xbasis,ybasis,zbasis)) # gauge pressure (not absolute pressure)
u  = dist.VectorField(coords, name='u',  bases=(xbasis,ybasis,zbasis)) # velocity          (perturbed)
a  = dist.VectorField(coords, name='a',  bases=(xbasis,ybasis,zbasis)) # vector potential  (perturbed)
b  = dist.VectorField(coords, name='b',  bases=(xbasis,ybasis,zbasis)) # B-field           (perturbed)
U0 = dist.VectorField(coords, name='U0', bases=(xbasis,ybasis,zbasis)) # background velocity 
A0 = dist.VectorField(coords, name='A0', bases=(xbasis,ybasis,zbasis)) # background vector potential
B0 = dist.VectorField(coords, name='B0', bases=(xbasis,ybasis,zbasis)) # background B-field
tau_p1 = dist.Field(name='tau_p1')

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

# Background shear flow & B-field (via vector potential A)
U0['g'][0] =  1 - 2*z
U0['g'][1] =  0
U0['g'][2] =  0
A0['g'][0] =  beta_y*z**2/2 + By0*z
A0['g'][1] = -beta_x*z**2/2 - Bx0*z
A0['g'][2] =  0
B0 = d3.curl(A0)
b  = d3.curl(a)

# Problem
problem = d3.IVP([u, a, p, tau_p1], namespace=locals())
problem.add_equation("dt(u) +grad(p)/Ro -E*lap(u)/Ro = - grad(b@b)/Ro +2*(u@ez)*ex -U0@grad(u) -u@grad(u) -cross(ey,u)/Ro +La*b@grad(B0)/Ro +La*B0@grad(b)/Ro +La*b@grad(b)/Ro")
problem.add_equation("dt(a) -lap(a)/Rm = cross((U0 + u),(B0+b))") # with Coulomb gauge
problem.add_equation("div(u) + tau_p1 = 0")
problem.add_equation("integ(p) = 0") # Pressure gauge

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
# Add small velocity perturbations
u.fill_random('g', seed=42, distribution='normal', scale=1e-3)
a.fill_random('g', seed=42, distribution='normal', scale=1e-5)

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.02, max_writes=10)
snapshots.add_task(d3.Interpolate((u+U0)@ex,'y',0.5), layout='g', name='vx_xz') 
snapshots.add_task(d3.Interpolate(b@ex,'y',0.5), layout='g', name='bx_xz')
snapshots.add_task(d3.Interpolate(p   ,'y',0.5), layout='g', name='pressure')

graph1 = solver.evaluator.add_file_handler('graph1', sim_dt=0.02)
graph1.add_task(d3.integ(0.5*u@u), name='KE')
graph2 = solver.evaluator.add_file_handler('graph2', sim_dt=0.02)
graph2.add_task(d3.integ(0.5*b@b), name='ME')

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.5, threshold=0.05,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(u@u)/nu, name='Re')

# Main loop
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
