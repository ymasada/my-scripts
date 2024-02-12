"""
To run and plot using e.g. 4 processes:
    $ mpiexec -n 4 python3 shear_dynamo.py
    $ mpiexec -n 4 python3 shear_dynamo.py --restart
    $ mpiexec -n 4 python3 plot_u_xy.py slices_u_xy/*.h5
"""    
import sys
import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

# Allow restarting via command line
restart = (len(sys.argv) > 1 and sys.argv[1] == '--restart')

# Parameters
Lx,Ly,Lz  = 1,1,1
#Nx,Ny,Nz  = 128,128,128
#Nx,Ny,Nz  = 64,64,64
Nx,Ny,Nz  = 32,32,32 
Rossby    = 0.1
Elsasser  = 1e-3
Ekman     = 1e-4
mReynolds = 1e3

dealias = 3/2
stop_sim_time = 1000
timestepper   = d3.RK222
max_timestep  = 5e-2
dtype = np.float64

# Bases
coords = d3.CartesianCoordinates('x','y','z')
dist   = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=dealias)
zbasis = d3.ChebyshevT( coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

# Fields
p       = dist.Field(name='p'  , bases=(xbasis,ybasis,zbasis))              # gauge pressure (not absolute pressure)
phi     = dist.Field(name='phi', bases=(xbasis,ybasis,zbasis))              # scalar potential     (perturbed)      
u       = dist.VectorField(coords, name='u',  bases=(xbasis,ybasis,zbasis)) # velocity             (perturbed)
a       = dist.VectorField(coords, name='a',  bases=(xbasis,ybasis,zbasis)) # vector potential     (perturbed)
b       = dist.VectorField(coords, name='b',  bases=(xbasis,ybasis,zbasis)) # b-field              (perturbed)
j       = dist.VectorField(coords, name='j',  bases=(xbasis,ybasis,zbasis)) # electric current     (perturbed)
em      = dist.VectorField(coords, name='em', bases=(xbasis,ybasis,zbasis)) # electro-motive force (perturbed)
U0      = dist.VectorField(coords, name='U0', bases=(xbasis,ybasis,zbasis)) # velocity             (background)
A0      = dist.VectorField(coords, name='A0', bases=(xbasis,ybasis,zbasis)) # vector potential     (background)
B0      = dist.VectorField(coords, name='B0', bases=(xbasis,ybasis,zbasis)) # b-field              (background)
tau_u1  = dist.VectorField(coords, name='tau_u1', bases=(xbasis,ybasis))
tau_u2  = dist.VectorField(coords, name='tau_u2', bases=(xbasis,ybasis))
tau_a1  = dist.VectorField(coords, name='tau_a1', bases=(xbasis,ybasis))
tau_a2  = dist.VectorField(coords, name='tau_a2', bases=(xbasis,ybasis))
tau_p   = dist.Field(name='tau_p')
tau_phi = dist.Field(name='tau_phi')

# Substitutions
Ro = Rossby
La = Elsasser
E  = Ekman 
Rm = mReynolds
Re = Ro/E
nu = 1/Re
beta_x  = 1E-2
beta_y  = 1E-2
Bx0     = 1E-3
By0     = 1E-3
x, y, z = dist.local_grids(xbasis, ybasis, zbasis) 
ex,ey,ez = coords.unit_vector_fields(dist)
dz = lambda A: d3.Differentiate(A, coords['z'])
lift_basis = zbasis.derivative_basis(1)     # Chebyshev U basis
lift = lambda A: d3.Lift(A, lift_basis, -1) # Shortcut for multiplying by U_{N-1}(y)
grad_u = d3.grad(u) + ez*lift(tau_u1) # Operator representing Gu
grad_a = d3.grad(a) + ez*lift(tau_a1) # Operator representing Ga
dot    = lambda A, B: d3.DotProduct(A, B)
b   = d3.curl(a)
j   = d3.curl(b)
em  = d3.cross(u,b)
ux  =  u@ex; uy  =  u@ey; uz  =  u@ez
ax  =  a@ex; ay  =  a@ey; az  =  a@ez
bx  =  b@ex; by  =  b@ey; bz  =  b@ez
emx = em@ex; emy = em@ey; emz = em@ez

# Background shear flow & B-field (via vector potential A)
U0['g'][0] =  1.0 - 2.0*z
U0['g'][1] =  0.0
U0['g'][2] =  0.0
A0['g'][0] = (beta_y*z**2/2.0 + By0*z)
A0['g'][1] =-(beta_x*z**2/2.0 + Bx0*z)
A0['g'][2] =  0.0
B0 = d3.curl(A0)

# Problem
problem = d3.IVP([u, a, p, phi, tau_p, tau_phi, tau_u1, tau_u2, tau_a1, tau_a2], namespace=locals())
problem.add_equation("trace(grad_u) + tau_p   = 0")
problem.add_equation("trace(grad_a) + tau_phi = 0")
problem.add_equation("dt(u) +grad(p)/Ro     -E/Ro*div(grad_u) + lift(tau_u2) = - La*grad(b@b/2)/Ro +2*(u@ez)*ex -U0@grad(u) -u@grad(u) -cross(ez,u)/Ro +La*b@grad(B0)/Ro +La*B0@grad(b)/Ro +La*b@grad(b)/Ro ")
problem.add_equation("dt(a) +grad(phi) - (1.0/Rm)*div(grad_a) + lift(tau_a2) = cross((U0+u),(B0+b))")
problem.add_equation("integ(p  ) = 0")                             # pressure gauge
problem.add_equation("integ(phi) = 0")                             # scalar potential (perturbed) = 0 
problem.add_equation("dz(ux)(z= 0) = 0") #BC for ux @z=0
problem.add_equation("dz(uy)(z= 0) = 0") #BC for uy @z=0
problem.add_equation("    uz(z= 0) = 0") #BC for uz @z=0
problem.add_equation("dz(ux)(z=Lz) = 0") #BC for ux @z=Lz
problem.add_equation("dz(uy)(z=Lz) = 0") #BC for uy @z=Lz
problem.add_equation("    uz(z=Lz) = 0") #BC for uz @z=Lz

#closed boundary for both top and bottom:
problem.add_equation("           a( z= 0) = 0", condition="nx != 0  or ny != 0") #BC for a  @z=0
problem.add_equation("   dot(ex,a)( z= 0) = 0", condition="nx == 0 and ny == 0") #BC for ax @z=0
problem.add_equation("   dot(ey,a)( z= 0) = 0", condition="nx == 0 and ny == 0") #BC for ay @z=0
problem.add_equation("dz(dot(ez,a))(z= 0) = 0", condition="nx == 0 and ny == 0") #BC for az @z=0
problem.add_equation("dz(ax)(z=Lz) = 0") #BC for ax @z=Lz
problem.add_equation("dz(ay)(z=Lz) = 0") #BC for ay @z=Lz
problem.add_equation("    az(z=Lz) = 0") #BC for az @z=Lz

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
if not restart:
    u.fill_random('g', distribution='normal', scale=1e-4)
    file_handler_mode = 'overwrite'
else:
    write, initial_timestep = solver.load_state('checkpoints/checkpoints_s22.h5')
    initial_timestep = 1e-4
    file_handler_mode = 'append'
    
# Analysis
cut_plane = 0.9
# For restarting calculation
checkpoints = solver.evaluator.add_file_handler('checkpoints', sim_dt=2.0, max_writes=1,  mode=file_handler_mode)
checkpoints.add_tasks(solver.state)

# For visualizing temporal evolution of snapshots of u, b, omega on 2D slices 
#-- FOR U (visuzalization purpose)
slices_u_yz = solver.evaluator.add_file_handler('slices_u_yz', sim_dt=0.2, max_writes=10, mode=file_handler_mode)
slices_u_yz.add_task(d3.Interpolate(ux,'x',cut_plane), layout='g', name='ux_yz') 
slices_u_yz.add_task(d3.Interpolate(uy,'x',cut_plane), layout='g', name='uy_yz')
slices_u_yz.add_task(d3.Interpolate(uz,'x',cut_plane), layout='g', name='uz_yz')
#--
slices_u_xz = solver.evaluator.add_file_handler('slices_u_xz', sim_dt=0.2, max_writes=10, mode=file_handler_mode)
slices_u_xz.add_task(d3.Interpolate(ux,'y',cut_plane), layout='g', name='ux_xz') 
slices_u_xz.add_task(d3.Interpolate(uy,'y',cut_plane), layout='g', name='uy_xz')
slices_u_xz.add_task(d3.Interpolate(uz,'y',cut_plane), layout='g', name='uz_xz')
#--
slices_u_xy = solver.evaluator.add_file_handler('slices_u_xy', sim_dt=0.2, max_writes=10, mode=file_handler_mode)
slices_u_xy.add_task(d3.Interpolate(ux,'z',cut_plane), layout='g', name='ux_xy') 
slices_u_xy.add_task(d3.Interpolate(uy,'z',cut_plane), layout='g', name='uy_xy')
slices_u_xy.add_task(d3.Interpolate(uz,'z',cut_plane), layout='g', name='uz_xy')
#-- FOR B (visuzalization purpose)
slices_b_yz = solver.evaluator.add_file_handler('slices_b_yz', sim_dt=0.2, max_writes=10, mode=file_handler_mode)
slices_b_yz.add_task(d3.Interpolate(bx,'x',cut_plane), layout='g', name='bx_yz') 
slices_b_yz.add_task(d3.Interpolate(by,'x',cut_plane), layout='g', name='by_yz')
slices_b_yz.add_task(d3.Interpolate(bz,'x',cut_plane), layout='g', name='bz_yz')
#--
slices_b_xz = solver.evaluator.add_file_handler('slices_b_xz', sim_dt=0.2, max_writes=10, mode=file_handler_mode)
slices_b_xz.add_task(d3.Interpolate(bx,'y',cut_plane), layout='g', name='bx_xz') 
slices_b_xz.add_task(d3.Interpolate(by,'y',cut_plane), layout='g', name='by_xz')
slices_b_xz.add_task(d3.Interpolate(bz,'y',cut_plane), layout='g', name='bz_xz')
#--
slices_b_xy = solver.evaluator.add_file_handler('slices_b_xy', sim_dt=0.2, max_writes=10, mode=file_handler_mode)
slices_b_xy.add_task(d3.Interpolate(bx,'z',cut_plane), layout='g', name='bx_xy') 
slices_b_xy.add_task(d3.Interpolate(by,'z',cut_plane), layout='g', name='by_xy')
slices_b_xy.add_task(d3.Interpolate(bz,'z',cut_plane), layout='g', name='bz_xy')
#-- FOR Omega (vorticity) (visuzalization purpose)
slices_om_yz = solver.evaluator.add_file_handler('slices_om_yz', sim_dt=0.2, max_writes=10, mode=file_handler_mode)
slices_om_yz.add_task(d3.Interpolate(d3.curl(u)@ex,'x',cut_plane), layout='g', name='omx_yz') 
slices_om_yz.add_task(d3.Interpolate(d3.curl(u)@ey,'x',cut_plane), layout='g', name='omy_yz')
slices_om_yz.add_task(d3.Interpolate(d3.curl(u)@ez,'x',cut_plane), layout='g', name='omz_yz')
#--
slices_om_xz = solver.evaluator.add_file_handler('slices_om_xz', sim_dt=0.2, max_writes=10, mode=file_handler_mode)
slices_om_xz.add_task(d3.Interpolate(d3.curl(u)@ex,'y',cut_plane), layout='g', name='omx_xz') 
slices_om_xz.add_task(d3.Interpolate(d3.curl(u)@ey,'y',cut_plane), layout='g', name='omy_xz')
slices_om_xz.add_task(d3.Interpolate(d3.curl(u)@ez,'y',cut_plane), layout='g', name='omz_xz')
#--
slices_om_xy = solver.evaluator.add_file_handler('slices_om_xy', sim_dt=0.2, max_writes=10, mode=file_handler_mode)
slices_om_xy.add_task(d3.Interpolate(d3.curl(u)@ex,'z',cut_plane), layout='g', name='omx_xy') 
slices_om_xy.add_task(d3.Interpolate(d3.curl(u)@ey,'z',cut_plane), layout='g', name='omy_xy')
slices_om_xy.add_task(d3.Interpolate(d3.curl(u)@ez,'z',cut_plane), layout='g', name='omz_xy')

# For analysing temporal evolutions of horizontal means of b
bx_h = d3.Average(bx, ('x','y'));by_h = d3.Average(by, ('x','y'));bz_h = d3.Average(bz, ('x','y'))
ex_h = d3.Average(emx,('x','y'));ey_h = d3.Average(emy,('x','y'));ez_h = d3.Average(emz,('x','y'))
td_bx  = solver.evaluator.add_file_handler('td_bx', sim_dt=0.05, max_writes=10, mode=file_handler_mode)
td_by  = solver.evaluator.add_file_handler('td_by', sim_dt=0.05, max_writes=10, mode=file_handler_mode)
td_bz  = solver.evaluator.add_file_handler('td_bz', sim_dt=0.05, max_writes=10, mode=file_handler_mode)
td_ex  = solver.evaluator.add_file_handler('td_ex', sim_dt=0.05, max_writes=10, mode=file_handler_mode)
td_ey  = solver.evaluator.add_file_handler('td_ey', sim_dt=0.05, max_writes=10, mode=file_handler_mode)
td_ez  = solver.evaluator.add_file_handler('td_ez', sim_dt=0.05, max_writes=10, mode=file_handler_mode)
td_bx.add_task(bx_h, name='bx_h')
td_by.add_task(by_h, name='by_h')
td_bz.add_task(bz_h, name='bz_h')
td_ex.add_task(ex_h, name='ex_h')
td_ey.add_task(ey_h, name='ey_h')
td_ez.add_task(ez_h, name='ez_h')

# For analysing temporal evolutions of volume mean values
mean1  = solver.evaluator.add_file_handler('mean1',  sim_dt=0.05, mode=file_handler_mode)
mean2  = solver.evaluator.add_file_handler('mean2',  sim_dt=0.05, mode=file_handler_mode)
mean3  = solver.evaluator.add_file_handler('mean3',  sim_dt=0.05, mode=file_handler_mode)
mean4  = solver.evaluator.add_file_handler('mean4',  sim_dt=0.05, mode=file_handler_mode)
mean5  = solver.evaluator.add_file_handler('mean5',  sim_dt=0.05, mode=file_handler_mode)
mean6  = solver.evaluator.add_file_handler('mean6',  sim_dt=0.05, mode=file_handler_mode)
mean7  = solver.evaluator.add_file_handler('mean7',  sim_dt=0.05, mode=file_handler_mode)
mean8  = solver.evaluator.add_file_handler('mean8',  sim_dt=0.05, mode=file_handler_mode)
mean9  = solver.evaluator.add_file_handler('mean9',  sim_dt=0.05, mode=file_handler_mode)

mean1.add_task( d3.integ(0.5*u@u),      name='KE')
mean2.add_task( d3.integ(0.5*b@b),      name='ME')
mean3.add_task( d3.integ(bx_h**2+by_h**2+bz_h**2), name='MME')
mean4.add_task( d3.integ(d3.curl(u)@u), name='Ki-H')
mean5.add_task( d3.integ(u@b),          name='Cr-H')
mean6.add_task( d3.integ(a@b),          name='Mg-H')
mean7.add_task( d3.integ(j@b),          name='Cu-H')
mean8.add_task( d3.integ(d3.div(b)),    name='Gauss')
mean9.add_task( d3.integ(phi),          name='SP')

# CFL
dt = 1e-4
CFL = d3.CFL(solver, initial_dt=dt, cadence=1, safety=0.25, threshold=0.05,
             max_change=1.5, max_dt=max_timestep)
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
