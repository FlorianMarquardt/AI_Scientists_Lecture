from equations_of_motion import MysteryAsymmetricDoubleWellInit
from sciexplorer.physics_eom  import ode_solve
import jax.numpy as jnp

dt = 0.001 # time step seen by the agent
solver_steps_per_timestep = 10 # simulator uses a smaller time step internally for better accuracy
T=20.0

args = dict(a = 1.1321, b = 0.8123, c=0.1, gamma=0.043, 
    dt=dt, 
    solver_steps_per_timestep=solver_steps_per_timestep,)

ex=MysteryAsymmetricDoubleWellInit(**args)

def run(q_init,q_dot_init):
    """
Run the experiment for given q and q_dot. Returns
ts,Q with array of times ts and array Q of shape
[time_steps,2], where Q[:,0] is q(t).
    """
    result=ex.observe_evolution(q_init, q_dot_init)
    return result['ts'],result['array']

def solve(X0,rhs):
    """
Solve a differential equation of given right hand side,
with rhs(X,t,params) [params is not used here]. This
has to return dX/dt.

X0 are the initial conditions.
    """
    return ode_solve(X0,rhs,jnp.array([]),dt,T,solver_steps_per_timestep)
    