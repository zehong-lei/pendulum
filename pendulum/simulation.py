import numpy as np
from scipy.integrate import solve_ivp
from pendulum.dynamics import pendulum_dynamics


DEFAULT_PARAMS = {
    'm': 1.0,
    'l': 1.0,
    'g': 9.81,
    'b': 0.1,
    'u_max': 10.0,
}


def simulate_pendulum(x0, controller, params=None, t_final=10.0, dt=0.01):
    """
    Simulate pendulum closed-loop dynamics.
    Returns:
        t: shape [N]
        x: shape [N, 2]
        u: shape [N]
    """
    if params is None:
        params = DEFAULT_PARAMS

    t_span = (0.0, t_final)
    t_eval = np.linspace(0.0, t_final, int(round(t_final / dt)) + 1)

    def closed_loop(t, x):
        u = controller(t, x)
        return pendulum_dynamics(t, x, u, params)

    sol = solve_ivp(closed_loop, t_span, x0, t_eval=t_eval,
                    method='RK45', rtol=1e-8, atol=1e-10)

    t = sol.t
    x = sol.y.T  # [N, 2]

    # Recompute control signal at each time step for logging
    u = np.array([controller(t[i], x[i]) for i in range(len(t))])

    return t, x, u
