import numpy as np


def wrap_angle(theta):
    """Normalize angle to [-pi, pi]."""
    return (theta + np.pi) % (2 * np.pi) - np.pi


def pendulum_dynamics(t, x, u, params):
    """
    Continuous-time pendulum dynamics.
    x = [theta, omega]
    returns xdot = [theta_dot, omega_dot]
    """
    theta, omega = x
    m = params['m']
    l = params['l']
    g = params['g']
    b = params['b']

    theta_dot = omega
    omega_dot = -(g / l) * np.sin(theta) - (b / (m * l**2)) * omega + (1 / (m * l**2)) * u

    return np.array([theta_dot, omega_dot])


def total_energy(x, params):
    """
    Return total mechanical energy.
    Zero potential energy at downward equilibrium theta=0.
    E = 0.5*m*l^2*omega^2 + m*g*l*(1 - cos(theta))
    """
    theta, omega = x
    m = params['m']
    l = params['l']
    g = params['g']
    return 0.5 * m * l**2 * omega**2 + m * g * l * (1 - np.cos(theta))


def upright_energy(params):
    """
    Return target energy corresponding to upright equilibrium theta=pi.
    E_d = m*g*l*(1 - cos(pi)) = 2*m*g*l
    """
    m = params['m']
    l = params['l']
    g = params['g']
    return 2 * m * g * l
