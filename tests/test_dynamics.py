import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from pendulum.dynamics import wrap_angle, pendulum_dynamics, total_energy, upright_energy

PARAMS = {'m': 1.0, 'l': 1.0, 'g': 9.81, 'b': 0.1, 'u_max': 10.0}


class TestWrapAngle:
    def test_zero(self):
        assert wrap_angle(0.0) == pytest.approx(0.0)

    def test_pi(self):
        assert abs(wrap_angle(np.pi)) == pytest.approx(np.pi, abs=1e-10)

    def test_over_pi(self):
        assert wrap_angle(np.pi + 0.1) == pytest.approx(-np.pi + 0.1)

    def test_negative(self):
        assert wrap_angle(-np.pi - 0.5) == pytest.approx(np.pi - 0.5)


class TestPendulumDynamics:
    def test_downward_equilibrium(self):
        xdot = pendulum_dynamics(0.0, [0.0, 0.0], 0.0, PARAMS)
        assert xdot[0] == pytest.approx(0.0)
        assert xdot[1] == pytest.approx(0.0)

    def test_upright_equilibrium(self):
        # sin(pi) = 0, so upright is also an ODE equilibrium with u=0
        xdot = pendulum_dynamics(0.0, [np.pi, 0.0], 0.0, PARAMS)
        assert xdot[0] == pytest.approx(0.0, abs=1e-10)
        assert xdot[1] == pytest.approx(0.0, abs=1e-10)

    def test_energy_at_downward(self):
        assert total_energy([0.0, 0.0], PARAMS) == pytest.approx(0.0)

    def test_energy_at_upright(self):
        E = total_energy([np.pi, 0.0], PARAMS)
        assert E == pytest.approx(upright_energy(PARAMS))

    def test_upright_energy_value(self):
        assert upright_energy(PARAMS) == pytest.approx(2 * 9.81)

    def test_control_torque_effect(self):
        # u > 0 should increase omega_dot relative to u=0
        xdot_0 = pendulum_dynamics(0.0, [np.pi / 4, 0.0], 0.0, PARAMS)
        xdot_u = pendulum_dynamics(0.0, [np.pi / 4, 0.0], 5.0, PARAMS)
        assert xdot_u[1] > xdot_0[1]

    def test_damping_opposes_motion(self):
        # positive omega → damping reduces omega_dot
        xdot_damped   = pendulum_dynamics(0.0, [0.0, 2.0], 0.0, PARAMS)
        xdot_undamped = pendulum_dynamics(0.0, [0.0, 2.0], 0.0, {**PARAMS, 'b': 0.0})
        assert xdot_damped[1] < xdot_undamped[1]
