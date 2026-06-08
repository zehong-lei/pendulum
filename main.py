#!/usr/bin/env python3
"""
Pendulum Nonlinear Control Study

Usage:
    python main.py natural_dynamics
    python main.py feedback_linearization
    python main.py energy_shaping_local_stabilization
    python main.py optimal_control

Sections:
    01  natural_dynamics                    No input — study natural behavior
    02  feedback_linearization              Full torque — cancel nonlinearities globally
    03  energy_shaping_local_stabilization  Limited torque — swing-up + LQR
    04  optimal_control                     Minimum-time bang-bang via Pontryagin
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


def main():
    experiment = sys.argv[1] if len(sys.argv) > 1 else 'natural_dynamics'

    import importlib.util

    def _load(rel_path):
        path = os.path.join(os.path.dirname(__file__), rel_path)
        spec = importlib.util.spec_from_file_location('_mod', path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    dispatch = {
        'natural_dynamics':
            lambda: _load('pendulum/01_natural_dynamics/analysis.py').run_natural_dynamics(),
        'feedback_linearization':
            lambda: _load('pendulum/02_feedback_linearization/analysis.py').run_feedback_linearization(),
        'energy_shaping_local_stabilization':
            lambda: _load('pendulum/03_energy_shaping_local_stabilization/analysis.py').run_energy_shaping_local_stabilization(),
        'optimal_control':
            lambda: _load('pendulum/04_optimal_control/analysis.py').run_optimal_control(),
    }

    fn = dispatch.get(experiment)
    if fn is None:
        print(f'Unknown experiment: {experiment}')
        print(f'Available: {list(dispatch.keys())}')
        sys.exit(1)
    fn()


if __name__ == '__main__':
    main()
