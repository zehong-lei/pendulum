"""
Natural Dynamics: Pendulum — no controller.

01  Equations of motion
02  Phase portrait (undamped vs damped)
03  Equilibria and stability
04  Energy landscape
05  Time-domain trajectories (libration / near-separatrix / rotation)
06  Energy dissipation (undamped vs damped)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import solve_ivp

from pendulum.dynamics import pendulum_dynamics, total_energy
from pendulum.simulation import DEFAULT_PARAMS

RESULTS = 'results/01_natural_dynamics'
PARAMS = DEFAULT_PARAMS


def _p_undamped():
    return {**PARAMS, 'b': 0.0}


def _integrate(x0, params, t_final=10.0, n=2000):
    def f(t, x):
        return pendulum_dynamics(t, x, 0.0, params)
    sol = solve_ivp(f, (0, t_final), x0, dense_output=True,
                    rtol=1e-9, atol=1e-11)
    t = np.linspace(0, t_final, n)
    return t, sol.sol(t)  # t:[n]  x:[2, n]


def _separatrix_curve(params):
    m, l, g = params['m'], params['l'], params['g']
    E_sep = 2 * m * g * l
    th = np.linspace(-2.2 * np.pi, 2.2 * np.pi, 800)
    disc = 2 * (E_sep - m * g * l * (1 - np.cos(th))) / (m * l**2)
    valid = disc >= 0
    om = np.where(valid, np.sqrt(np.where(valid, disc, 0.0)), np.nan)
    return th, om


def _save(filename):
    path = f'{RESULTS}/{filename}'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  saved: {path}')


# ---------------------------------------------------------------------------
# 01  Equations of motion
# ---------------------------------------------------------------------------

def run_01_equations():
    print('\n--- 01: Equations of Motion ---')
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axis('off')

    m, l, g, b = PARAMS['m'], PARAMS['l'], PARAMS['g'], PARAMS['b']
    E_sep = 2 * m * g * l

    lines = '\n'.join([
        r'$\mathbf{State:}\ \ \mathbf{x} = [\,\theta,\ \omega\,]$',
        '',
        r'$\dot{\theta} = \omega$',
        '',
        r'$\dot{\omega} = -\dfrac{g}{l}\sin\theta\ -\ \dfrac{b}{ml^2}\,\omega$'
        r'$\ +\ \dfrac{1}{ml^2}\,u$',
        '',
        r'$\mathbf{Natural\ dynamics}\ (u=0)\mathbf{:}$',
        '',
        r'$\dot{\omega} = -\dfrac{g}{l}\sin\theta\ -\ \dfrac{b}{ml^2}\,\omega$',
        '',
        rf'$m={m}\ \mathrm{{kg}},\quad l={l}\ \mathrm{{m}},'
        rf'\quad g={g}\ \mathrm{{m/s^2}},\quad b={b}\ \mathrm{{(damping)}}$',
        '',
        rf'$E_{{sep}} = 2mgl = {E_sep:.2f}\ \mathrm{{J}}$',
    ])

    ax.text(0.5, 0.5, lines,
            transform=ax.transAxes,
            ha='center', va='center',
            fontsize=15, linespacing=2.0,
            bbox=dict(boxstyle='round,pad=1.2',
                      facecolor='#f7f7f7', edgecolor='#bbbbbb'))
    plt.tight_layout()
    _save('01_equations.png')


# ---------------------------------------------------------------------------
# 02  Phase portrait
# ---------------------------------------------------------------------------

def run_02_phase_portrait():
    print('\n--- 02: Phase Portrait ---')

    theta = np.linspace(-2.2 * np.pi, 2.2 * np.pi, 600)
    omega = np.linspace(-7.5, 7.5, 600)
    TH, OM = np.meshgrid(theta, omega)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    fig.suptitle('Phase Portrait', fontsize=14)

    for ax, (params, title) in zip(axes, [
        (_p_undamped(), 'Undamped  (b = 0)'),
        (PARAMS,        f'Damped  (b = {PARAMS["b"]})'),
    ]):
        m, l, g, b = params['m'], params['l'], params['g'], params['b']
        DT  = OM
        DOM = -(g / l) * np.sin(TH) - (b / (m * l**2)) * OM
        speed = np.hypot(DT, DOM)
        speed[speed < 1e-6] = 1e-6

        ax.streamplot(theta, omega, DT, DOM,
                      color=np.log1p(speed), cmap='Blues',
                      density=1.6, linewidth=0.75, arrowsize=0.9)

        is_damped = (b > 0)

        th_sep, om_sep = _separatrix_curve(params)
        sep_label = ('E = 2mgl  (undamped separatrix ref.)'
                     if is_damped else 'separatrix  E = 2mgl')
        ax.plot(th_sep,  om_sep, 'r--', lw=1.8, alpha=0.9, label=sep_label)
        ax.plot(th_sep, -om_sep, 'r--', lw=1.8, alpha=0.9)

        E_sep = 2 * m * g * l
        for frac, lw, col in [(0.25, 0.6, '0.65'), (0.55, 0.6, '0.55'),
                               (0.88, 0.6, '0.45'), (1.30, 0.6, '0.45'),
                               (1.80, 0.6, '0.55'), (2.60, 0.6, '0.65')]:
            E_t = frac * E_sep
            disc = 2 * (E_t - m * g * l * (1 - np.cos(theta))) / (m * l**2)
            valid = disc >= 0
            if not valid.any():
                continue
            om_c = np.where(valid, np.sqrt(np.where(valid, disc, 0.0)), np.nan)
            ax.plot(theta,  om_c, '-', lw=lw, color=col, alpha=0.7)
            ax.plot(theta, -om_c, '-', lw=lw, color=col, alpha=0.7)

        down_label = ('downward eq.  (asymptotically stable)'
                      if is_damped else 'downward eq.  (Lyapunov stable / center)')
        for k in range(-2, 3):
            th = 2 * k * np.pi
            if abs(th) <= 2.2 * np.pi:
                ax.plot(th, 0, 'go', ms=9, zorder=7,
                        label=down_label if k == 0 else '')
        for k in range(-2, 2):
            th = (2 * k + 1) * np.pi
            if abs(th) <= 2.2 * np.pi:
                ax.plot(th, 0, 'r^', ms=9, zorder=7,
                        label='upright eq. (unstable saddle)' if k == 0 else '')

        ax.set_xlim(-2.2 * np.pi, 2.2 * np.pi)
        ax.set_ylim(-7.5, 7.5)
        ax.set_xticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi])
        ax.set_xticklabels(['-2π', '-π', '0', 'π', '2π'])
        ax.set_xlabel('θ  (rad)', fontsize=12)
        ax.set_ylabel('ω  (rad/s)', fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.axhline(0, color='k', lw=0.3, alpha=0.4)
        ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    _save('02_phase_portrait.png')


# ---------------------------------------------------------------------------
# 03  Equilibria and stability  (geometry only, no eigenvalues)
# ---------------------------------------------------------------------------

def run_03_equilibria_stability():
    print('\n--- 03: Equilibria and Stability ---')

    delta = 1.5
    e_th = np.linspace(-delta, delta, 35)
    e_om = np.linspace(-delta, delta, 35)
    ETH, EOM = np.meshgrid(e_th, e_om)

    configs = [
        (0.0,   _p_undamped(), 'Downward  θ=0 — undamped',
         'Closed orbits\n→ Lyapunov stable\n   (not asymptotically)'),
        (0.0,   PARAMS,        f'Downward  θ=0 — damped (b={PARAMS["b"]})',
         'Spiral in\n→ asymptotically stable'),
        (np.pi, _p_undamped(), 'Upright  θ=π — undamped',
         'Saddle: trajectories flee\n→ unstable'),
        (np.pi, PARAMS,        f'Upright  θ=π — damped (b={PARAMS["b"]})',
         'Still a saddle\n→ unstable\n(damping doesn\'t help)'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle('Equilibria and Stability — read from phase portrait geometry',
                 fontsize=13)

    stable_color   = 'tab:green'
    unstable_color = 'tab:red'

    for ax, (theta_eq, params, title, note) in zip(axes.flat, configs):
        m, l, g, b = params['m'], params['l'], params['g'], params['b']
        TH_abs = ETH + theta_eq
        DT  = EOM
        DOM = -(g / l) * np.sin(TH_abs) - (b / (m * l**2)) * EOM
        speed = np.hypot(DT, DOM)
        speed[speed < 1e-6] = 1e-6

        ax.streamplot(e_th, e_om, DT, DOM,
                      color=np.log1p(speed), cmap='Blues',
                      density=1.4, linewidth=0.8)

        dot_color = stable_color if theta_eq == 0.0 else unstable_color
        ax.plot(0, 0, '*', color=dot_color, ms=16, zorder=7)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Δθ  (rad)', fontsize=9)
        ax.set_ylabel('Δω  (rad/s)', fontsize=9)
        ax.axhline(0, color='k', lw=0.3, alpha=0.4)
        ax.axvline(0, color='k', lw=0.3, alpha=0.4)
        ax.text(0.03, 0.97, note, transform=ax.transAxes,
                va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.4',
                          facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    _save('03_equilibria_stability.png')


# ---------------------------------------------------------------------------
# 04  Energy landscape
# ---------------------------------------------------------------------------

def run_04_energy_landscape():
    print('\n--- 04: Energy Landscape ---')

    m, l, g = PARAMS['m'], PARAMS['l'], PARAMS['g']
    E_sep = 2 * m * g * l

    theta = np.linspace(-2.2 * np.pi, 2.2 * np.pi, 700)
    omega = np.linspace(-7.5, 7.5, 700)
    TH, OM = np.meshgrid(theta, omega)
    E = 0.5 * m * l**2 * OM**2 + m * g * l * (1 - np.cos(TH))

    fig = plt.figure(figsize=(16, 6))
    fig.suptitle('Energy Landscape', fontsize=13)

    ax1 = fig.add_subplot(1, 2, 1)
    cf = ax1.contourf(TH, OM, np.clip(E, 0, 3 * E_sep), levels=60, cmap='viridis')
    plt.colorbar(cf, ax=ax1, label='E  (J)')
    cs = ax1.contour(TH, OM, E, levels=[E_sep], colors='red', linewidths=2.0)
    ax1.clabel(cs, fmt='E = 2mgl', fontsize=8)
    ax1.contour(TH, OM, E,
                levels=[0.25*E_sep, 0.55*E_sep, 0.88*E_sep,
                         1.30*E_sep, 1.80*E_sep, 2.60*E_sep],
                colors='white', linewidths=0.7, alpha=0.5)
    for k in range(-2, 3):
        th = 2 * k * np.pi
        if abs(th) <= 2.2 * np.pi:
            ax1.plot(th, 0, 'go', ms=9, zorder=7)
    for k in range(-2, 2):
        th = (2 * k + 1) * np.pi
        if abs(th) <= 2.2 * np.pi:
            ax1.plot(th, 0, 'r^', ms=9, zorder=7)
    ax1.set_xlim(-2.2*np.pi, 2.2*np.pi)
    ax1.set_ylim(-7.5, 7.5)
    ax1.set_xticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi])
    ax1.set_xticklabels(['-2π', '-π', '0', 'π', '2π'])
    ax1.set_xlabel('θ  (rad)', fontsize=12)
    ax1.set_ylabel('ω  (rad/s)', fontsize=12)
    ax1.set_title('E(θ, ω)  — color = energy', fontsize=11)

    ax2 = fig.add_subplot(1, 2, 2)
    th_1d = np.linspace(-2.2 * np.pi, 2.2 * np.pi, 600)
    V = m * g * l * (1 - np.cos(th_1d))
    ax2.plot(th_1d, V, 'b-', lw=2.0, label='V(θ) = mgl(1 − cosθ)')
    ax2.axhline(E_sep, color='red', ls='--', lw=1.5,
                label=f'E_sep = 2mgl = {E_sep:.2f} J')
    ax2.fill_between(th_1d, 0, np.minimum(V, E_sep), where=(V <= E_sep),
                     alpha=0.18, color='tab:green', label='libration  (E < E_sep)')
    ax2.fill_between(th_1d, E_sep, 3 * E_sep,
                     alpha=0.12, color='tab:orange', label='rotation  (E > E_sep)')
    for k in range(-2, 3):
        th = 2 * k * np.pi
        if abs(th) <= 2.2 * np.pi:
            ax2.plot(th, 0, 'go', ms=9, zorder=7)
    for k in range(-2, 2):
        th = (2 * k + 1) * np.pi
        if abs(th) <= 2.2 * np.pi:
            ax2.plot(th, E_sep, 'r^', ms=9, zorder=7)
    ax2.set_xlabel('θ  (rad)', fontsize=12)
    ax2.set_ylabel('Potential energy  V  (J)', fontsize=12)
    ax2.set_title('V(θ)  — potential slice at ω = 0', fontsize=11)
    ax2.set_xlim(-2.2 * np.pi, 2.2 * np.pi)
    ax2.set_ylim(-0.5, 3 * E_sep)
    ax2.set_xticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi])
    ax2.set_xticklabels(['-2π', '-π', '0', 'π', '2π'])
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _save('04_energy_landscape.png')


# ---------------------------------------------------------------------------
# 05  Time-domain trajectories
# ---------------------------------------------------------------------------

def run_05_time_domain():
    print('\n--- 05: Time-Domain Trajectories ---')

    m, l, g = PARAMS['m'], PARAMS['l'], PARAMS['g']
    E_sep = 2 * m * g * l
    omega_sep = np.sqrt(2 * E_sep / (m * l**2))  # ω at θ=0 on separatrix ≈ 6.26

    p = _p_undamped()
    cases = [
        ([0.0,  3.0],               p, 10.0,
         'Libration  (E < E_sep)\nlow-energy oscillation'),
        ([0.0,  0.99 * omega_sep],  p, 30.0,
         'Near separatrix  (E ≈ E_sep)\nslows dramatically near upright'),
        ([0.0,  1.20 * omega_sep],  p,  8.0,
         'Rotation  (E > E_sep)\nfull revolution'),
    ]

    th_sep, om_sep = _separatrix_curve(p)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('Time-Domain Trajectories — Undamped', fontsize=13)

    for row, (x0, params, t_final, label) in enumerate(cases):
        t, x = _integrate(x0, params, t_final=t_final, n=3000)
        th_t, om_t = x[0], x[1]

        # left: θ(t) and ω(t) on twin axes
        ax_l = axes[row, 0]
        ax_l.plot(t, th_t, 'b-', lw=1.2, label='θ  (rad)')
        ax_r = ax_l.twinx()
        ax_r.plot(t, om_t, 'r-', lw=1.0, alpha=0.75, label='ω  (rad/s)')
        ax_l.set_ylabel('θ  (rad)', color='b', fontsize=9)
        ax_r.set_ylabel('ω  (rad/s)', color='r', fontsize=9)
        ax_l.tick_params(axis='y', labelcolor='b')
        ax_r.tick_params(axis='y', labelcolor='r')
        ax_l.set_xlabel('t  (s)', fontsize=9)
        ax_l.set_title(label, fontsize=9)
        ax_l.axhline(np.pi,  color='gray', lw=0.6, ls='--', alpha=0.5)
        ax_l.axhline(-np.pi, color='gray', lw=0.6, ls='--', alpha=0.5)
        ax_l.axhline(0,      color='k',    lw=0.3, alpha=0.3)
        lines_l, labs_l = ax_l.get_legend_handles_labels()
        lines_r, labs_r = ax_r.get_legend_handles_labels()
        ax_l.legend(lines_l + lines_r, labs_l + labs_r, fontsize=7, loc='upper right')

        # right: trajectory on phase plane
        ax_p = axes[row, 1]
        ax_p.plot(th_t, om_t, 'b-', lw=1.0, alpha=0.85)
        ax_p.plot(th_t[0], om_t[0], 'go', ms=8, zorder=6, label='start')
        ax_p.plot(th_sep,  om_sep, 'r--', lw=1.0, alpha=0.5, label='separatrix')
        ax_p.plot(th_sep, -om_sep, 'r--', lw=1.0, alpha=0.5)
        ax_p.plot(0,      0,      'go', ms=7, zorder=5)
        ax_p.plot( np.pi, 0,      'r^', ms=7, zorder=5)
        ax_p.plot(-np.pi, 0,      'r^', ms=7, zorder=5)
        ax_p.set_xlabel('θ  (rad)', fontsize=9)
        ax_p.set_ylabel('ω  (rad/s)', fontsize=9)
        ax_p.set_title('Phase plane', fontsize=9)
        ax_p.set_xlim(-2.2 * np.pi, 2.2 * np.pi)
        ax_p.set_xticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi])
        ax_p.set_xticklabels(['-2π', '-π', '0', 'π', '2π'])
        ax_p.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    _save('05_time_domain.png')


# ---------------------------------------------------------------------------
# 06  Energy dissipation
# ---------------------------------------------------------------------------

def run_06_energy_dissipation():
    print('\n--- 06: Energy Dissipation ---')

    x0 = [1.5, 0.0]
    t_final = 30.0
    p_u = _p_undamped()

    t_u, x_u = _integrate(x0, p_u,   t_final=t_final, n=3000)
    t_d, x_d = _integrate(x0, PARAMS, t_final=t_final, n=3000)

    E_u = np.array([total_energy(x_u[:, i], p_u)    for i in range(len(t_u))])
    E_d = np.array([total_energy(x_d[:, i], PARAMS) for i in range(len(t_d))])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Energy Dissipation', fontsize=13)

    ax = axes[0]
    ax.plot(t_u, E_u, 'b-', lw=1.5, label='undamped  (b = 0)')
    ax.plot(t_d, E_d, 'r-', lw=1.5, label=f'damped  (b = {PARAMS["b"]})')
    ax.set_xlabel('t  (s)', fontsize=12)
    ax.set_ylabel('Total energy  E  (J)', fontsize=12)
    ax.set_title('E(t)  — undamped vs damped', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(x_u[0], x_u[1], 'b-', lw=0.9, alpha=0.7, label='undamped  (closed orbit)')
    ax2.plot(x_d[0], x_d[1], 'r-', lw=0.9, alpha=0.85, label='damped  (spiral in)')
    ax2.plot(x0[0], x0[1], 'ko', ms=7, zorder=6, label='start')
    ax2.plot(0, 0, 'go', ms=9, zorder=7)
    ax2.set_xlabel('θ  (rad)', fontsize=12)
    ax2.set_ylabel('ω  (rad/s)', fontsize=12)
    ax2.set_title('Phase plane — same initial condition', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _save('06_energy_dissipation.png')


# ---------------------------------------------------------------------------
# 07  Lyapunov stability
# ---------------------------------------------------------------------------

def run_07_lyapunov():
    print('\n--- 07: Lyapunov Stability ---')

    m, l, g, b = PARAMS['m'], PARAMS['l'], PARAMS['g'], PARAMS['b']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        r'Lyapunov Stability — $V(\theta,\omega) = E(\theta,\omega)$'
        '\n'
        r'$\dot{V} = -b\omega^2 \leq 0$   (computed from natural dynamics)',
        fontsize=12
    )

    # ---- left: V(θ,ω) level sets around downward equilibrium ----
    ax1 = axes[0]
    th = np.linspace(-np.pi, np.pi, 400)
    om = np.linspace(-5.0, 5.0, 400)
    TH, OM = np.meshgrid(th, om)
    V = 0.5 * m * l**2 * OM**2 + m * g * l * (1 - np.cos(TH))

    cf = ax1.contourf(TH, OM, V, levels=30, cmap='YlOrRd')
    plt.colorbar(cf, ax=ax1, label='V = E  (J)')
    ax1.contour(TH, OM, V, levels=10, colors='white', linewidths=0.6, alpha=0.6)
    ax1.plot(0, 0, 'go', ms=10, zorder=7, label='downward eq.  V = 0')
    ax1.set_xlabel('θ  (rad)', fontsize=11)
    ax1.set_ylabel('ω  (rad/s)', fontsize=11)
    ax1.set_title('V(θ, ω) — positive definite\nminimum at downward equilibrium', fontsize=10)
    ax1.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax1.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    ax1.legend(fontsize=9)

    # ---- middle: V̇ = -bω² as a function of ω ----
    ax2 = axes[1]
    om_1d = np.linspace(-5.0, 5.0, 300)
    Vdot  = -b * om_1d**2
    ax2.plot(om_1d, Vdot, 'r-', lw=2.0)
    ax2.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax2.fill_between(om_1d, Vdot, 0, alpha=0.15, color='red',
                     label=r'$\dot{V} = -b\omega^2 \leq 0$')
    ax2.plot(0, 0, 'ko', ms=8, zorder=5, label=r'$\dot{V} = 0$ only at $\omega = 0$')
    ax2.set_xlabel('ω  (rad/s)', fontsize=11)
    ax2.set_ylabel(r'$\dot{V}$  (W)', fontsize=11)
    ax2.set_title(r'$\dot{V}(\omega) = -b\omega^2$'
                  '\nnegative semi-definite', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ---- right: V(t) along a damped trajectory ----
    ax3 = axes[2]
    x0 = [1.5, 1.0]
    t_traj, x_traj = _integrate(x0, PARAMS, t_final=25.0, n=2000)
    V_t = np.array([total_energy(x_traj[:, i], PARAMS) for i in range(len(t_traj))])
    Vdot_t = np.gradient(V_t, t_traj)

    ax3.plot(t_traj, V_t, 'b-', lw=1.5, label='V(t) = E(t)')
    ax3.fill_between(t_traj, V_t, alpha=0.12, color='blue')
    ax3.axhline(0, color='k', lw=0.8, ls='--', alpha=0.4, label='V = 0  (equilibrium)')
    ax3.set_xlabel('t  (s)', fontsize=11)
    ax3.set_ylabel('V  (J)', fontsize=11)
    ax3.set_title('V(t) along damped trajectory\nmonotonically decreasing → 0', fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    _save('07_lyapunov.png')


# ---------------------------------------------------------------------------

def run_natural_dynamics():
    print('\n' + '=' * 55)
    print('  Natural Dynamics: Pendulum  (no controller)')
    print('=' * 55)
    os.makedirs(RESULTS, exist_ok=True)
    run_01_equations()
    run_02_phase_portrait()
    run_03_equilibria_stability()
    run_04_energy_landscape()
    run_05_time_domain()
    run_06_energy_dissipation()
    run_07_lyapunov()
    print(f'\nDone.  Results → {RESULTS}/')
