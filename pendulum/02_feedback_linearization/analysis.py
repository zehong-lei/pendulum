"""
Feedback Linearization — fully actuated pendulum, unconstrained torque.

01  Control law derivation
02  Closed-loop phase portrait vs open-loop
03  Convergence from multiple initial conditions
04  Torque requirements
05  Torque saturation: failure → motivation for energy shaping
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import solve_ivp

from pendulum.dynamics import pendulum_dynamics, wrap_angle, total_energy
from pendulum.simulation import DEFAULT_PARAMS

RESULTS = 'results/02_feedback_linearization'
PARAMS  = DEFAULT_PARAMS
KP      = 10.0
KD      = 6.0   # poles: s² + 6s + 10 = 0  →  s = -3 ± j


# ---------------------------------------------------------------------------
# Core controller
# ---------------------------------------------------------------------------

def _fl_torque(x, kp, kd, params, u_max=None):
    theta, omega = x
    m, l, g, b = params['m'], params['l'], params['g'], params['b']
    e = wrap_angle(theta - np.pi)
    v = -kp * e - kd * omega
    u = m * l**2 * (v + (g / l) * np.sin(theta) + (b / (m * l**2)) * omega)
    return float(np.clip(u, -u_max, u_max) if u_max else u)


def _simulate(x0, kp, kd, t_final=10.0, n=2000, u_max=None):
    ctrl = lambda t, x: _fl_torque(x, kp, kd, PARAMS, u_max)

    def f(t, x):
        return pendulum_dynamics(t, x, ctrl(t, x), PARAMS)

    sol = solve_ivp(f, (0, t_final), x0, dense_output=True,
                    rtol=1e-9, atol=1e-11)
    t = np.linspace(0, t_final, n)
    x = sol.sol(t)
    u = np.array([ctrl(t[i], x[:, i]) for i in range(n)])
    return t, x, u


def _save(filename):
    path = f'{RESULTS}/{filename}'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  saved: {path}')


# ---------------------------------------------------------------------------
# 01  Control law
# ---------------------------------------------------------------------------

def run_01_law():
    print('\n--- 01: Control Law ---')

    m, l, g, b = PARAMS['m'], PARAMS['l'], PARAMS['g'], PARAMS['b']
    kp, kd = KP, KD
    disc = kd**2 - 4 * kp
    re = -kd / 2
    im = np.sqrt(abs(disc)) / 2

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis('off')

    text = '\n'.join([
        r'$\bf{Original\ dynamics:}$',
        r'$\ddot{\theta}\ =\ -\dfrac{g}{l}\sin\theta\ -\ \dfrac{b}{ml^2}\omega\ +\ \dfrac{1}{ml^2}\,u$',
        '',
        r'$\bf{Feedback\ linearization\ —\ choose\ } u \bf{\ to\ cancel\ nonlinearities:}$',
        r'$u_{FL}\ =\ ml^2\!\left[\,v\ +\ \dfrac{g}{l}\sin\theta\ +\ \dfrac{b}{ml^2}\omega\,\right]$',
        '',
        r'$\bf{Closed\text{-}loop\ (exact\ cancellation):}$',
        r'$\ddot{\theta}\ =\ v\qquad$ (double integrator)',
        '',
        r'$\bf{Linear\ PD\ on\ upright\ error:}$',
        r'$e_\theta = \mathrm{wrap}(\theta - \pi),\quad'
        r'v = -k_p\,e_\theta - k_d\,\omega$',
        '',
        rf'$k_p={kp},\ k_d={kd}'
        rf'\quad\Rightarrow\quad'
        rf's^2 + {kd}s + {kp} = 0'
        rf'\quad\Rightarrow\quad'
        rf's = {re:.0f} \pm {im:.0f}j$',
    ])

    ax.text(0.5, 0.5, text,
            transform=ax.transAxes,
            ha='center', va='center',
            fontsize=14, linespacing=2.2,
            bbox=dict(boxstyle='round,pad=1.2',
                      facecolor='#f7f7f7', edgecolor='#bbbbbb'))
    plt.tight_layout()
    _save('01_law.png')


# ---------------------------------------------------------------------------
# 02  Phase portrait comparison
# ---------------------------------------------------------------------------

def run_02_portrait():
    print('\n--- 02: Phase Portrait ---')

    theta = np.linspace(-np.pi, 3 * np.pi, 500)
    omega = np.linspace(-8, 8, 500)
    TH, OM = np.meshgrid(theta, omega)

    m, l, g, b = PARAMS['m'], PARAMS['l'], PARAMS['g'], PARAMS['b']

    # open-loop vector field
    DT_ol  = OM
    DOM_ol = -(g / l) * np.sin(TH) - (b / (m * l**2)) * OM

    # closed-loop vector field (FL with no saturation)
    E_err  = (TH - np.pi + np.pi) % (2 * np.pi) - np.pi  # wrap(θ-π)
    DOM_cl = -KP * E_err - KD * OM                         # after cancellation

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    fig.suptitle('Phase Portrait: Open-Loop vs Feedback Linearization', fontsize=13)

    for ax, (DT, DOM, title, eq_label) in zip(axes, [
        (DT_ol, DOM_ol, 'Open-loop (no control)',    'saddle — unstable'),
        (DT_ol, DOM_cl, f'FL closed-loop  (kp={KP}, kd={KD})', 'stable spiral'),
    ]):
        speed = np.hypot(DT, DOM)
        speed[speed < 1e-6] = 1e-6
        ax.streamplot(theta, omega, DT, DOM,
                      color=np.log1p(speed), cmap='Blues',
                      density=1.5, linewidth=0.75, arrowsize=0.9)

        # mark upright equilibrium
        for k in range(-1, 3):
            th_u = (2 * k + 1) * np.pi
            if theta[0] <= th_u <= theta[-1]:
                col = 'tab:red' if 'Open' in title else 'tab:green'
                ax.plot(th_u, 0, '*', color=col, ms=14, zorder=7,
                        label=f'upright  ({eq_label})' if k == 0 else '')

        ax.set_xlim(theta[0], theta[-1])
        ax.set_ylim(-8, 8)
        ax.set_xticks([-np.pi, 0, np.pi, 2 * np.pi])
        ax.set_xticklabels(['-π', '0', 'π', '2π'])
        ax.set_xlabel('θ  (rad)', fontsize=12)
        ax.set_ylabel('ω  (rad/s)', fontsize=12)
        ax.set_title(title, fontsize=11)
        ax.axhline(0, color='k', lw=0.3, alpha=0.4)
        ax.legend(fontsize=9, loc='upper right')

    plt.tight_layout()
    _save('02_portrait.png')


# ---------------------------------------------------------------------------
# 03  Convergence from multiple initial conditions
# ---------------------------------------------------------------------------

def run_03_convergence():
    print('\n--- 03: Convergence from Multiple ICs ---')

    ics = [
        ([0.0,         0.0], 'θ=0, ω=0  (downward, at rest)'),
        ([0.0,         3.0], 'θ=0, ω=3'),
        ([np.pi / 2,   0.0], 'θ=π/2, ω=0  (side)'),
        ([np.pi / 2,   3.0], 'θ=π/2, ω=3'),
        ([3 * np.pi / 2, 0.0], 'θ=3π/2, ω=0  (other side)'),
        ([np.pi + 1.2, 0.0], 'θ=π+1.2, ω=0'),
    ]
    colors = plt.cm.tab10(np.linspace(0, 0.6, len(ics)))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Feedback Linearization — Convergence from Multiple Initial Conditions',
                 fontsize=12)

    ax_ph, ax_t = axes[0], axes[1]

    for (x0, label), col in zip(ics, colors):
        t, x, _ = _simulate(x0, KP, KD, t_final=8.0, n=1500)
        th_t = x[0]

        # phase plane
        ax_ph.plot(th_t, x[1], '-', color=col, lw=1.0, alpha=0.85)
        ax_ph.plot(th_t[0], x[1][0], 'o', color=col, ms=7, zorder=6)

        # θ(t)
        ax_t.plot(t, th_t, '-', color=col, lw=1.2, label=label)

    ax_ph.plot(np.pi, 0, 'g*', ms=16, zorder=7, label='upright  (target)')
    ax_ph.set_xlabel('θ  (rad)', fontsize=12)
    ax_ph.set_ylabel('ω  (rad/s)', fontsize=12)
    ax_ph.set_title('Phase plane', fontsize=11)
    ax_ph.legend(fontsize=7, loc='upper right')
    ax_ph.set_xticks([-np.pi, 0, np.pi, 2 * np.pi])
    ax_ph.set_xticklabels(['-π', '0', 'π', '2π'])
    ax_ph.axhline(0, color='k', lw=0.3, alpha=0.4)

    ax_t.axhline(np.pi, color='gray', lw=1.0, ls='--', alpha=0.6,
                 label='θ = π  (upright)')
    ax_t.set_xlabel('t  (s)', fontsize=12)
    ax_t.set_ylabel('θ  (rad)', fontsize=12)
    ax_t.set_title('θ(t)  — all converge to π', fontsize=11)
    ax_t.legend(fontsize=7, loc='upper right')
    ax_t.grid(True, alpha=0.3)
    ax_t.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    ax_t.set_yticklabels(['0', 'π/2', 'π', '3π/2'])

    plt.tight_layout()
    _save('03_convergence.png')


# ---------------------------------------------------------------------------
# 04  Torque requirements
# ---------------------------------------------------------------------------

def run_04_torque():
    print('\n--- 04: Torque Requirements ---')

    ics = [
        ([np.pi + 0.3, 0.0], 'near upright  θ = π+0.3'),
        ([np.pi / 2,   0.0], 'side  θ = π/2'),
        ([0.0,         0.0], 'downward  θ = 0'),
    ]
    colors = ['tab:green', 'tab:orange', 'tab:red']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Torque Requirements — FL (no saturation)', fontsize=12)

    ax_u, ax_th = axes[0], axes[1]
    u_max_line = PARAMS['u_max']

    for (x0, label), col in zip(ics, colors):
        t, x, u = _simulate(x0, KP, KD, t_final=6.0, n=1500)
        ax_u.plot(t, u,    '-', color=col, lw=1.3, label=f'{label}   peak={np.max(np.abs(u)):.1f} Nm')
        ax_th.plot(t, x[0], '-', color=col, lw=1.3, label=label)

    ax_u.axhline( u_max_line, color='k', ls='--', lw=1.2, label=f'u_max = ±{u_max_line} Nm')
    ax_u.axhline(-u_max_line, color='k', ls='--', lw=1.2)
    ax_u.set_xlabel('t  (s)', fontsize=12)
    ax_u.set_ylabel('u  (Nm)', fontsize=12)
    ax_u.set_title('Control torque u(t)', fontsize=11)
    ax_u.legend(fontsize=9)
    ax_u.grid(True, alpha=0.3)

    ax_th.axhline(np.pi, color='gray', ls='--', lw=1.0, alpha=0.6, label='upright  θ=π')
    ax_th.set_xlabel('t  (s)', fontsize=12)
    ax_th.set_ylabel('θ  (rad)', fontsize=12)
    ax_th.set_title('θ(t)', fontsize=11)
    ax_th.legend(fontsize=9)
    ax_th.grid(True, alpha=0.3)
    ax_th.set_yticks([0, np.pi / 2, np.pi])
    ax_th.set_yticklabels(['0', 'π/2', 'π'])

    plt.tight_layout()
    _save('04_torque.png')


# ---------------------------------------------------------------------------
# 05  Saturation: failure and motivation for energy shaping
# ---------------------------------------------------------------------------

def run_05_saturation():
    print('\n--- 05: Saturation Failure ---')

    x0     = [0.0, 0.0]   # downward, at rest
    u_max  = PARAMS['u_max']   # 10 Nm
    t_final = 15.0

    t_inf, x_inf, u_inf = _simulate(x0, KP, KD, t_final=t_final, n=2000)
    t_sat, x_sat, u_sat = _simulate(x0, KP, KD, t_final=t_final, n=2000,
                                     u_max=u_max)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f'Torque Saturation: FL unconstrained vs clipped to ±{u_max} Nm\n'
        f'(starting from downward  θ=0, ω=0)',
        fontsize=12
    )

    # θ(t)
    ax = axes[0, 0]
    ax.plot(t_inf, x_inf[0], 'b-', lw=1.5, label='FL unconstrained')
    ax.plot(t_sat, x_sat[0], 'r-', lw=1.5, label=f'FL  clipped ±{u_max} Nm')
    ax.axhline(np.pi, color='gray', ls='--', lw=1.0, alpha=0.7, label='upright θ=π')
    ax.set_xlabel('t  (s)'); ax.set_ylabel('θ  (rad)')
    ax.set_title('θ(t)', fontsize=11)
    ax.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_yticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # u(t)
    ax = axes[0, 1]
    ax.plot(t_inf, u_inf, 'b-', lw=1.5, label='FL unconstrained')
    ax.plot(t_sat, u_sat, 'r-', lw=1.5, label=f'FL  clipped ±{u_max} Nm')
    ax.axhline( u_max, color='k', ls='--', lw=1.0, label=f'±{u_max} Nm limit')
    ax.axhline(-u_max, color='k', ls='--', lw=1.0)
    ax.set_xlabel('t  (s)'); ax.set_ylabel('u  (Nm)')
    ax.set_title('Control torque u(t)', fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # phase plane
    for ax, (t_, x_, u_, label, col) in zip(
        [axes[1, 0], axes[1, 1]],
        [(t_inf, x_inf, u_inf, 'FL unconstrained', 'tab:blue'),
         (t_sat, x_sat, u_sat, f'FL ±{u_max} Nm', 'tab:red')]
    ):
        ax.plot(x_[0], x_[1], '-', color=col, lw=1.0)
        ax.plot(x_[0][0], x_[1][0], 'ko', ms=8, zorder=6, label='start')
        ax.plot(np.pi, 0, 'g*', ms=14, zorder=7, label='upright (target)')
        ax.set_xlabel('θ  (rad)'); ax.set_ylabel('ω  (rad/s)')
        ax.set_title(f'Phase plane — {label}', fontsize=10)
        ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        ax.set_xticklabels(['0', 'π/2', 'π', '3π/2'])
        ax.axhline(0, color='k', lw=0.3, alpha=0.4)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save('05_saturation.png')


# ---------------------------------------------------------------------------
# 06  Animation: downward → upright
# ---------------------------------------------------------------------------

def run_06_animation():
    print('\n--- 06: Animation ---')
    from matplotlib.animation import FuncAnimation, PillowWriter

    x0 = [0.0, 0.0]
    t_sim, x_sim, u_sim = _simulate(x0, KP, KD, t_final=8.0, n=4000)

    fps, t_end = 25, 8.0
    n_frames = int(t_end * fps)
    t_a  = np.linspace(0, t_end, n_frames)
    th_a = np.interp(t_a, t_sim, x_sim[0])
    u_a  = np.interp(t_a, t_sim, u_sim)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5),
                             gridspec_kw={'width_ratios': [1, 1.3, 1.3]})
    fig.suptitle('Feedback Linearization  —  downward → upright  (θ=0 → θ=π)', fontsize=12)
    fig.patch.set_facecolor('white')

    # pendulum panel
    ax_p = axes[0]
    ax_p.set_aspect('equal')
    ax_p.set_xlim(-1.4, 1.4);  ax_p.set_ylim(-1.4, 1.4)
    ax_p.set_xticks([]);        ax_p.set_yticks([])
    ax_p.set_facecolor('#f5f5f5')
    ax_p.set_title('Pendulum', fontsize=10)
    ax_p.plot(0, 1,  'g*', ms=13, zorder=3, label='target (upright)')
    ax_p.plot(0, 0,  'k.', ms=14, zorder=6)     # pivot

    trail, = ax_p.plot([], [], 'b-', lw=1.5, alpha=0.25, zorder=2)
    rod,   = ax_p.plot([], [], '-',  lw=5, color='#333', solid_capstyle='round', zorder=3)
    bob,   = ax_p.plot([], [], 'o',  color='tab:blue', ms=22, zorder=4)
    info   = ax_p.text(0, -1.33, '', ha='center', fontsize=8.5,
                       bbox=dict(facecolor='white', edgecolor='none', alpha=0.9))

    # θ(t) panel
    ax_t = axes[1]
    ax_t.set_xlim(0, t_end);  ax_t.set_ylim(-0.1, 4.0)
    ax_t.axhline(np.pi, ls='--', color='gray', lw=1.0, alpha=0.7)
    ax_t.text(0.15, np.pi + 0.13, 'target  θ=π', fontsize=8, color='gray')
    ax_t.set_xlabel('t  (s)');  ax_t.set_ylabel('θ  (rad)')
    ax_t.set_title('θ(t)', fontsize=10)
    ax_t.set_yticks([0, np.pi/2, np.pi])
    ax_t.set_yticklabels(['0', 'π/2', 'π'])
    ax_t.grid(True, alpha=0.3)
    th_line, = ax_t.plot([], [], 'b-', lw=1.8)
    th_dot,  = ax_t.plot([], [], 'o',  color='tab:blue', ms=6, zorder=5)

    # u(t) panel
    ax_u = axes[2]
    u_range = max(abs(u_sim)) * 1.15
    ax_u.set_xlim(0, t_end);  ax_u.set_ylim(-u_range, u_range)
    ax_u.axhline(0, color='k', lw=0.5, alpha=0.4)
    ax_u.set_xlabel('t  (s)');  ax_u.set_ylabel('u  (Nm)')
    ax_u.set_title('Control torque u(t)  [no saturation]', fontsize=10)
    ax_u.grid(True, alpha=0.3)
    u_line, = ax_u.plot([], [], color='tab:red', lw=1.8)
    u_dot,  = ax_u.plot([], [], 'o',  color='tab:red', ms=6, zorder=5)

    plt.tight_layout()

    trail_len = 35

    def update(i):
        th = th_a[i];  t_ = t_a[i];  u_ = u_a[i]
        bx, by = np.sin(th), -np.cos(th)
        s = max(0, i - trail_len)
        trail.set_data(np.sin(th_a[s:i+1]), -np.cos(th_a[s:i+1]))
        rod.set_data([0, bx], [0, by])
        bob.set_data([bx], [by])
        info.set_text(f't = {t_:.2f} s     θ = {th:.2f}     u = {u_:.1f} Nm')
        th_line.set_data(t_a[:i+1], th_a[:i+1])
        th_dot.set_data([t_], [th])
        u_line.set_data(t_a[:i+1], u_a[:i+1])
        u_dot.set_data([t_], [u_])
        return trail, rod, bob, info, th_line, th_dot, u_line, u_dot

    ani = FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps, blit=True)

    try:
        from matplotlib.animation import FFMpegWriter
        path = f'{RESULTS}/06_animation.mp4'
        ani.save(path, writer=FFMpegWriter(fps=fps, bitrate=1800))
    except Exception:
        path = f'{RESULTS}/06_animation.gif'
        ani.save(path, writer=PillowWriter(fps=fps))

    plt.close()
    print(f'  saved: {path}')


# ---------------------------------------------------------------------------

def run_feedback_linearization():
    print('\n' + '=' * 55)
    print('  Feedback Linearization  (fully actuated, u unconstrained)')
    print('=' * 55)
    os.makedirs(RESULTS, exist_ok=True)
    run_01_law()
    run_02_portrait()
    run_03_convergence()
    run_04_torque()
    run_05_saturation()
    run_06_animation()
    print(f'\nDone.  Results → {RESULTS}/')
