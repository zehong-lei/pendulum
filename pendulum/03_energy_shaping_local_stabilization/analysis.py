"""
Energy Shaping + Local Stabilization
Complete downward → upright control with limited torque.

Energy shaping:       far from upright, inject energy toward E_sep = 2mgl
Local stabilization:  near upright, apply linear state feedback (LQR here)
Switching:            latch from energy shaping to LQR when near upright

01  Energy shaping law
02  Energy convergence  E(t) → E_sep, switch point
03  Local stabilization concept — methods + LQR implementation
04  Complete trajectory θ(t), ω(t), two phases colour-coded
05  Phase portrait — full swing-up + stabilisation
06  Region of attraction: LQR alone vs Energy Shaping + LQR
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are

from pendulum.dynamics import pendulum_dynamics, wrap_angle, total_energy
from pendulum.simulation import DEFAULT_PARAMS

RESULTS = 'results/03_energy_shaping_local_stabilization'
PARAMS  = DEFAULT_PARAMS

K_E              = 2.0    # energy shaping gain
SWITCH_ANGLE     = 0.30   # rad  — switch when |wrap(θ-π)| < this
SWITCH_OMEGA     = 2.0    # rad/s
LQR_Q            = np.diag([10.0, 1.0])
LQR_R            = np.array([[0.1]])


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------

def _lqr_gain(params=PARAMS, Q=LQR_Q, R=LQR_R):
    m, l, g, b = params['m'], params['l'], params['g'], params['b']
    A = np.array([[0.0,       1.0],
                  [g / l, -b / (m * l**2)]])
    B = np.array([[0.0], [1.0 / (m * l**2)]])
    P = solve_continuous_are(A, B, Q, R)
    return (np.linalg.inv(R) @ B.T @ P).flatten()   # shape [2]


def make_swingup_ctrl(params=PARAMS, k_E=K_E,
                      angle_thresh=SWITCH_ANGLE, omega_thresh=SWITCH_OMEGA):
    K   = _lqr_gain(params)
    m, l, g, b = params['m'], params['l'], params['g'], params['b']
    E_d = 2 * m * g * l
    u_max = params['u_max']
    in_lqr = [False]

    def ctrl(t, x):
        theta, omega = x
        e = abs(wrap_angle(theta - np.pi))
        if in_lqr[0] or (e < angle_thresh and abs(omega) < omega_thresh):
            in_lqr[0] = True
            err = np.array([wrap_angle(theta - np.pi), omega])
            u = float(-K @ err)
        else:
            E        = total_energy(x, params)
            omega_eff = omega if abs(omega) > 1e-2 else 1e-2
            u        = k_E * (E_d - E) * omega_eff
        return float(np.clip(u, -u_max, u_max))

    return ctrl, in_lqr


def make_lqr_only_ctrl(params=PARAMS):
    """True local LQR — no angle wrapping, valid only near upright."""
    K     = _lqr_gain(params)
    u_max = params['u_max']

    def ctrl(t, x):
        theta, omega = x
        err = np.array([theta - np.pi, omega])   # unwrapped: linear around upright
        u   = float(-K @ err)
        return float(np.clip(u, -u_max, u_max))

    return ctrl


def _simulate(x0, ctrl, t_final=20.0, n=3000):
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
# 01  Energy shaping law
# ---------------------------------------------------------------------------

def run_01_energy_shaping_law():
    print('\n--- 01: Energy Shaping Law ---')

    m, l, g, b = PARAMS['m'], PARAMS['l'], PARAMS['g'], PARAMS['b']
    E_d = 2 * m * g * l

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis('off')

    text = '\n'.join([
        r'$\bf{Goal:}\ $ drive total energy $E \rightarrow E_d = 2mgl$ (upright energy)',
        '',
        r'$\bf{Total\ energy:}$',
        r'$E(\theta,\omega)\ =\ \frac{1}{2}ml^2\omega^2\ +\ mgl(1-\cos\theta)$',
        '',
        r'$\bf{Time\ derivative\ along\ trajectories\ (with\ control\ }u\bf{):}$',
        r'$\dot{E}\ =\ u\,\omega\ -\ b\,\omega^2$',
        '',
        r'$\bf{Choose\ } u \bf{\ to\ drive\ } E \rightarrow E_d\bf{:}$',
        r'$u_{ES}\ =\ k_E\,(E_d - E)\,\omega_{\mathrm{eff}}$',
        r'$\Rightarrow\quad \dot{E}\ =\ k_E\,(E_d-E)\,\omega^2\ -\ b\,\omega^2$',
        '',
        r'$\bullet\ E < E_d\ \Rightarrow\ \dot{E} > 0\quad$ (energy injected)',
        r'$\bullet\ E > E_d\ \Rightarrow\ \dot{E} < 0\quad$ (energy removed)',
        r'$\bullet\ \omega \approx 0\ $: regularise with $\omega_{\mathrm{eff}} = \max(|\omega|, \epsilon)\cdot\mathrm{sign}(\omega)$',
        '',
        rf'$k_E = {K_E},\quad E_d = 2mgl = {E_d:.2f}\ \mathrm{{J}},\quad u_{{max}} = {PARAMS["u_max"]}\ \mathrm{{Nm}}$',
    ])

    ax.text(0.5, 0.5, text,
            transform=ax.transAxes,
            ha='center', va='center',
            fontsize=13, linespacing=2.1,
            bbox=dict(boxstyle='round,pad=1.2',
                      facecolor='#f7f7f7', edgecolor='#bbbbbb'))
    plt.tight_layout()
    _save('01_energy_shaping_law.png')


# ---------------------------------------------------------------------------
# 02  Energy convergence
# ---------------------------------------------------------------------------

def run_02_energy_convergence():
    print('\n--- 02: Energy Convergence ---')

    x0 = [0.0, 0.0]
    ctrl, in_lqr_ref = make_swingup_ctrl()
    t, x, u = _simulate(x0, ctrl, t_final=20.0)

    m, l, g, b = PARAMS['m'], PARAMS['l'], PARAMS['g'], PARAMS['b']
    E_d = 2 * m * g * l
    E_t = np.array([total_energy(x[:, i], PARAMS) for i in range(len(t))])

    # find switch time: first index where |wrap(θ-π)| < thresh and |ω| < thresh
    angle_err = np.abs(np.array([wrap_angle(x[0, i] - np.pi) for i in range(len(t))]))
    switch_mask = (angle_err < SWITCH_ANGLE) & (np.abs(x[1]) < SWITCH_OMEGA)
    switch_idx  = int(np.argmax(switch_mask)) if switch_mask.any() else len(t) - 1
    t_switch    = t[switch_idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Energy Convergence — Energy Shaping → Switch → LQR', fontsize=12)

    ax = axes[0]
    ax.plot(t[:switch_idx], E_t[:switch_idx], 'b-', lw=1.5, label='Energy shaping phase')
    ax.plot(t[switch_idx:], E_t[switch_idx:], 'g-', lw=1.5, label='LQR phase')
    ax.axhline(E_d, color='red', ls='--', lw=1.5, label=f'E_d = 2mgl = {E_d:.2f} J')
    ax.axvline(t_switch, color='gray', ls=':', lw=1.2, label=f'switch at t={t_switch:.1f}s')
    ax.set_xlabel('t  (s)', fontsize=12)
    ax.set_ylabel('E  (J)', fontsize=12)
    ax.set_title('E(t)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(t[:switch_idx], u[:switch_idx], 'b-', lw=1.2, label='Energy shaping phase')
    ax2.plot(t[switch_idx:], u[switch_idx:], 'g-', lw=1.2, label='LQR phase')
    ax2.axhline( PARAMS['u_max'], color='k', ls='--', lw=1.0, alpha=0.5)
    ax2.axhline(-PARAMS['u_max'], color='k', ls='--', lw=1.0, alpha=0.5)
    ax2.axvline(t_switch, color='gray', ls=':', lw=1.2)
    ax2.set_xlabel('t  (s)', fontsize=12)
    ax2.set_ylabel('u  (Nm)', fontsize=12)
    ax2.set_title('Control torque u(t)', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _save('02_energy_convergence.png')


# ---------------------------------------------------------------------------
# 03  Local stabilization concept
# ---------------------------------------------------------------------------

def run_03_local_stabilization():
    print('\n--- 03: Local Stabilization ---')

    m, l, g, b = PARAMS['m'], PARAMS['l'], PARAMS['g'], PARAMS['b']
    K = _lqr_gain()
    A = np.array([[0.0, 1.0], [g / l, -b / (m * l**2)]])
    B = np.array([[0.0], [1.0 / (m * l**2)]])
    A_cl = A - B.reshape(2, 1) @ K.reshape(1, 2)
    eigs_ol = np.linalg.eigvals(A)
    eigs_cl = np.linalg.eigvals(A_cl)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Local Stabilization near Upright  (θ = π)', fontsize=13)

    # ---- left: concept text ----
    ax = axes[0]
    ax.axis('off')
    text = '\n'.join([
        r'$\bf{Linearise\ at\ upright\ }(\theta=\pi,\ \omega=0)\bf{:}$',
        rf'$A = [[0,\ 1],\ [{g/l:.2f},\ {-b/(m*l**2):.2f}]]$',
        '',
        rf'Open-loop eigenvalues:  $\lambda = {eigs_ol[0]:.3f},\ {eigs_ol[1]:.3f}$',
        r'$\Rightarrow$ real positive  →  unstable saddle',
        '',
        r'$\bf{Local\ stabilization\ methods:}$',
        r'$\bullet\ $ PD control         $u = -k_p e_\theta - k_d \omega$   (manual tuning)',
        r'$\bullet\ $ Pole placement      assign eigenvalues directly',
        r'$\bullet\ $ LQR                minimise $\int(x^TQx + u^TRu)\,dt$',
        '',
        r'$\bf{Here:\ LQR\ with}$'
        r'$\ Q = \mathrm{diag}(10,1),\ R = 0.1$',
        rf'$K = [{K[0]:.2f},\ {K[1]:.2f}]$',
        rf'Closed-loop eigenvalues:  $\lambda = {eigs_cl[0]:.3f},\ {eigs_cl[1]:.3f}$',
        r'$\Rightarrow$ negative real parts  →  stable',
    ])
    ax.text(0.5, 0.5, text,
            transform=ax.transAxes,
            ha='center', va='center',
            fontsize=11.5, linespacing=2.0,
            bbox=dict(boxstyle='round,pad=1.0',
                      facecolor='#f7f7f7', edgecolor='#bbbbbb'))

    # ---- right: local phase portrait under LQR ----
    ax2 = axes[1]
    delta = 1.2
    e_th = np.linspace(-delta, delta, 35)
    e_om = np.linspace(-delta, delta, 35)
    ETH, EOM = np.meshgrid(e_th, e_om)
    DT  = EOM
    DOM = A_cl[1, 0] * ETH + A_cl[1, 1] * EOM
    speed = np.hypot(DT, DOM)
    speed[speed < 1e-6] = 1e-6
    ax2.streamplot(e_th, e_om, DT, DOM,
                   color=np.log1p(speed), cmap='Blues',
                   density=1.4, linewidth=0.8)
    ax2.plot(0, 0, 'g*', ms=16, zorder=7, label='upright (stable under LQR)')
    ax2.set_xlabel('Δθ  (rad)', fontsize=11)
    ax2.set_ylabel('Δω  (rad/s)', fontsize=11)
    ax2.set_title('LQR closed-loop — local phase portrait', fontsize=11)
    ax2.axhline(0, color='k', lw=0.3, alpha=0.4)
    ax2.axvline(0, color='k', lw=0.3, alpha=0.4)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    _save('03_local_stabilization.png')


# ---------------------------------------------------------------------------
# 04  Complete trajectory
# ---------------------------------------------------------------------------

def run_04_trajectory():
    print('\n--- 04: Complete Trajectory ---')

    x0 = [0.0, 0.0]
    ctrl, _ = make_swingup_ctrl()
    t, x, u = _simulate(x0, ctrl, t_final=20.0)

    angle_err  = np.abs(np.array([wrap_angle(x[0, i] - np.pi) for i in range(len(t))]))
    switch_mask = (angle_err < SWITCH_ANGLE) & (np.abs(x[1]) < SWITCH_OMEGA)
    switch_idx  = int(np.argmax(switch_mask)) if switch_mask.any() else len(t) - 1
    t_switch    = t[switch_idx]

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    fig.suptitle('Complete Swing-Up Trajectory  (θ=0 → θ=π)\n'
                 f'Energy Shaping  [blue]  →  LQR  [green]  '
                 f'(switch at t = {t_switch:.1f} s)', fontsize=12)

    # θ(t)
    ax = axes[0]
    ax.plot(t[:switch_idx+1], x[0, :switch_idx+1], 'b-', lw=1.5)
    ax.plot(t[switch_idx:],   x[0, switch_idx:],   'g-', lw=1.5)
    ax.axhline(np.pi, color='red', ls='--', lw=1.2, alpha=0.7, label='upright  θ=π')
    ax.axvline(t_switch, color='gray', ls=':', lw=1.2)
    ax.set_ylabel('θ  (rad)', fontsize=12)
    ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax.set_yticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ω(t) and u(t) on twin axes
    ax2 = axes[1]
    ax2.plot(t[:switch_idx+1], u[:switch_idx+1], 'b-', lw=1.2, alpha=0.85)
    ax2.plot(t[switch_idx:],   u[switch_idx:],   'g-', lw=1.2, alpha=0.85)
    ax2.axhline( PARAMS['u_max'], color='k', ls='--', lw=0.8, alpha=0.4)
    ax2.axhline(-PARAMS['u_max'], color='k', ls='--', lw=0.8, alpha=0.4,
                label=f'±u_max = ±{PARAMS["u_max"]} Nm')
    ax2.axvline(t_switch, color='gray', ls=':', lw=1.2)
    ax2_r = ax2.twinx()
    ax2_r.plot(t[:switch_idx+1], x[1, :switch_idx+1], 'b--', lw=1.0, alpha=0.5)
    ax2_r.plot(t[switch_idx:],   x[1, switch_idx:],   'g--', lw=1.0, alpha=0.5)
    ax2.set_xlabel('t  (s)', fontsize=12)
    ax2.set_ylabel('u  (Nm)', fontsize=12, color='k')
    ax2_r.set_ylabel('ω  (rad/s)', fontsize=12, color='gray')
    ax2_r.tick_params(axis='y', labelcolor='gray')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _save('04_trajectory.png')


# ---------------------------------------------------------------------------
# 05  Phase portrait
# ---------------------------------------------------------------------------

def run_05_phase_portrait():
    print('\n--- 05: Phase Portrait ---')

    x0 = [0.0, 0.0]
    ctrl, _ = make_swingup_ctrl()
    t, x, u = _simulate(x0, ctrl, t_final=20.0)

    angle_err   = np.abs(np.array([wrap_angle(x[0, i] - np.pi) for i in range(len(t))]))
    switch_mask = (angle_err < SWITCH_ANGLE) & (np.abs(x[1]) < SWITCH_OMEGA)
    switch_idx  = int(np.argmax(switch_mask)) if switch_mask.any() else len(t) - 1

    m, l, g = PARAMS['m'], PARAMS['l'], PARAMS['g']
    E_sep = 2 * m * g * l
    th_fine = np.linspace(-2.2 * np.pi, 2.2 * np.pi, 600)
    disc    = 2 * (E_sep - m * g * l * (1 - np.cos(th_fine))) / (m * l**2)
    om_sep  = np.where(disc >= 0, np.sqrt(np.where(disc >= 0, disc, 0.0)), np.nan)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(th_fine,  om_sep, 'r--', lw=1.2, alpha=0.6, label='separatrix  E=2mgl')
    ax.plot(th_fine, -om_sep, 'r--', lw=1.2, alpha=0.6)

    ax.plot(x[0, :switch_idx+1], x[1, :switch_idx+1],
            'b-', lw=1.8, label='energy shaping phase')
    ax.plot(x[0, switch_idx:],   x[1, switch_idx:],
            'g-', lw=1.8, label='LQR phase')

    ax.plot(x[0, 0],          x[1, 0],          'ko',  ms=10, zorder=7, label='start  (0, 0)')
    ax.plot(x[0, switch_idx], x[1, switch_idx], 'y*',  ms=14, zorder=7, label='switch point')
    ax.plot(np.pi,             0,                'g*',  ms=14, zorder=7, label='upright (target)')

    ax.set_xlim(-0.5, 2.2 * np.pi)
    ax.set_ylim(-7, 7)
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    ax.set_xlabel('θ  (rad)', fontsize=12)
    ax.set_ylabel('ω  (rad/s)', fontsize=12)
    ax.set_title('Phase Portrait — Complete Swing-Up + Stabilisation', fontsize=12)
    ax.axhline(0, color='k', lw=0.3, alpha=0.4)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    _save('05_phase_portrait.png')


# ---------------------------------------------------------------------------
# 06  Region of attraction
# ---------------------------------------------------------------------------

def run_06_roa():
    print('\n--- 06: Region of Attraction ---')

    from scipy.linalg import solve_continuous_are, cholesky

    m, l, g, b = PARAMS['m'], PARAMS['l'], PARAMS['g'], PARAMS['b']
    u_max = PARAMS['u_max']

    # LQR Riccati solution
    A = np.array([[0.0,       1.0],
                  [g / l, -b / (m * l**2)]])
    B = np.array([[0.0], [1.0 / (m * l**2)]])
    P = solve_continuous_are(A, B, LQR_Q, LQR_R)
    K = _lqr_gain()   # shape [2]

    # Largest Lyapunov level set V = e^T P e ≤ c_max
    # where u = -K e is still within ±u_max.
    # max_{e^T P e = c} |K e| = sqrt(c) * ||K L^{-T}||  (L = cholesky(P))
    # Set = u_max → c_max = (u_max / ||K L^{-T}||)^2
    L        = cholesky(P, lower=True)          # P = L L^T
    KL_inv   = K @ np.linalg.inv(L.T)           # row vector
    sigma    = np.linalg.norm(KL_inv)
    c_max    = (u_max / sigma) ** 2

    # Ellipse boundary  { e : e^T P e = c_max }
    phi       = np.linspace(0, 2 * np.pi, 400)
    unit_z    = np.array([np.cos(phi), np.sin(phi)])
    ellipse_e = np.sqrt(c_max) * np.linalg.solve(L.T, unit_z)  # [2, 400]

    # Sample trajectories
    # inside ellipse: start near upright, LQR is valid → converges
    # outside ellipse: θ=π/2, ω=-4 (horizontal, spinning toward downward fast)
    #   LQR saturates at +10 Nm but gravity is -9.81 Nm; net decel is tiny,
    #   pendulum sweeps through bottom and does NOT converge within short time.
    inside_ic  = [np.pi + 0.25, 0.5]
    outside_ic = [np.pi / 2,   -4.0]    # far outside  (θ = π/2, ω = -4)

    t_in,  x_in,  _ = _simulate(inside_ic,  make_lqr_only_ctrl(), t_final=6.0, n=1000)
    t_out, x_out, _ = _simulate(outside_ic, make_lqr_only_ctrl(), t_final=4.0, n=600)
    e_in  = np.array([x_in[0]  - np.pi, x_in[1]])
    e_out = np.array([wrap_angle(x_out[0] - np.pi), x_out[1]])

    # ES+LQR simulation scan in absolute (θ, ω) space
    print('  Scanning ES+LQR...')
    theta_vals = np.linspace(-np.pi, np.pi, 41)
    omega_vals = np.linspace(-6.0,   6.0,   41)

    def _converged(x_final):
        return (abs(wrap_angle(x_final[0] - np.pi)) < 0.15 and
                abs(x_final[1]) < 0.5)

    roa_es = np.zeros((len(theta_vals), len(omega_vals)))
    total  = len(theta_vals) * len(omega_vals)
    count  = 0
    for i, th in enumerate(theta_vals):
        for j, om in enumerate(omega_vals):
            ctrl, _ = make_swingup_ctrl()
            _, x, _ = _simulate([th, om], ctrl, t_final=15.0, n=1200)
            roa_es[i, j] = float(_converged(x[:, -1]))
            count += 1
            if count % 200 == 0:
                print(f'    {count}/{total}')

    # ---- plot ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Region of Attraction: LQR alone vs Energy Shaping + LQR', fontsize=12)

    # left: LQR RoA — Lyapunov ellipse in error coordinates
    ax = axes[0]
    ax.fill(ellipse_e[0], ellipse_e[1],
            alpha=0.25, color='tab:green', label='LQR RoA  (Lyapunov estimate)')
    ax.plot(ellipse_e[0], ellipse_e[1], 'g-', lw=2.0)

    ax.plot(e_in[0],  e_in[1],  'b-',  lw=1.2, alpha=0.8, label='inside ellipse → converges')
    ax.plot(e_out[0], e_out[1], 'r-',  lw=1.2, alpha=0.8, label='outside ellipse → not guaranteed (4 s)')
    ax.plot(e_in[0, 0],  e_in[1, 0],  'bo', ms=7, zorder=6)
    ax.plot(e_out[0, 0], e_out[1, 0], 'ro', ms=7, zorder=6)
    ax.plot(0, 0, 'g*', ms=14, zorder=7, label='upright (target)')

    # mark downward equilibrium in error coords: e_theta = -π
    ax.axvline(-np.pi, color='gray', ls=':', lw=1.0, alpha=0.7)
    ax.axvline( np.pi, color='gray', ls=':', lw=1.0, alpha=0.7)
    ax.text(-np.pi, 5.5, 'downward\nequilibrium', ha='center', fontsize=8, color='gray')

    ax.set_xlim(-np.pi - 0.3, np.pi + 0.3)
    ax.set_ylim(-6.5, 6.5)
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    ax.set_xlabel('e_θ = θ − π  (rad)', fontsize=12)
    ax.set_ylabel('ω  (rad/s)', fontsize=12)
    ax.set_title('LQR only — Lyapunov RoA estimate\n'
                 f'(guaranteed only within ellipse,  u_max = {u_max} Nm)', fontsize=10)
    ax.axhline(0, color='k', lw=0.3, alpha=0.4)
    ax.axvline(0, color='k', lw=0.3, alpha=0.4)
    ax.legend(fontsize=8, loc='upper right')

    # right: ES+LQR simulation scan
    ax2 = axes[1]
    TH, OM = np.meshgrid(theta_vals, omega_vals, indexing='ij')
    ax2.contourf(TH, OM, roa_es, levels=[-0.5, 0.5, 1.5],
                 colors=['#ffcccc', '#ccffcc'])
    ax2.contour(TH, OM, roa_es, levels=[0.5], colors='k', linewidths=1.2)
    success = mpatches.Patch(color='#ccffcc', label='converges to upright')
    fail    = mpatches.Patch(color='#ffcccc', label='fails')
    ax2.legend(handles=[success, fail], fontsize=9)
    ax2.plot(0, 0, 'ko', ms=9, zorder=6, label='downward (start)')
    ax2.plot(np.pi, 0, 'g*', ms=12, zorder=6)
    ax2.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax2.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    ax2.set_xlabel('θ  (rad)', fontsize=12)
    ax2.set_ylabel('ω  (rad/s)', fontsize=12)
    ax2.set_title('Energy Shaping + LQR — simulation scan\n'
                  '(converges from full state space)', fontsize=10)
    ax2.axhline(0, color='k', lw=0.3, alpha=0.4)

    plt.tight_layout()
    _save('06_roa.png')


# ---------------------------------------------------------------------------
# 07  Animation: complete downward → upright, two-phase colour-coded
# ---------------------------------------------------------------------------

def run_07_animation():
    print('\n--- 07: Animation ---')
    from matplotlib.animation import FuncAnimation, PillowWriter

    x0 = [0.0, 0.0]
    ctrl, _ = make_swingup_ctrl()
    t_sim, x_sim, u_sim = _simulate(x0, ctrl, t_final=15.0, n=6000)

    m, l, g, b = PARAMS['m'], PARAMS['l'], PARAMS['g'], PARAMS['b']
    E_d   = 2 * m * g * l
    E_sim = np.array([total_energy(x_sim[:, i], PARAMS) for i in range(len(t_sim))])

    # switch index (first frame where latch would trigger)
    angle_err   = np.abs(np.array([wrap_angle(x_sim[0, i] - np.pi) for i in range(len(t_sim))]))
    switch_mask = (angle_err < SWITCH_ANGLE) & (np.abs(x_sim[1]) < SWITCH_OMEGA)
    switch_idx  = int(np.argmax(switch_mask)) if switch_mask.any() else len(t_sim) - 1
    t_switch    = t_sim[switch_idx]

    fps, t_end = 20, 15.0
    n_frames = int(t_end * fps)
    t_a  = np.linspace(0, t_end, n_frames)
    th_a = np.interp(t_a, t_sim, x_sim[0])
    u_a  = np.interp(t_a, t_sim, u_sim)
    E_a  = np.interp(t_a, t_sim, E_sim)

    sw_frame = int(t_switch / t_end * n_frames)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5),
                             gridspec_kw={'width_ratios': [1, 1.3, 1.3]})
    fig.suptitle('Energy Shaping + LQR  —  downward → upright  (θ=0 → θ=π)', fontsize=12)
    fig.patch.set_facecolor('white')

    # pendulum panel
    ax_p = axes[0]
    ax_p.set_aspect('equal')
    ax_p.set_xlim(-1.4, 1.4);  ax_p.set_ylim(-1.4, 1.4)
    ax_p.set_xticks([]);        ax_p.set_yticks([])
    ax_p.set_facecolor('#f5f5f5')
    ax_p.set_title('Pendulum', fontsize=10)
    ax_p.plot(0, 1, 'g*', ms=13, zorder=3)
    ax_p.plot(0, 0, 'k.', ms=14, zorder=6)

    trail,      = ax_p.plot([], [], '-',  lw=1.5, alpha=0.25, zorder=2)
    rod,        = ax_p.plot([], [], '-',  lw=5, color='#333', solid_capstyle='round', zorder=3)
    bob,        = ax_p.plot([], [], 'o',  ms=22, zorder=4)
    phase_label = ax_p.text(0, 1.30, 'Energy Shaping', ha='center', fontsize=9,
                            fontweight='bold', color='tab:blue')
    info        = ax_p.text(0, -1.33, '', ha='center', fontsize=8.5,
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.9))

    # E(t) panel
    ax_e = axes[1]
    ax_e.set_xlim(0, t_end);  ax_e.set_ylim(-0.5, 22)
    ax_e.axhline(E_d, ls='--', color='red',  lw=1.2, label=f'$E_d$ = {E_d:.1f} J')
    ax_e.axvline(t_switch, ls=':',  color='gray', lw=1.0, alpha=0.7)
    ax_e.text(t_switch + 0.2, 20.5, f'switch\nt={t_switch:.1f}s', fontsize=7.5, color='gray')
    ax_e.set_xlabel('t  (s)');  ax_e.set_ylabel('E  (J)')
    ax_e.set_title('Energy E(t)', fontsize=10)
    ax_e.legend(fontsize=8, loc='lower right')
    ax_e.grid(True, alpha=0.3)
    E_es_line,  = ax_e.plot([], [], 'b-', lw=1.8, label='ES phase')
    E_lqr_line, = ax_e.plot([], [], 'g-', lw=1.8, label='LQR phase')
    E_dot,      = ax_e.plot([], [], 'ko',  ms=5,  zorder=5)

    # u(t) panel
    ax_u = axes[2]
    u_max_v = PARAMS['u_max']
    ax_u.set_xlim(0, t_end);  ax_u.set_ylim(-u_max_v * 1.25, u_max_v * 1.25)
    ax_u.axhline( u_max_v, ls='--', color='k', lw=0.8, alpha=0.5, label=f'±{u_max_v} Nm')
    ax_u.axhline(-u_max_v, ls='--', color='k', lw=0.8, alpha=0.5)
    ax_u.axhline(0, color='k', lw=0.5, alpha=0.4)
    ax_u.axvline(t_switch, ls=':', color='gray', lw=1.0, alpha=0.7)
    ax_u.set_xlabel('t  (s)');  ax_u.set_ylabel('u  (Nm)')
    ax_u.set_title('Control torque u(t)', fontsize=10)
    ax_u.legend(fontsize=8)
    ax_u.grid(True, alpha=0.3)
    u_es_line,  = ax_u.plot([], [], 'b-', lw=1.8)
    u_lqr_line, = ax_u.plot([], [], 'g-', lw=1.8)
    u_dot,      = ax_u.plot([], [], 'ko',  ms=5,  zorder=5)

    plt.tight_layout()

    trail_len = 35

    def update(i):
        th = th_a[i];  t_ = t_a[i];  u_ = u_a[i];  E_ = E_a[i]
        bx, by = np.sin(th), -np.cos(th)
        is_lqr = (i >= sw_frame)
        color  = 'tab:green' if is_lqr else 'tab:blue'

        s = max(0, i - trail_len)
        trail.set_data(np.sin(th_a[s:i+1]), -np.cos(th_a[s:i+1]))
        trail.set_color(color)
        rod.set_data([0, bx], [0, by])
        bob.set_data([bx], [by])
        bob.set_color(color)
        phase_label.set_text('LQR' if is_lqr else 'Energy Shaping')
        phase_label.set_color(color)
        info.set_text(f't = {t_:.2f} s     u = {u_:.1f} Nm')

        sw = min(i + 1, sw_frame)
        E_es_line.set_data(t_a[:sw], E_a[:sw])
        if is_lqr:
            E_lqr_line.set_data(t_a[sw_frame:i+1], E_a[sw_frame:i+1])
        E_dot.set_data([t_], [E_])

        u_es_line.set_data(t_a[:sw], u_a[:sw])
        if is_lqr:
            u_lqr_line.set_data(t_a[sw_frame:i+1], u_a[sw_frame:i+1])
        u_dot.set_data([t_], [u_])

        return (trail, rod, bob, phase_label, info,
                E_es_line, E_lqr_line, E_dot,
                u_es_line, u_lqr_line, u_dot)

    ani = FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps, blit=True)

    try:
        from matplotlib.animation import FFMpegWriter
        path = f'{RESULTS}/07_animation.mp4'
        ani.save(path, writer=FFMpegWriter(fps=fps, bitrate=1800))
    except Exception:
        path = f'{RESULTS}/07_animation.gif'
        ani.save(path, writer=PillowWriter(fps=fps))

    plt.close()
    print(f'  saved: {path}')


# ---------------------------------------------------------------------------

def run_energy_shaping_local_stabilization():
    print('\n' + '=' * 60)
    print('  Energy Shaping + Local Stabilization')
    print('  downward  →  upright  (torque-limited)')
    print('=' * 60)
    os.makedirs(RESULTS, exist_ok=True)
    run_01_energy_shaping_law()
    run_02_energy_convergence()
    run_03_local_stabilization()
    run_04_trajectory()
    run_05_phase_portrait()
    run_06_roa()
    run_07_animation()
    print(f'\nDone.  Results → {RESULTS}/')
