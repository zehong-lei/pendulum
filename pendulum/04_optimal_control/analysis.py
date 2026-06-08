"""
Optimal Control

01  LQR as optimal control   — cost function, HJB, Riccati, value function V = eᵀPe
02  Pontryagin's minimum principle — bang-bang structure for torque-limited pendulum
03  Minimum-time trajectory  — numerical bang-bang shooting; compare with ES+LQR
04  Comparison               — phase plane + time traces + performance table
05  Animation                — min-time optimal trajectory
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
from scipy.optimize import minimize as sp_minimize

from pendulum.dynamics import pendulum_dynamics, wrap_angle, total_energy
from pendulum.simulation import DEFAULT_PARAMS

RESULTS = 'results/04_optimal_control'
PARAMS  = DEFAULT_PARAMS

LQR_Q = np.diag([10.0, 1.0])
LQR_R = np.array([[0.1]])

# ES+LQR parameters (same as section 03)
K_E          = 2.0
SWITCH_ANGLE = 0.30
SWITCH_OMEGA = 2.0


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _lqr_gain(params=PARAMS, Q=LQR_Q, R=LQR_R):
    m, l, g, b = params['m'], params['l'], params['g'], params['b']
    A = np.array([[0.0, 1.0], [g / l, -b / (m * l**2)]])
    B = np.array([[0.0], [1.0 / (m * l**2)]])
    P = solve_continuous_are(A, B, Q, R)
    K = (np.linalg.inv(R) @ B.T @ P).flatten()
    return K, P


def _simulate(x0, ctrl, t_final=10.0, n=2000, params=PARAMS):
    def f(t, x):
        return pendulum_dynamics(t, x, ctrl(t, x), params)
    sol = solve_ivp(f, (0, t_final), x0, dense_output=True, rtol=1e-9, atol=1e-11)
    t   = np.linspace(0, t_final, n)
    x   = sol.sol(t)
    u   = np.array([ctrl(t[i], x[:, i]) for i in range(n)])
    return t, x, u


def _make_es_lqr(params=PARAMS):
    """Reproduce the ES+LQR swingup controller from section 03."""
    K, _ = _lqr_gain(params)
    u_max = params['u_max']
    m, l, g, b = params['m'], params['l'], params['g'], params['b']
    E_d = 2 * m * g * l
    in_lqr = [False]

    def ctrl(t, x):
        theta, omega = x
        e = abs(wrap_angle(theta - np.pi))
        if in_lqr[0] or (e < SWITCH_ANGLE and abs(omega) < SWITCH_OMEGA):
            in_lqr[0] = True
            err = np.array([wrap_angle(theta - np.pi), omega])
            u = float(-K @ err)
        else:
            E = total_energy(x, params)
            om_eff = omega if abs(omega) > 1e-2 else 1e-2 * np.sign(omega + 1e-9)
            u = K_E * (E_d - E) * om_eff
        return float(np.clip(u, -u_max, u_max))

    return ctrl


def _shoot_bang_bang(switch_times, signs, T, x0, params):
    """Simulate piecewise-constant bang-bang u = signs[k] * u_max."""
    u_max = params['u_max']
    boundaries = [0.0] + list(switch_times) + [T]
    x = np.array(x0, dtype=float)
    for k, sgn in enumerate(signs):
        ta, tb = boundaries[k], boundaries[k + 1]
        if tb <= ta + 1e-9:
            continue
        u = float(sgn * u_max)
        sol = solve_ivp(
            lambda t, s: pendulum_dynamics(t, s, u, params),
            (ta, tb), x, method='RK45', rtol=1e-9, atol=1e-11
        )
        x = sol.y[:, -1]
    return x


def _find_min_time(x0, params=PARAMS, verbose=False):
    """
    Find minimum-time bang-bang from x0 to upright (π, 0).
    Tries 1-switch and 2-switch sign sequences via Nelder-Mead.
    Returns (T_opt, switch_times, signs).
    """
    best_T     = np.inf
    best_sw    = None
    best_signs = None

    # Only CCW-first sequences: initial effective control must be +u_max
    # (pendulum swings CCW from downward toward upright)
    for signs in ([1, -1], [1, -1, 1]):
        n_sw = len(signs) - 1

        def objective(vars, signs=signs, n_sw=n_sw):
            sw_raw = np.sort(vars[:n_sw])
            T      = vars[n_sw]
            # require first switch > 0.15 s to avoid degenerate CW solution
            if T < 0.3 or sw_raw[0] < 0.15 or np.any(sw_raw >= T):
                return 1e8
            x_f  = _shoot_bang_bang(sw_raw, signs, T, x0, params)
            e_th = wrap_angle(x_f[0] - np.pi)
            e_om = x_f[1]
            return T + 8000.0 * (e_th**2 + e_om**2)

        for T_try in [1.2, 1.8, 2.5, 3.5]:
            sw0 = [T_try * (k + 1) / (n_sw + 1) for k in range(n_sw)]
            x0_opt = np.array(sw0 + [T_try])

            res = sp_minimize(
                objective, x0_opt, method='Nelder-Mead',
                options={'maxiter': 20000, 'xatol': 1e-7, 'fatol': 1e-7}
            )
            sw_opt = np.sort(res.x[:n_sw])
            T_opt  = res.x[n_sw]

            if T_opt < 0.3:
                continue
            x_f  = _shoot_bang_bang(sw_opt, signs, T_opt, x0, params)
            err  = wrap_angle(x_f[0] - np.pi)**2 + x_f[1]**2

            if verbose:
                print(f'    signs={signs}  T={T_opt:.3f}  err={err:.4f}')

            if err < 0.02 and T_opt < best_T:
                best_T     = T_opt
                best_sw    = sw_opt
                best_signs = signs

    return best_T, best_sw, best_signs


def _reconstruct_bang_bang(switch_times, signs, T, x0, params, n=1000):
    """Full trajectory reconstruction for bang-bang solution."""
    u_max = params['u_max']
    boundaries = [0.0] + list(switch_times) + [T]
    all_t, all_x, all_u = [], [], []
    x = np.array(x0, dtype=float)

    for k, sgn in enumerate(signs):
        ta, tb = boundaries[k], boundaries[k + 1]
        if tb <= ta + 1e-9:
            continue
        u_val = float(sgn * u_max)
        n_seg = max(2, int(n * (tb - ta) / T))
        sol = solve_ivp(
            lambda t, s: pendulum_dynamics(t, s, u_val, params),
            (ta, tb), x,
            dense_output=True, method='RK45', rtol=1e-9, atol=1e-11
        )
        t_seg = np.linspace(ta, tb, n_seg)
        x_seg = sol.sol(t_seg)
        all_t.append(t_seg)
        all_x.append(x_seg)
        all_u.extend([u_val] * n_seg)
        x = x_seg[:, -1]

    t = np.concatenate(all_t)
    x = np.concatenate(all_x, axis=1)
    u = np.array(all_u)
    return t, x, u


def _save(filename):
    path = f'{RESULTS}/{filename}'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  saved: {path}')


# ---------------------------------------------------------------------------
# 01  LQR as optimal control
# ---------------------------------------------------------------------------

def run_01_lqr_optimal():
    print('\n--- 01: LQR as Optimal Control ---')

    K, P = _lqr_gain()
    m, l, g, b = PARAMS['m'], PARAMS['l'], PARAMS['g'], PARAMS['b']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('LQR as Optimal Control  (linearised system near upright)', fontsize=13)

    # left: derivation text
    ax = axes[0]
    ax.axis('off')
    text = '\n'.join([
        r'$\bf{Infinite\text{-}horizon\ LQR\ problem:}$',
        r'$\min_{u}\ J = \int_0^{\infty}(x^T Q\,x + u^T R\,u)\,dt$',
        '',
        r'$\bf{Hamilton\text{-}Jacobi\text{-}Bellman\ (HJB):}$',
        r'$\min_u \left[x^TQx + u^TRu + \nabla V \cdot f\right] = 0$',
        '',
        r'$\bf{Ansatz\ } V^*(x) = x^T P x \bf{\ (quadratic):}$',
        r'$A^TP + PA - PBR^{-1}B^TP + Q = 0\quad$ (Riccati)',
        '',
        r'$\bf{Optimal\ policy:}$',
        r'$u^* = -Kx,\qquad K = R^{-1}B^TP$',
        '',
        rf'$K = [{K[0]:.2f},\ {K[1]:.2f}],\quad'
        rf'Q = \mathrm{{diag}}(10,1),\ R = 0.1$',
        '',
        r'$\bf{Interpretation:}$',
        r'$V^*(x) = x^TPx$ is the minimum cost',
        r'achievable from state $x$.',
        r'LQR trajectories follow $\nabla V^*$ downhill.',
    ])
    ax.text(0.5, 0.5, text, transform=ax.transAxes,
            ha='center', va='center', fontsize=11, linespacing=2.1,
            bbox=dict(boxstyle='round,pad=1.2', facecolor='#f7f7f7', edgecolor='#bbb'))

    # right: value function contour + LQR trajectories
    ax2 = axes[1]
    e_th_grid = np.linspace(-1.5, 1.5, 300)
    e_om_grid = np.linspace(-5.0, 5.0, 300)
    ETH, EOM  = np.meshgrid(e_th_grid, e_om_grid)
    V = P[0,0]*ETH**2 + (P[0,1]+P[1,0])*ETH*EOM + P[1,1]*EOM**2

    levels = np.linspace(0, np.percentile(V, 80), 18)
    cp = ax2.contourf(ETH, EOM, V, levels=levels, cmap='YlOrRd_r', alpha=0.75)
    ax2.contour(ETH, EOM, V, levels=levels, colors='k', linewidths=0.35, alpha=0.35)
    plt.colorbar(cp, ax=ax2, label='V*(e)  =  optimal cost-to-go')

    K_lqr, _ = _lqr_gain()
    for e0 in [[-1.2, 0.0], [-0.9, 3.5], [0.5, -4.0], [1.1, 2.0], [-0.4, -3.0]]:
        x_abs = [np.pi + e0[0], e0[1]]
        ctrl  = lambda t, x, K=K_lqr: float(
            np.clip(-K @ np.array([wrap_angle(x[0]-np.pi), x[1]]), -PARAMS['u_max'], PARAMS['u_max']))
        t_tr, x_tr, _ = _simulate(x_abs, ctrl, t_final=5.0, n=600)
        e_tr = np.array([wrap_angle(x_tr[0] - np.pi), x_tr[1]])
        ax2.plot(e_tr[0], e_tr[1], 'b-', lw=1.3, alpha=0.75)
        ax2.plot(e_tr[0, 0], e_tr[1, 0], 'bo', ms=6, zorder=5)

    ax2.plot(0, 0, 'g*', ms=16, zorder=6, label='upright (target)')
    ax2.set_xlabel(r'$e_\theta = \theta - \pi$  (rad)', fontsize=11)
    ax2.set_ylabel(r'$\omega$  (rad/s)', fontsize=11)
    ax2.set_title(r'Value function $V^*(e) = e^T P e$' + '\nLQR trajectories follow the gradient', fontsize=10)
    ax2.axhline(0, color='k', lw=0.3, alpha=0.4)
    ax2.axvline(0, color='k', lw=0.3, alpha=0.4)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    _save('01_lqr_optimal.png')


# ---------------------------------------------------------------------------
# 02  Pontryagin's minimum principle
# ---------------------------------------------------------------------------

def run_02_pontryagin():
    print("\n--- 02: Pontryagin's Minimum Principle ---")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Pontryagin's Minimum Principle  —  torque-limited pendulum", fontsize=13)

    # left: theory
    ax = axes[0]
    ax.axis('off')
    text = '\n'.join([
        r'$\bf{Minimum\text{-}time\ problem:}$',
        r'$\min_{u}\ T\quad\mathrm{s.t.}\ |u| \leq u_{max},$',
        r'$\dot{x}=f(x,u),\ x(0)=[0,0],\ x(T)=[\pi,0]$',
        '',
        r'$\bf{Hamiltonian\ (}L=1\bf{):}$',
        r'$H = 1 + \lambda_1\omega + \lambda_2(-\frac{g}{l}\sin\theta - \frac{b}{ml^2}\omega + \frac{u}{ml^2})$',
        '',
        r'$\bf{Co\text{-}state\ equations:}$',
        r'$\dot{\lambda}_1 = \lambda_2\frac{g}{l}\cos\theta,\quad \dot{\lambda}_2 = -\lambda_1 + \lambda_2\frac{b}{ml^2}$',
        '',
        r'$\bf{Pontryagin\ minimum\ condition:}$',
        r'$u^*(t) = \arg\min_{|u| \leq u_{max}} H = -u_{max}\,\mathrm{sign}(\lambda_2(t))$',
        '',
        r'$\Rightarrow\quad\bf{bang\text{-}bang\ control:}$',
        r'$u^*(t) \in \{+u_{max},\ -u_{max}\}\quad\forall\,t$',
        '',
        r'Switches when $\lambda_2(t)$ changes sign.',
        r'Typically 1 switch for pendulum swing-up.',
    ])
    ax.text(0.5, 0.5, text, transform=ax.transAxes,
            ha='center', va='center', fontsize=11, linespacing=2.1,
            bbox=dict(boxstyle='round,pad=1.2', facecolor='#f7f7f7', edgecolor='#bbb'))

    # right: schematic u(t) showing bang-bang structure
    ax2 = axes[1]
    u_max = PARAMS['u_max']
    t_sw1 = 0.75
    T_bb  = 1.55

    t_plot = np.array([0, t_sw1 - 1e-4, t_sw1, T_bb])
    u_plot = np.array([u_max, u_max, -u_max, -u_max])

    ax2.step(t_plot, u_plot, where='post', color='tab:blue', lw=2.5, label='bang-bang $u^*(t)$')
    ax2.axhline( u_max, ls='--', color='k', lw=1.0, alpha=0.5, label=f'$\\pm u_{{max}}={u_max}$ Nm')
    ax2.axhline(-u_max, ls='--', color='k', lw=1.0, alpha=0.5)
    ax2.axhline(0, color='k', lw=0.4, alpha=0.4)
    ax2.axvline(t_sw1, ls=':', color='tab:red', lw=1.5, label=f'switch  ($t^* \\approx {t_sw1}$ s)')
    ax2.axvline(T_bb,  ls=':', color='tab:green', lw=1.5, label=f'arrival  ($T \\approx {T_bb}$ s)')

    ax2.fill_between([0, t_sw1],   [ u_max,  u_max],  0, alpha=0.12, color='tab:blue')
    ax2.fill_between([t_sw1, T_bb],[-u_max, -u_max],  0, alpha=0.12, color='tab:orange')

    ax2.text(t_sw1/2,       u_max*0.55, '+$u_{max}$\n(accelerate)', ha='center', fontsize=10, color='tab:blue')
    ax2.text((t_sw1+T_bb)/2,-u_max*0.55, '$-u_{max}$\n(decelerate)', ha='center', fontsize=10, color='tab:orange')

    ax2.set_xlim(-0.05, T_bb + 0.15)
    ax2.set_ylim(-u_max * 1.35, u_max * 1.35)
    ax2.set_xlabel('t  (s)', fontsize=11)
    ax2.set_ylabel('u  (Nm)', fontsize=11)
    ax2.set_title('Bang-bang control structure\n(1-switch schematic for minimum time)', fontsize=10)
    ax2.legend(fontsize=9, loc='lower right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _save('02_pontryagin.png')


# ---------------------------------------------------------------------------
# 03  Minimum-time trajectory
# ---------------------------------------------------------------------------

def run_03_min_time():
    print('\n--- 03: Minimum-Time Trajectory ---')

    x0 = [0.0, 0.0]
    print('  Optimising bang-bang switches...')
    T_opt, sw_opt, signs_opt = _find_min_time(x0, verbose=True)
    print(f'  T* = {T_opt:.4f} s   switches = {sw_opt}   signs = {signs_opt}')

    # Reconstruct full trajectory
    t_bb, x_bb, u_bb = _reconstruct_bang_bang(sw_opt, signs_opt, T_opt, x0, PARAMS)
    E_bb = np.array([total_energy(x_bb[:, i], PARAMS) for i in range(len(t_bb))])
    E_d  = 2 * PARAMS['m'] * PARAMS['g'] * PARAMS['l']

    # ES+LQR reference
    ctrl_es = _make_es_lqr()
    t_es, x_es, u_es = _simulate(x0, ctrl_es, t_final=15.0, n=3000)
    E_es  = np.array([total_energy(x_es[:, i], PARAMS) for i in range(len(t_es))])
    # find when ES+LQR reaches upright
    reached_es = np.where(
        (np.abs(np.array([wrap_angle(x_es[0,i]-np.pi) for i in range(len(t_es))])) < 0.05)
        & (np.abs(x_es[1]) < 0.2)
    )[0]
    T_es_reach = t_es[reached_es[0]] if len(reached_es) else t_es[-1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f'Minimum-Time Bang-Bang Trajectory\n'
        f'$T^* = {T_opt:.3f}$ s   (signs: {signs_opt})   '
        f'vs ES+LQR reach time ≈ {T_es_reach:.1f} s',
        fontsize=12
    )

    # θ(t)
    ax = axes[0, 0]
    ax.plot(t_bb, x_bb[0], 'b-', lw=2.0, label=f'optimal  (T*={T_opt:.2f}s)')
    ax.plot(t_es, x_es[0], 'g-', lw=1.2, alpha=0.7, label='ES+LQR reference')
    ax.axhline(np.pi, ls='--', color='gray', lw=1.0, alpha=0.7)
    ax.text(0.01, np.pi+0.08, 'target θ=π', fontsize=8, color='gray',
            transform=ax.get_yaxis_transform())
    for sw in sw_opt:
        ax.axvline(sw, ls=':', color='tab:red', lw=1.2, alpha=0.7)
    ax.set_xlabel('t  (s)'); ax.set_ylabel('θ  (rad)')
    ax.set_title('θ(t)', fontsize=10)
    ax.set_yticks([0, np.pi/2, np.pi])
    ax.set_yticklabels(['0', 'π/2', 'π'])
    ax.set_xlim(left=-0.05)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # ω(t)
    ax = axes[0, 1]
    ax.plot(t_bb, x_bb[1], 'b-', lw=2.0, label='optimal')
    ax.plot(t_es, x_es[1], 'g-', lw=1.2, alpha=0.7, label='ES+LQR')
    ax.axhline(0, color='k', lw=0.4, alpha=0.4)
    for sw in sw_opt:
        ax.axvline(sw, ls=':', color='tab:red', lw=1.2, alpha=0.7)
    ax.set_xlabel('t  (s)'); ax.set_ylabel('ω  (rad/s)')
    ax.set_title('ω(t)', fontsize=10)
    ax.set_xlim(left=-0.05)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # u(t)
    ax = axes[1, 0]
    ax.step(t_bb, u_bb, where='post', color='tab:blue', lw=2.0, label='optimal (bang-bang)')
    ax.plot(t_es, u_es, color='tab:green', lw=1.0, alpha=0.7, label='ES+LQR')
    ax.axhline( PARAMS['u_max'], ls='--', color='k', lw=0.8, alpha=0.5)
    ax.axhline(-PARAMS['u_max'], ls='--', color='k', lw=0.8, alpha=0.5,
               label=f'$\\pm {PARAMS["u_max"]}$ Nm')
    ax.axhline(0, color='k', lw=0.4, alpha=0.4)
    for sw in sw_opt:
        ax.axvline(sw, ls=':', color='tab:red', lw=1.2, alpha=0.7, label='switch' if sw == sw_opt[0] else '')
    ax.set_xlabel('t  (s)'); ax.set_ylabel('u  (Nm)')
    ax.set_title('Control torque u(t)', fontsize=10)
    ax.set_xlim(left=-0.05)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # E(t)
    ax = axes[1, 1]
    ax.plot(t_bb, E_bb, 'b-', lw=2.0, label='optimal')
    ax.plot(t_es, E_es, 'g-', lw=1.2, alpha=0.7, label='ES+LQR')
    ax.axhline(E_d, ls='--', color='red', lw=1.2, label=f'$E_d$ = {E_d:.1f} J')
    for sw in sw_opt:
        ax.axvline(sw, ls=':', color='tab:red', lw=1.2, alpha=0.7)
    ax.set_xlabel('t  (s)'); ax.set_ylabel('E  (J)')
    ax.set_title('Energy E(t)', fontsize=10)
    ax.set_xlim(left=-0.05)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save('03_min_time.png')

    return T_opt, sw_opt, signs_opt, t_bb, x_bb, u_bb


# ---------------------------------------------------------------------------
# 04  Comparison
# ---------------------------------------------------------------------------

def run_04_comparison(T_opt, sw_opt, signs_opt, t_bb, x_bb, u_bb):
    print('\n--- 04: Comparison ---')

    x0 = [0.0, 0.0]
    ctrl_es = _make_es_lqr()
    t_es, x_es, u_es = _simulate(x0, ctrl_es, t_final=15.0, n=3000)

    u_max = PARAMS['u_max']

    def reach_time(t, x):
        for i in range(len(t)):
            if abs(wrap_angle(x[0,i]-np.pi)) < 0.05 and abs(x[1,i]) < 0.2:
                return t[i]
        return t[-1]

    T_es_reach = reach_time(t_es, x_es)
    T_bb_reach = reach_time(t_bb, x_bb)

    int_u2_es = np.trapz(u_es**2, t_es)
    int_u2_bb = np.trapz(u_bb**2, t_bb)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Optimal vs Energy-Shaping+LQR  —  Performance Comparison', fontsize=13)

    # left: phase plane
    ax = axes[0]
    ax.plot(x_es[0], x_es[1], 'g-', lw=1.5, alpha=0.8, label='ES+LQR')
    ax.plot(x_bb[0], x_bb[1], 'b-', lw=2.0, alpha=0.9, label='min-time optimal')
    for sw in sw_opt:
        idx = np.searchsorted(t_bb, sw)
        ax.plot(x_bb[0, idx], x_bb[1, idx], 'rs', ms=9, zorder=6)
    ax.plot(x_bb[0, 0],  x_bb[1, 0],  'ko', ms=9, zorder=7, label='start (0, 0)')
    ax.plot(np.pi, 0, 'g*', ms=16, zorder=7, label='target (π, 0)')
    ax.set_xlabel('θ  (rad)', fontsize=11)
    ax.set_ylabel('ω  (rad/s)', fontsize=11)
    ax.set_title('Phase plane  (red squares = bang-bang switches)', fontsize=10)
    ax.set_xlim(-0.1, np.pi + 0.1)
    ax.set_xticks([0, np.pi/2, np.pi])
    ax.set_xticklabels(['0', 'π/2', 'π'])
    ax.axhline(0, color='k', lw=0.3, alpha=0.4)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # right: metrics table + bar chart
    ax2 = axes[1]
    ax2.axis('off')

    metrics = [
        ('Time to upright  (s)',        f'{T_bb_reach:.3f}',     f'{T_es_reach:.1f}'),
        ('∫u² dt  (Nm²·s)',             f'{int_u2_bb:.1f}',      f'{int_u2_es:.1f}'),
        ('Control effort  (bang-bang?)', 'yes  (u = ±u_max)',     'no  (smooth)'),
        ('Needs offline opt.?',          'yes',                    'no'),
        ('Works from any IC?',           'only near designed IC',  'yes (by design)'),
    ]

    col_labels = ['Metric', 'Min-time optimal', 'ES + LQR']
    table = ax2.table(
        cellText=metrics,
        colLabels=col_labels,
        loc='center',
        cellLoc='left',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 2.2)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor('#d0e8ff')
            cell.set_text_props(fontweight='bold')
        elif c == 1:
            cell.set_facecolor('#ddeeff')
        elif c == 2:
            cell.set_facecolor('#ddffd4')

    ax2.set_title('Performance summary', fontsize=11, pad=15)

    plt.tight_layout()
    _save('04_comparison.png')


# ---------------------------------------------------------------------------
# 05  Animation
# ---------------------------------------------------------------------------

def run_05_animation(T_opt, sw_opt, signs_opt, t_bb, x_bb, u_bb):
    print('\n--- 05: Animation ---')
    from matplotlib.animation import FuncAnimation, PillowWriter

    u_max = PARAMS['u_max']
    E_d   = 2 * PARAMS['m'] * PARAMS['g'] * PARAMS['l']
    E_bb  = np.array([total_energy(x_bb[:, i], PARAMS) for i in range(len(t_bb))])

    fps   = 30
    t_end = T_opt * 1.05  # tiny pad after arrival
    t_a   = np.linspace(0, t_end, int(t_end * fps) + 1)
    th_a  = np.interp(t_a, t_bb, x_bb[0])
    om_a  = np.interp(t_a, t_bb, x_bb[1])
    u_a   = np.interp(t_a, t_bb, u_bb)
    E_a   = np.interp(t_a, t_bb, E_bb)
    n_frames = len(t_a)

    # switch frame indices
    sw_frames = [np.searchsorted(t_a, sw) for sw in sw_opt]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5),
                             gridspec_kw={'width_ratios': [1, 1.3, 1.3]})
    fig.suptitle(f'Minimum-Time Optimal Control  (T* = {T_opt:.3f} s)', fontsize=12)
    fig.patch.set_facecolor('white')

    # pendulum panel
    ax_p = axes[0]
    ax_p.set_aspect('equal')
    ax_p.set_xlim(-1.4, 1.4); ax_p.set_ylim(-1.4, 1.4)
    ax_p.set_xticks([]); ax_p.set_yticks([])
    ax_p.set_facecolor('#f5f5f5')
    ax_p.set_title('Pendulum', fontsize=10)
    ax_p.plot(0, 1, 'g*', ms=13, zorder=3)
    ax_p.plot(0, 0, 'k.', ms=14, zorder=6)

    trail, = ax_p.plot([], [], '-',  lw=1.5, alpha=0.25, zorder=2)
    rod,   = ax_p.plot([], [], '-',  lw=5, color='#333', solid_capstyle='round', zorder=3)
    bob,   = ax_p.plot([], [], 'o',  color='tab:blue', ms=22, zorder=4)
    info   = ax_p.text(0, -1.33, '', ha='center', fontsize=8.5,
                       bbox=dict(facecolor='white', edgecolor='none', alpha=0.9))

    # θ(t) panel
    ax_t = axes[1]
    ax_t.set_xlim(0, t_end + 0.05); ax_t.set_ylim(-0.1, 3.8)
    ax_t.axhline(np.pi, ls='--', color='gray', lw=1.0, alpha=0.7)
    ax_t.text(0.01, np.pi+0.12, 'target θ=π', fontsize=8, color='gray')
    for sw in sw_opt:
        ax_t.axvline(sw, ls=':', color='tab:red', lw=1.2, alpha=0.6)
    ax_t.set_xlabel('t  (s)'); ax_t.set_ylabel('θ  (rad)')
    ax_t.set_yticks([0, np.pi/2, np.pi])
    ax_t.set_yticklabels(['0', 'π/2', 'π'])
    ax_t.set_title('θ(t)', fontsize=10)
    ax_t.grid(True, alpha=0.3)
    th_line, = ax_t.plot([], [], 'b-', lw=1.8)
    th_dot,  = ax_t.plot([], [], 'o',  color='tab:blue', ms=6, zorder=5)

    # u(t) panel
    ax_u = axes[2]
    ax_u.set_xlim(0, t_end + 0.05); ax_u.set_ylim(-u_max*1.3, u_max*1.3)
    ax_u.axhline( u_max, ls='--', color='k', lw=0.8, alpha=0.5, label=f'±{u_max} Nm')
    ax_u.axhline(-u_max, ls='--', color='k', lw=0.8, alpha=0.5)
    ax_u.axhline(0, color='k', lw=0.4, alpha=0.4)
    for sw in sw_opt:
        ax_u.axvline(sw, ls=':', color='tab:red', lw=1.2, alpha=0.6, label='switch')
    ax_u.set_xlabel('t  (s)'); ax_u.set_ylabel('u  (Nm)')
    ax_u.set_title('Control torque u(t)  [bang-bang]', fontsize=10)
    ax_u.legend(fontsize=8)
    ax_u.grid(True, alpha=0.3)
    u_line, = ax_u.step([], [], where='post', color='tab:blue', lw=2.0)
    u_dot,  = ax_u.plot([], [], 'o', color='tab:blue', ms=6, zorder=5)

    plt.tight_layout()
    trail_len = 25

    def update(i):
        th = th_a[i]; t_ = t_a[i]; u_ = u_a[i]
        bx, by = np.sin(th), -np.cos(th)
        s = max(0, i - trail_len)
        trail.set_data(np.sin(th_a[s:i+1]), -np.cos(th_a[s:i+1]))
        rod.set_data([0, bx], [0, by])
        bob.set_data([bx], [by])
        info.set_text(f't = {t_:.3f} s     u = {u_:.0f} Nm')
        th_line.set_data(t_a[:i+1], th_a[:i+1])
        th_dot.set_data([t_], [th])
        u_line.set_data(t_a[:i+1], u_a[:i+1])
        u_dot.set_data([t_], [u_])
        return trail, rod, bob, info, th_line, th_dot, u_line, u_dot

    ani = FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps, blit=True)

    try:
        from matplotlib.animation import FFMpegWriter
        path = f'{RESULTS}/05_animation.mp4'
        ani.save(path, writer=FFMpegWriter(fps=fps, bitrate=1800))
    except Exception:
        path = f'{RESULTS}/05_animation.gif'
        ani.save(path, writer=PillowWriter(fps=fps))

    plt.close()
    print(f'  saved: {path}')


# ---------------------------------------------------------------------------

def run_optimal_control():
    print('\n' + '=' * 60)
    print('  Optimal Control')
    print('=' * 60)
    os.makedirs(RESULTS, exist_ok=True)
    run_01_lqr_optimal()
    run_02_pontryagin()
    T_opt, sw_opt, signs_opt, t_bb, x_bb, u_bb = run_03_min_time()
    run_04_comparison(T_opt, sw_opt, signs_opt, t_bb, x_bb, u_bb)
    run_05_animation(T_opt, sw_opt, signs_opt, t_bb, x_bb, u_bb)
    print(f'\nDone.  Results → {RESULTS}/')
