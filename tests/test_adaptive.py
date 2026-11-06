"""Numpy mirror of the AdaptiveFlow controller (no torch needed).

The mirror re-implements the node's walk with the same constants
(SAFETY, GROW_MAX, SHRINK_MIN, MIN_DT, initial dt, trial budget and
the fixed-Euler fallback) against a linear ODE with a closed-form
solution:  dx/dsigma = -LAMBDA * (x - sin(2*sigma)),  LAMBDA = 1.5.

The model oracle follows the ComfyUI sampler contract used by the
node: model(x, sigma) returns the denoised prediction
denoised = x - sigma * v, with v = dx/dsigma.

Checks:
1. Tighter tolerance -> more accepted steps, smaller final error.
2. A loose tolerance finishes in fewer steps than a nominal
   30-step fixed grid on this tame field.
3. An unreachable tolerance with a tiny budget still completes
   (trial-budget fallback) and returns a finite state at t = 0.
"""
import numpy as np

SAFETY = 0.9
GROW_MAX = 2.5
SHRINK_MIN = 0.1
MIN_DT = 1e-4
SHIFT = 3.0
LAMBDA = 1.5
P = 0.36    # particular solution x_p = P sin(2s) + Q cos(2s)
Q = -0.48


def shift_sigma(t):
    return SHIFT * t / (1.0 + (SHIFT - 1.0) * t)


def velocity(x, sigma):
    return -LAMBDA * (x - np.sin(2.0 * sigma))


def denoised(x, sigma):
    return x - sigma * velocity(x, sigma)


def rms(a):
    return float(np.sqrt(np.mean(a.astype(np.float64) ** 2)))


def walk(x, tol, max_steps):
    stats = {"accepted": 0, "trials": 0, "evals": 0}
    t = 1.0
    dt = -1.0 / 30.0
    max_trials = 2 * int(max_steps) + 16
    sigma = 1.0
    while t > 0.0:
        if stats["trials"] >= max_trials:
            break
        dt_trial = dt
        if t + dt_trial < 0.0:
            dt_trial = -t
        t_next = t + dt_trial
        s_next = shift_sigma(t_next) if t_next > 0.0 else 0.0
        den = denoised(x, sigma)
        stats["evals"] += 1
        stats["trials"] += 1
        if s_next == 0.0:
            x = den
            t = 0.0
            stats["accepted"] += 1
            break
        v1 = (x - den) / sigma
        x_euler = x + v1 * (s_next - sigma)
        den2 = denoised(x_euler, s_next)
        stats["evals"] += 1
        v2 = (x_euler - den2) / s_next
        x_heun = x + 0.5 * (v1 + v2) * (s_next - sigma)
        err = rms(x_heun - x_euler) / (rms(x_heun) + 1e-6)
        factor = SAFETY * (tol / max(err, 1e-12)) ** 0.5
        if err <= tol or abs(dt_trial) <= MIN_DT:
            x = x_heun
            t = t_next
            sigma = s_next
            stats["accepted"] += 1
            dt = dt_trial * min(max(factor, 0.2), GROW_MAX)
        else:
            dt = dt_trial * min(max(factor, SHRINK_MIN), 0.95)
    if t > 0.0:  # fixed-Euler fallback, mirrors the node
        remaining = max(1, int(round(t * 30)))
        grid_t = [t * (1.0 - k / remaining) for k in range(remaining + 1)]
        for k in range(remaining):
            s_cur = shift_sigma(grid_t[k])
            s_nxt = shift_sigma(grid_t[k + 1]) if grid_t[k + 1] > 0.0 else 0.0
            den = denoised(x, s_cur)
            stats["evals"] += 1
            if s_nxt == 0.0:
                x = den
            else:
                x = x + (x - den) / s_cur * (s_nxt - s_cur)
            stats["accepted"] += 1
    return x, stats


def analytic_final(x_start):
    xp1 = P * np.sin(2.0) + Q * np.cos(2.0)
    c = (x_start - xp1) * np.exp(LAMBDA)
    return Q + c  # x_p(0) = Q


def main():
    rng = np.random.default_rng(7)
    x0 = rng.normal(size=(2, 4, 8, 8)) * 0.7
    exact = analytic_final(x0)

    results = {}
    for tol in (0.05, 0.005, 0.0005):
        xf, st = walk(x0.copy(), tol, max_steps=120)
        err = float(np.max(np.abs(xf - exact)))
        results[tol] = (st["accepted"], err)
        print(f"tol={tol:<7} accepted={st['accepted']:<4} "
              f"trials={st['trials']:<4} evals={st['evals']:<4} "
              f"final max err={err:.5f}")

    counts = [results[t][0] for t in (0.05, 0.005, 0.0005)]
    errs = [results[t][1] for t in (0.05, 0.005, 0.0005)]
    assert counts[0] < counts[1] < counts[2], "step counts must grow"
    assert errs[0] > errs[1] > errs[2], "errors must shrink"
    assert counts[0] < 30, "loose tol should beat the nominal grid"
    assert errs[2] < errs[0] / 5.0

    xf, st = walk(x0.copy(), 1e-9, max_steps=6)
    assert np.all(np.isfinite(xf)), "fallback must stay finite"
    print(f"fallback run: accepted={st['accepted']}, evals={st['evals']}, "
          "completed at t=0")
    print("all checks passed")


if __name__ == "__main__":
    main()
