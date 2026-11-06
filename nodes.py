"""Adaptive Flow - a tolerance-driven sampler for rectified flow.

One node for Anima-family rectified flow:

- Adaptive Flow Sampler walks from sigma = 1 to 0 without a fixed
  step grid. At each trial step it evaluates the model twice - an
  Euler estimate and a Heun (trapezoid) estimate of the same step -
  and uses their difference as a local error estimate. The step is
  accepted (with the Heun value, the more accurate of the two) when
  the error is within the user's tolerance; the next step size is
  grown or shrunk by the standard controller factor. Where the
  trajectory is tame the sampler strides, where it curves it slows
  down - the step count is an outcome, not an input.

Steps are taken in the linear-shift parameter t (sigma =
shift*t / (1 + (shift-1)*t)), so a trial step means the same thing
anywhere on the warped schedule. The error is the RMS difference
between the two estimates, relative to the RMS of the state.

Safety valves: a minimum step floor (below it a step is accepted
as-is rather than retried forever) and a trial budget (default
2 x max_steps trials; if it runs out mid-flight the remainder is
finished with plain fixed-step Euler so the node always returns a
complete sample).

Independent implementation of the classical embedded Euler-Heun
pair with step-size control, written for this pack.
"""

import torch

import comfy.sample
import comfy.samplers

SAFETY = 0.9          # controller safety factor
GROW_MAX = 2.5        # max step growth per accepted step
SHRINK_MIN = 0.1      # max shrink per rejected step
MIN_DT = 1e-4         # t-space step floor: accept as-is below this


def shift_sigma(t, shift):
    """sigma for parameter t on the linear-shift schedule."""
    s = float(shift)
    return s * t / (1.0 + (s - 1.0) * t)


def flow_linear_shift_sigmas(steps, shift, denoise=1.0):
    """Linear-shift sigma grid, descending, ending at exactly 0.
    Used here only as the nominal schedule handed to sample_custom
    (progress total); the adaptive walk ignores it."""
    t = torch.linspace(float(denoise), 0.0, int(steps) + 1,
                       dtype=torch.float32)
    s = float(shift)
    sigmas = s * t / (1.0 + (s - 1.0) * t)
    sigmas[-1] = 0.0
    return sigmas


def _model_wants_temporal(model):
    dm = getattr(getattr(model, "model", None), "diffusion_model", None)
    mod = type(dm).__module__ if dm is not None else ""
    return "cosmos" in mod or "anima" in mod


def _fix_latent(model, latent_image):
    """Channel-fix the samples tensor; returns (samples, expanded)."""
    samples = comfy.sample.fix_empty_latent_channels(
        model, latent_image["samples"])
    expanded = _model_wants_temporal(model) and samples.ndim == 4
    if expanded:
        samples = samples.unsqueeze(2)
    return samples, expanded


def _rms(t):
    return float(torch.sqrt(torch.mean(t.float() ** 2)))


def _adaptive_loop(model, x, extra_args, callback, tolerance,
                   max_steps, shift, stats):
    """Embedded Euler-Heun walk in t from 1 to 0.

    stats is a dict filled with accepted / trials / evals counts.
    """
    tol = float(tolerance)
    t = 1.0
    dt = -1.0 / 30.0  # first trial: one nominal 30-step interval
    max_trials = 2 * int(max_steps) + 16
    sigma = 1.0
    denoised = None
    while t > 0.0:
        if stats["trials"] >= max_trials:
            break
        dt_trial = dt
        if t + dt_trial < 0.0:
            dt_trial = -t  # land exactly on t = 0
        t_next = t + dt_trial
        s_next = shift_sigma(t_next, shift) if t_next > 0.0 else 0.0
        denoised = model(x, sigma * x.new_ones([x.shape[0]]),
                         **extra_args)
        stats["evals"] += 1
        stats["trials"] += 1
        if s_next == 0.0:
            # Final step: the x0 prediction itself is the sample.
            x = denoised
            t = 0.0
            stats["accepted"] += 1
            break
        v1 = (x - denoised) / sigma
        x_euler = x + v1 * (s_next - sigma)
        denoised2 = model(x_euler,
                          s_next * x_euler.new_ones([x_euler.shape[0]]),
                          **extra_args)
        stats["evals"] += 1
        v2 = (x_euler - denoised2) / s_next
        x_heun = x + 0.5 * (v1 + v2) * (s_next - sigma)
        err = _rms(x_heun - x_euler) / (_rms(x_heun) + 1e-6)
        factor = SAFETY * (tol / max(err, 1e-12)) ** 0.5
        if err <= tol or abs(dt_trial) <= MIN_DT:
            x = x_heun
            t = t_next
            sigma = s_next
            stats["accepted"] += 1
            if callback is not None:
                callback({"x": x, "i": stats["accepted"] - 1,
                          "sigma": float(sigma),
                          "denoised": denoised2})
            dt = dt_trial * min(max(factor, 0.2), GROW_MAX)
        else:
            dt = dt_trial * min(max(factor, SHRINK_MIN), 0.95)
    if t > 0.0:
        # Trial budget exhausted: finish the remainder with plain
        # Euler on a uniform t grid so the sample still completes.
        print(f"[AdaptiveFlow] trial budget exhausted at t={t:.4f}; "
              "finishing with fixed Euler steps")
        remaining = max(1, int(round(t * 30)))
        grid_t = [t * (1.0 - k / remaining) for k in range(remaining + 1)]
        for k in range(remaining):
            s_cur = shift_sigma(grid_t[k], shift)
            s_nxt = (shift_sigma(grid_t[k + 1], shift)
                     if grid_t[k + 1] > 0.0 else 0.0)
            denoised = model(x, s_cur * x.new_ones([x.shape[0]]),
                             **extra_args)
            stats["evals"] += 1
            if s_nxt == 0.0:
                x = denoised
            else:
                v = (x - denoised) / s_cur
                x = x + v * (s_nxt - s_cur)
            stats["accepted"] += 1
            if callback is not None:
                callback({"x": x, "i": stats["accepted"] - 1,
                          "sigma": float(s_nxt), "denoised": denoised})
    return x


class AdaptiveFlowSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0,
                                 "max": 0xffffffffffffffff}),
                "cfg": ("FLOAT", {"default": 4.5, "min": 0.0,
                                  "max": 30.0, "step": 0.1}),
                "flow_shift": ("FLOAT", {"default": 3.0, "min": 0.5,
                                         "max": 10.0, "step": 0.1}),
                "tolerance": ("FLOAT", {"default": 0.01, "min": 0.0005,
                                        "max": 0.2, "step": 0.0005}),
                "max_steps": ("INT", {"default": 120, "min": 4,
                                      "max": 500}),
            },
        }

    RETURN_TYPES = ("LATENT", "INT")
    RETURN_NAMES = ("latent", "steps_used")
    FUNCTION = "sample"
    CATEGORY = "sampling/adaptiveflow"

    def sample(self, model, positive, negative, latent_image, seed,
               cfg, flow_shift, tolerance, max_steps):
        latent_samples, expanded = _fix_latent(model, latent_image)
        noise = comfy.sample.prepare_noise(latent_samples, seed)
        if expanded:
            noise = noise.unsqueeze(2)
        # Nominal grid for sample_custom (start sigma / progress
        # total); the adaptive walk picks its own steps.
        sigmas = flow_linear_shift_sigmas(max_steps, flow_shift)
        stats = {"accepted": 0, "trials": 0, "evals": 0}

        def sampler_function(model_, x, sigs, extra_args=None,
                             callback=None, disable=None, **kwargs):
            return _adaptive_loop(model_, x, extra_args or {},
                                  callback, tolerance, max_steps,
                                  flow_shift, stats)

        sampler = comfy.samplers.KSAMPLER(sampler_function)
        samples = comfy.sample.sample_custom(
            model, noise, cfg, sampler, sigmas, positive, negative,
            latent_samples,
            noise_mask=latent_image.get("noise_mask"), callback=None,
            disable_pbar=False, seed=seed)
        print(f"[AdaptiveFlow] tol={tolerance}, shift={flow_shift}, "
              f"cfg={cfg} -> accepted={stats['accepted']}, "
              f"trials={stats['trials']}, evals={stats['evals']}")
        out = dict(latent_image)
        if expanded:
            samples = samples.squeeze(2)
        out["samples"] = samples
        return (out, stats["accepted"])


NODE_CLASS_MAPPINGS = {
    "AdaptiveFlowSampler": AdaptiveFlowSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdaptiveFlowSampler": "Adaptive Flow Sampler (tolerance)",
}
