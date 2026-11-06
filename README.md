# ComfyUI-AdaptiveFlow

A tolerance-driven adaptive step sampler for rectified flow models
(Anima family), for ComfyUI.

## The idea

Fixed step grids spend the same number of steps everywhere: too few
where the sampling trajectory curves, wastefully many where it runs
straight. This sampler has no fixed grid. At every trial step it
computes two estimates of the step - a cheap Euler estimate and a
more accurate Heun (trapezoid) estimate - and treats their
difference as the step's local error. If the error is inside the
tolerance you set, the step is accepted (using the Heun value) and
the next step grows; if not, the step shrinks and retries. Step
count becomes an outcome of the tolerance, not an input.

## Node

**Adaptive Flow Sampler (tolerance)** - `sampling/adaptiveflow`

Inputs:

- `seed`, `cfg`, `flow_shift` - as usual for this family of packs.
- `tolerance` (default 0.01) - allowed per-step error: the RMS
  difference between the Euler and Heun estimates, relative to the
  RMS of the state. Smaller = slower and more faithful.
- `max_steps` (default 120) - budget cap; the trial budget is
  2 x max_steps. If it ever runs out mid-flight, the remainder is
  finished with plain fixed-step Euler so a complete sample always
  comes out.

Outputs: the final `latent`, and `steps_used` (accepted steps), so
you can see what a tolerance actually cost.

Notes:

- Steps are taken in the linear-shift parameter t, so a trial step
  means the same thing anywhere on the warped schedule.
- Each accepted step costs 2 model evaluations (the error estimate
  is the second one), rejected steps cost 2 as well.
- Full runs only (denoise 1.0): the walk starts at sigma = 1. The
  last step lands exactly on sigma = 0 and returns the model's x0
  prediction, as in the sibling packs.

## Install

```
cd ComfyUI/custom_nodes
git clone https://github.com/seren-alegro/ComfyUI-AdaptiveFlow
```

Or copy this folder as `ComfyUI-AdaptiveFlow` into
`ComfyUI/custom_nodes` manually. Then restart ComfyUI.

## Example workflow

`examples/adaptive_flow_basic.json` - a minimal core-nodes-only
workflow with tolerance 0.01. The checkpoint name in the loader is
a placeholder - point it at any Anima-family checkpoint, and if
your setup loads the text encoder / VAE from separate files, swap
that part of the chain accordingly. Try tolerance 0.05 and 0.002
next to it and watch `steps_used` move; note that `max_steps` is a
budget cap, not a step count - lowering it changes nothing until
the walk actually reaches it.

## Tests

`tests/test_adaptive.py` - numpy mirror of the step-size controller
(no torch needed): step counts and final errors across tolerances,
plus the trial-budget fallback path.

## License

MIT

---

## 한국어 요약

고정된 스텝 그리드 대신, 매 스텝마다 Euler 추정과 Heun 추정을 둘 다
계산해 그 차이를 그 스텝의 오차로 보고, 허용 오차(tolerance) 안에
들면 스텝을 받아들이고 키우고, 벗어나면 줄여서 재시도하는 적응형
샘플러예요. 궤적이 완만한 구간에서는 성큼성큼, 휘는 구간에서는
촘촘하게 갑니다. 스텝 수는 입력이 아니라 허용 오차의 결과가 되고,
실제로 몇 스텝을 썼는지는 `steps_used` 출력으로 확인할 수 있어요.
