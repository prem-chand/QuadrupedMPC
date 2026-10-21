---
phase: debugging
reviewers: [claude-cli]
reviewed_at: 2026-03-28
plans_reviewed: [recent-fixes]
---

# Cross-AI Plan Review — QuadrupedMPC Debug Session

## Claude CLI Review

**Summary.** These fixes address legitimate root causes (bloated inertia tensor, glacially-slow control loops, over-smoothed forces) and the result is a robot stops falling. The core diagnostics were correct. However, several parameter choices overshoot in the opposite direction, there is a silent buffer-mutation bug in the WBC that pre-dates this PR and is now amplified by the new pitch term, and the MPC horizon has been accidentally collapsed to 0.1s — a significant regression in predictive capability.

---

### Strengths

- **Inertia correction is directionally right.** The old `[0.18, 0.35, 0.3]` was clearly wrong; reducing it was necessary.
- **`mpc_counter == 0` guard** is a clean, targeted fix for the "first-step" cold-start problem.
- **`force_smooth_alpha = 0.1`** correctly restores responsiveness — the old 0.9 was almost a frozen buffer.
- **Rate increase is principled:** pushing WBC to 500 Hz is appropriate.

---

### Concerns

| Severity | Issue | Location |
|----------|-------|----------|
| **HIGH** | **MPC horizon collapsed from 0.3s → 0.1s.** `mpc_dt` changed from `0.03` to `0.01` with `horizon=10`, so prediction window is now shorter than one gait cycle (0.45s). MPC loses ability to anticipate contact transitions. | `config.py:126` |
| **HIGH** | **WBC mutates the `smoothed_forces` buffer in-place.** `forces_list` are NumPy *views* of `buffers.smoothed_forces`. Every `compute_torques()` call corrupts `smoothed_forces` with height/roll/pitch corrections. | `controller_manager.py:167`, `wbc.py:73-85` |
| **MEDIUM** | **New inertia `diag([0.1, 0.1, 0.02])` has `Izz < Ixx = Iyy`.** For a 15kg quadruped, centroidal `Izz` should be *larger* than `Ixx`, not smaller. | `config.py:124` |
| **MEDIUM** | **`stance_ratio` silently changed 0.65 → 0.80.** Swing duration shrinks from 157ms to 90ms. | `config.py:135` |
| **MEDIUM** | **Pitch feedback in WBC lacks damping.** `pitch_kp=20.0` with no `pitch_kd` term will oscillate. | `wbc.py` |

---

### Suggestions

1. **Fix buffer mutation bug:**
   ```python
   # controller_manager.py - copy, don't slice
   forces_list = [buffers.smoothed_forces[3*i:3*i+3].copy() for i in range(4)]
   ```

2. **Restore MPC horizon to ≥ 0.3s.** Either revert `mpc_dt` to `0.03` or increase `horizon` to 30 at `dt=0.01`.

3. **Verify inertia against URDF.** Extract centroidal composite inertia from Go2 URDF directly.

4. **Add `pitch_kd`** or gate pitch correction behind rate-limiting.

5. **Justify or revert `stance_ratio=0.80`** — if robot needs 80% stance, root cause should be fixed, not config.

---

**Overall risk level: MEDIUM**

The robot no longer falls, but buffer mutation and collapsed MPC horizon are latent correctness issues.
