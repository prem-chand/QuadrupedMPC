# Code Reviews

## Claude Review

### terrain_estimator.py

**Bug — `height_per_foot` includes swing feet (line 75)**
```python
height_per_foot = np.array([f[2] for f in foot_positions])  # uses ALL 4 feet
```
Swing feet are airborne; their world-frame z is meaningless as terrain height. Should filter by `contact_schedule`, falling back to last known stance height or `NaN`.

**Bug — `_estimate_normal` missing sign flip (line 122-128)**
`_estimate_slope` enforces `normal[2] > 0` (lines 99-100), but `_estimate_normal` doesn't. Can return a downward-pointing normal. The two methods also duplicate identical cross-product logic.

**Dead code — `_contact_history` / `_max_history`**
Initialized but never written or read anywhere.

---

### stair_controller.py

**Bug — `compute_stair_target` sets absolute z (line 87)**
```python
target[2] = self.step_height + 0.05  # absolute world z = 0.1 m
```
This sets foot height to a fixed world-frame value regardless of current terrain. Should be `terrain_height + self.step_height + clearance`. As written it will fail once the robot has climbed any stairs.

**Magic `stair_start = 2.0`**
Hard-coded to the current MuJoCo scene. Should be a constructor parameter.

**`foot_positions` parameter unused in `detect_stairs`**
Misleading signature.

---

### config.py

**`SystemConfig` fields default to `None`**
If the controller accesses `cfg.terrain.enabled` without a null check this silently crashes.

---

### Priority

1. `compute_stair_target` absolute z — functional correctness
2. `_estimate_normal` sign flip
3. `height_per_foot` swing feet
4. `None` config defaults

---

## Gemini Review

### terrain_estimator.py

**Bug:** `_estimate_slope` and `_estimate_normal` only use the first 3 feet even if 4 are in stance. Sensitive to sensor noise.
**Bug:** `_estimate_normal` lacks the "upward-facing" check present in `_estimate_slope`.
**Improvement:** Use Least Squares plane fitting if 4 feet available.
**Improvement:** Implement EMA for slope_roll/pitch to prevent jitter.

---

### stair_controller.py

**Bug:** `detect_stairs` uses hardcoded world coordinates (`x > 2.0`). Fails if robot starts elsewhere.
**Bug:** `compute_stair_target` only modifies Z-height. Doesn't advance foot toward target.
**Improvement:** Use `TerrainEstimator`'s `height_per_foot` to detect stairs dynamically.

---

### config.py

**Bug:** `MPCConfig.dt` must match `sim_dt * mpc_decimation`. If someone changes one without the other, MPC horizon timing drifts.

---

## Action Items

1. **High:** Fix `compute_stair_target` to use terrain-relative height
2. **High:** Fix `height_per_foot` to filter swing feet
3. **Medium:** Add upward-facing check to `_estimate_normal`
4. **Medium:** Remove hardcoded coordinates in `detect_stairs`
5. **Low:** Add EMA smoothing to terrain estimates
6. **Low:** Fix config None defaults

---

*Reviews generated: 2026-03-27*
