# Council review of `claude/integration` (2026-11-09, 12:30)

A read-only review by the planning model, acting as a hostile reviewer. It reviewed integration
commit c5ab004. It ran numbers in the sandbox and changed nothing in the repo. The text below is
the review as delivered, with light formatting. What was done about each finding is in
JOURNAL.md and DECISIONS.md.

## BLOCKERS

**B1. Every training episode contains a guaranteed trap. The teacher can only learn to stop,
not to drive.**

Evidence:
- `learning/orion_learning/env.py:118-122`: the patch_soil ranges.
- `:305-307`: `patch_on = cfg.loose_patch`, so the patch is always on. The centre y is
  U(-0.4, 0.4) and the radius r is U(0.3, 0.6).

What the numbers show:
- The patch is never passable. Out of 5000 patch soils sampled at 14 kg, 9.81 and beta 0.5,
  peak steady DP/W > 0 in **0.0 %**. The median z_target at s = 0.5 is 58 mm, and belly-out is
  80 mm.
- The patch is always crossed. The wheel track is ±0.26 m, and |cy| - 0.26 < r always holds,
  so the straight path crosses the patch in 100 % of episodes.
- The main soil is mostly impassable too. With the SPEC ranges over 20k samples, randomising
  g, m, beta and the multipliers, peak DP/W is > 0 in 32 %, ≥ 0.05 in 23 % and ≥ 0.1 in 17 %.
  Earth-only, those are 19 % and 13 %.
- On the SPEC distribution, B0 succeeds in 8 % of episodes, bellies out in 81 % and is hard
  stuck in 10 %. A constant 0.3 m/s command gives the same result.
- With the patch removed, B0 rises to 28 % success and 59 % belly-out.

Why it matters:
- The reward-optimal teacher is "stop immediately". The progress term is at most 10 × 1.5 m,
  against -20 for belly-out plus the forfeited step penalties.
- The student distils a latent for a policy that never has to drive.
- The B2-vs-student comparison (the F6 gate) becomes meaningless.
- The critic's labels are almost all 1.

Fix:
1. Sample the patch soil from the main draw with multiplicative degradation: k_phi ×
   LogU(0.4, 0.9), a_s × U(1, 2), K × U(1, 1.5). Don't use fixed loose-end ranges.
2. Make patch_on ~ Bernoulli(0.5), and widen patch y to U(-0.9, 0.9), so the path crosses the
   patch in about 50 % of episodes.
3. Use mixture rejection sampling on analytic peak DP/W at nominal load: 70 % of episodes need
   DP/W ≥ 0.03 and 30 % are unrestricted. Rejecting everything below 0.05 makes B0 succeed in
   91 % of episodes, which is too easy.
4. Pre-register a target band of 40-60 % B0-completable episodes, and log that fraction every
   training iteration.

Curriculum: start the rejection threshold at 0.08 and anneal it to 0.0 over the first third of
training. Keep the SPEC ranges as the evaluation distribution.

**B2. The history convention differs between training and deployment, in both time alignment
and action meaning.**

Training:
- `learning/orion_learning/history.py:235-241, 263-268`: row k = [obs_k, a_(k-1)]. It calls
  push(obs_t, a_(t-1)) and then runs the adapter.
- `student.py:262-276`: the action column holds the raw tanh output.

Runtime:
- `orion_policy/core.py:231`: it feeds history.array() before pushing.
- `:247`: it pushes (obs_t, the executed command).

So the adapter never sees the current observation on the rover. The action columns also differ
by up to 0.8 during acceleration. OBS_SPEC_HASH cannot catch this, because it covers only the
layout.

Fix:
- Make history.py the single implementation.
- Use one convention: row = [obs_k, executed a_(k-1)], pushed before inference. In the env,
  "executed" is st['cmd'] in action units.
- Add a semantics string to the hash: the history convention, the confidence-formula id and
  the low-pass tau.
- Add a golden test proving both code paths are bit-identical.

**B3. There are two different escape libraries.**
- `baselines.py:81-86` has 4 manoeuvres. These are used for the critic labels and in the sim.
- `orion_policy/sequences.py:360-368` has one 7.5 s sequence.
- The sim B0 escapes inside the policy, while the rover B0 relies on orion_safety
  FLAG_STUCK → escape.

So sim B0 ≠ hardware B0, and rho is labelled under one escape library while the rover runs the
other.

Fix:
- Create one `escape_library.py` in orion_learning, and import it from both sides.
- Keep B0 in one place.
- Add an identity test.

## MAJOR

- **M1. The sim premise is a knife edge on a guessed parameter.** With loose a_s = 0.5, B0
  succeeds 100 % in every cell (speed drops of -11 % and -16 %). With a_s = 1.0, B0 is 100 %
  hard stuck in both loose cells. Report the result as a sensitivity curve, call the sim
  uninformative on H1, and stop quoting the -17 % figure.
- **M2. Entrapment labels differ between the sim and the rover.**
  - In the sim, stuck means < 0.02 m/s for 10 s, and belly-out means z ≥ 0.08.
  - On the rover, it is FLAG_STUCK for 3 s followed by a 20 s window, and analysis labels from
    the HALT time, about 23 s after motion stops.
  - Fix: align both on the onset of the stuck episode.
- **M3. The power figures are for a different test.** The quoted 0.49 and 0.85 are for a
  two-sided Fisher at 0.05. The implemented rule needs rise ≥ 0.25 and p < 0.01/0.04, which
  gives power 0.34 at stage 1 and 0.73 at stage 2 (0.89 without the gate). The design is
  over-conservative: false-GO is about 0.001-0.009. Fix: either drop the gate (keep it as a
  descriptive threshold) or raise stage 2 to about 50 per cell. Also say that heavy-loose
  changes two factors, and pre-register the SAFETY outcome.
- **M4. The F7 replay harness is not wired in.** No test checks the runtime estimator against
  the env's truth.csv, and no test runs env bags through ObservationBuilder.
- **M5. The effort residual differs between env and runtime.** The env uses |effort| with a
  scalar mean and updates every step. The runtime uses d·I per wheel and holds the residual
  outside the informative window. Fix: share EffortModel and the mask.
- **M6. A clock offset fails silently.** The terrain node compares camera header stamps with
  the node clock using a 0.25 s timeout. An offset larger than that leaves EXTERNAL
  permanently stale, with no error. Fix: warn on a stamp-vs-receipt gap > 0.1 s, and have the
  trial runner refuse to start unless the source is EXTERNAL.
- **M7. With the confidence formula plus a lost fix, both stuck rules go blind.** Today c =
  exp(-t/3)(1 - |v_wheel - v_x|/0.5). Fix: c = exp(-t/3)·clip(1 - std/0.1, 0, 1), changed in
  the env and the runtime together. The speed rule should not need confidence while the fix is
  fresh.
- **M8. The paper describes methods the code doesn't have:** a VariBAD belief, quantile
  gating, info-gain probes, 5 escapes and wheel-velocity actions. Fix: add a table of every
  method difference to Part 2B, and either re-scope the paper or add an uncertainty estimate
  (a critic ensemble with quantile gating).
- **M9. The adapter and critic never see escape, probe or retreat behaviour in training.** Fix:
  run CriticGate and the stuck→escape logic inside the DAgger and critic collection loops.

## MINOR

- m1. In orion_system.cpp, estop false sends ESTOP_CLEAR straight away, bypassing the
  EstopReset confirmation. It is safe only because of the priority-255 zeros. Document it or
  gate it.
- m2. The twist_mux estop lock has timeout 0.0, so a dead e-stop publisher fails open. Set it
  to 0.5 once a repeating driver exists.
- m3. Nav2 and teleop are not gated by SafetyStatus. Document this in SAFETY.md.
- m4. Add a per-episode IMU roll/pitch bias of U(-2°, 2°).
- m5. Model multi-pass: rear wheels should see k_phi × U(1, 1.5) and a reduced z0.
- m6. Normalise the effort proxy by the measured (I_stall - I_free), and randomise I_ref
  by ±30 %.
- m7. With 16 Holm-corrected tests at n = 25, per-comparison power is negligible. Say so, or
  pool.
- m8. PolicyStatus has no critic_gating field. *(Already fixed in c5ab004.)*
- m9. Gazebo IMU 23 % low: this is a simulator artefact. Add a BRINGUP step: a straight run
  against camera truth, with < 5 % error.
- m10. No Gazebo test loads an ONNX artefact.
- m11. safety_node should warn if the sim clock is still 0 after 5 s.

## Verdicts

1. **Sim premise.** The physics are credible. The training distribution is not. Adopt the B1
   fix. The kill test on real hardware is the evidence.
2. **WP4 deviations.**
   - Ratify the dig-rate sinkage, on condition that DIG_RATE becomes a per-episode
     LogU(0.03, 0.3) and its belly-out-time distribution is reported.
   - Ratify the secant-implicit integrator, adding an energy test (body work ≤ motor work).
3. **WP3.**
   - Accept the IMU and zero-sim-effort results.
   - The confidence formula is a real hole (M7).
   - Don't add turning resistance yet. Mirror the informative-window hold in the env.
   - Sharing one implementation is mandatory: orion_terrain imports orion_learning.estimator
     and history.

## Overall

1. Not acceptable as it stands, because of B1 and B2.
2. The safety chain, firmware, runner and two-stage analysis are solid. The council found no
   path where the rover moves against HALT/ESTOP.
3. The sim is uninformative on H1. Say so.
4. Fix order: B1, B2, B3, then M2, M6 and M7 before any rover trial.
5. With the fixes, this is a credible Phase-0 package. If the method differences (M8) stay
   undisclosed, reviewers will reject it.
