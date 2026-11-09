# ORION: Research Context (full paper + project briefing)

> **Who this file is for:** any agent or new team member who needs to understand what ORION is, what problem it solves, why it is designed this way, and where the project stands. Part 0 is a briefing to read first. Part 1 is the complete research proposal, converted from the IEEE LaTeX source. Part 2 adds the decisions and findings made after the paper was written.
>
> **Status as of 2026-11-09:** research **proposal**. **No experimental results exist yet.** Every results table in Part 1 is intentionally empty. **Never fill a table, quote a number as a result, or describe the method as "shown to work" until real measured data exists.**
>
> **Source of truth:** the LaTeX paper (IEEEtran, 26 pages) is the formal version. If this file and `docs/SPEC.md` disagree on an *engineering* detail (interfaces, message names, trial counts), **`docs/SPEC.md` wins**, because it was revised after a later council review. See Part 2, section B, for the known differences.

---

# Part 0. Briefing for agents (read first)

## 0.1 One-paragraph summary

ORION is a four-wheel skid-steer research rover (ROS 2 Jazzy, Gazebo Harmonic, `ros2_control`) built by a student team at Maharaja Institute of Technology Mysore (author: Suhas G). The research question is how a rover should adapt **after landing**, when (a) its Earth testing is known to be **optimistic**, and (b) it has **one life**, with no resets, so getting stuck in soft soil is permanent. The core evidence is Hu *et al.* 2025 (J. Field Robotics, doi 10.1002/rob.22597). They showed that the standard Earth test for low gravity, which lightens the rover, ignores that Earth gravity also compacts the soil, so it over-predicts traction. ORION's answer is **one-life meta-reinforcement learning**:
- a context encoder infers gravity, soil and wheel health from proprioception, including motor current;
- a learned **recoverability margin** estimates how close the rover is to permanent entrapment;
- exploration is gated on a pessimistic (lower-quantile) margin;
- the wheels double as soil-probing instruments;
- an explicit escape library takes over near the edge.

## 0.2 The problem in plain words

1. Rovers die by getting stuck. Spirit was trapped in soft soil at "Troy" in 2009, and NASA abandoned extraction in 2010. The trap was not visible in planning images.
2. Earth tests lie in a known direction. Lightening the rover to lunar weight does not make the sand lunar. Earth sand stays denser and stronger, so the test predicts more traction than the rover will get.
3. So the rover must finish adapting on the surface. But standard meta-RL adapts by exploring and is evaluated with resets. On a planet, the exploratory drive that teaches you the soil is weak is the drive that traps you.
4. ORION's claim is fast adaptation **under irreversibility**, starting from a prior known to be optimistic, tested on a real rover in a sand bin with a 2×2 design that reproduces the direction of the Earth-test bias.

## 0.3 Key ideas and where they are defined

| Idea | Meaning | Defined in |
|---|---|---|
| Gravity split `c = (g_r, g_s, θ_soil, h)` | Rover-side and soil-side gravity are separate variables, so the Earth test `(g⋆, g⊕)` differs from flight `(g⋆, g⋆)` | §3.1, Def. 1 |
| Earth-test bias ΔJ | J(offset condition) − J(flight condition), expected to be > 0 (optimistic) | Def. 1 |
| Irreversible set / recoverability margin m_c(s) | Probability that the best escape behavior reaches a safe state within H seconds. **Renamed "entrapment risk" in `docs/SPEC.md`** (see Part 2B) | Defs. 2–3 |
| One-life objective + chance constraint | Return truncated at the first irreversible event; P[trap before T] ≤ δ | Eqs. (onelife), (chance) |
| Lower-quantile gating | Modes Drive / Cautious / Probe / Escape, chosen on the α-quantile of the margin under the belief | §4.4, Prop. 1 |
| Wheels as instruments | Short probe macro-actions valued by information gain minus cost and risk | §4.5 |
| Escape library Π_esc | 5 fixed behaviors (reverse, rocking, alternating pulsing, low-torque creep, pivot-out). They define recoverability and run in Escape mode | §4.6, App. C |
| 2×2 hardware design | Weight (full / reduced) × soil (compacted / loose). C = offset test, D = flight analog | §6, Fig. 2 |
| H1 / H2 / H3 | H1: the bias shows on hardware. H2: fewer traps at ≤15% speed loss. H3: the margin predicts traps ≥2 s ahead with AUC ≥ 0.8 | §7.1 |
| **Tipping-point sensing (Round 7, not yet adopted)** | Small wheel pulses → recovery rate λ → measured, gravity-invariant distance to entrapment | Part 2A |

## 0.4 What has been decided

- **Context-based meta-RL (VariBAD-style) is the core. MAML-PPO is a baseline.** On-board gradient adaptation needs risky rollouts (§8).
- **Proprioception only.** Vision is excluded so that perception gains cannot be confused with adaptation gains (§8).
- **No gradient updates on the rover.** Adaptation is a belief update only.
- **Gravity cannot be varied on Earth hardware.** Reduced gravity is simulation-only; hardware uses the 2×2 analog.
- **Results tables stay empty until measured.** The user explicitly chose "research proposal, results to come".
- **Phase 0 (3 weeks) is a de-risking gate with kill criteria** (§10, Table IX). If H1 fails, the project pivots to the fallback "one-life self-rescue" framing.
- **RL code must not come before the Phase 0 kill test** (software plan rule).
- **Ground truth on hardware comes from an overhead camera with AprilTag** (user approved, 2026-11-09). Wheel odometry is corrupted by slip.

## 0.5 Open questions and blockers

1. **Hardware list is still unknown.** Most important: can motor current be logged at ≥50 Hz? Can encoders and the IMU be logged at ≥100 Hz? The first decides H3 and probing. The second decides tipping-point sensing.
2. **Novelty is not proven.** It rests on targeted searches, not a systematic review. Phase 0 week 1 is the review.
3. **Paper TODOs:** author email, acknowledgment, JPL citation for Opportunity at "Purgatory", hardware [TBD] values in Table II, and the four new prior-art papers in Part 2C.
4. **Trial count mismatch** between the paper and SPEC.md (Part 2B).

## 0.6 Project history (how the concept evolved)

| Round | Concept | Why it changed |
|---|---|---|
| Original ARES plan | MAML-PPO on a terrain generator; "80% performance within 5 updates" | Not falsifiable; the "zero gravity" and power claims were unsupported |
| Council rating | 4.5/10. Recommended "purple cow": gravity-induced slip adaptation | Too incremental |
| Rounds 1–4 | Context-based vs MAML meta-RL for mid-drive soil and wheel faults | Solid but not distinctive |
| Round 5 | "The rover that rescues itself": sinkage detection + meta-learned self-extraction | Became the **fallback** framing |
| Round 6 | **"Earth lies about the Moon"**: one-life meta-RL with a learned recoverability margin, based on Hu et al. 2025 | **Current core** (this paper) |
| Investor teardown | Fund only the 3-week Phase 0; week-2 kill test with a non-RL controller | Adopted as Phase 0 |
| Round 7 | **Tipping-point sensing** via recovery rate after wheel pulses | Candidate; depends on a 2-day sled test (Part 2A) |

## 0.7 Rules for agents working on ORION

- Do not invent results, citations or hardware specs. Mark unknowns as `[TBD]` or `TODO`.
- Treat "passing" statuses from the old `hx` harness checks as unverified. Those checks only ran `echo`, so they always passed.
- Velocity ground truth comes from the overhead camera, not wheel odometry. Skid-steer yaw is over-reported by the diff-drive model.
- The learned controller commands wheel velocities through a forward-command controller. `diff_drive_controller` (TwistStamped on `/diff_drive_controller/cmd_vel`) is kept for the classical baseline and Nav2.
- When the paper and `docs/SPEC.md` disagree on engineering detail, follow SPEC.md and flag the difference.

---

# Part 1. The research proposal (complete text)

**Surviving the Earth-Test Bias: One-Life Meta-Reinforcement Learning with a Learned Recoverability Margin for Planetary Rover Mobility on Deformable Terrain**

Suhas G, Department of Computer Science and Engineering (AI & ML), Maharaja Institute of Technology Mysore, Karnataka, India.

*Converted from the LaTeX source. Citation numbers refer to the References list at the end of Part 1. Section numbers follow the PDF: 1 Introduction, 2 Related Work, 3 Problem, 4 Method, 5 Simulation, 6 Hardware, 7 Evaluation, 8 Design Rationale, 9 Threats, 10 Plan, 11 Conclusion, then the Appendices.*

## Abstract

Wheeled planetary rovers are qualified on Earth, yet the standard Earth-based method for imitating low gravity—reducing the rover’s mass by the gravity ratio—has recently been shown to produce systematically optimistic mobility predictions, because Earth gravity also compacts the soil the rover drives on \[1\]. The gap between Earth tests and the Moon or Mars therefore cannot be fully closed before launch. A rover must finish adapting after it lands, in a setting where it has a single life: there are no resets, and some mistakes, such as entrapment in soft soil, are permanent. Entrapment of this kind ended the drive phase of the Spirit rover in 2009–2010 \[2\]. Meta-reinforcement learning (meta-RL) offers fast adaptation to new dynamics, but existing methods adapt by exploring and are evaluated with resets, which makes their exploration unsafe exactly where it is needed.

This proposal presents ORION, a four-wheel skid-steer research rover, and a method that we call *one-life meta-RL with a learned recoverability margin*. The rover infers a latent description of the current gravity, soil and wheel-health conditions from proprioception, including motor current. It jointly meta-learns (i) a driving policy, (ii) a recoverability margin, a learned estimate of how close the rover is to a state from which no escape behavior succeeds, and (iii) the value of short probing maneuvers that use the wheels as soil instruments. Exploration during adaptation is permitted only while the margin is above a safety threshold. Training randomizes the gravity acting on the rover and the gravity that conditioned the soil as separate variables, so the policy learns not to trust an Earth-calibrated prior. We specify a simulation pipeline and a hardware protocol built around a $2\times2$ factorial design (rover weight $\times$ soil packing) that can reproduce the Earth-test bias on real hardware. We state three falsifiable hypotheses, five baselines, metrics, kill criteria and threats to validity. No results are reported; the paper is intended to be read and critiqued before data are collected.

Meta-reinforcement learning, planetary rovers, terramechanics, safe reinforcement learning, irreversibility, single-life learning, reduced gravity, sim-to-real transfer.



## Introduction

Mobility is the capability that turns a lander into an explorer, and it is also one of the most common ways a surface mission fails. In May 2009 the Spirit rover broke through a crust into loose soil at a site later named “Troy” \[3\]. Months of planning and commanded extraction attempts followed. In January 2010 NASA ended the attempts to free it, reclassifying Spirit as a stationary platform \[2\]; it fell silent later that year. In 2005 its twin, Opportunity, spent weeks embedded in a sand ripple nicknamed “Purgatory” **\[TODO: cite JPL mission report\]**. Both events share a structure that motivates this proposal. First, the soil that trapped the rover looked unremarkable in the images used to plan the drive. Second, the rover’s onboard behavior had no notion of being *near* an unrecoverable state; it continued to command the same motion until it was deeply embedded. Third, everything after that point was slow, human-in-the-loop recovery across interplanetary communication delays.

Engineers do not ignore this risk. Rover mobility is studied extensively on Earth in soil bins and analog sites, using terramechanics models such as the Bekker–Wong pressure–sinkage relations \[4, 5\]. Because gravity cannot be switched off on Earth, low-gravity mobility is usually imitated with a *gravitational offset*: the test rover is built or ballasted to weigh what the flight rover would weigh on the Moon or Mars. In 2025, Hu *et al.* \[1\] showed with an open-source, physics-based continuum simulation, correlated against NASA Glenn SLOPE laboratory tests on early VIPER prototypes, that this practice is misleading. Earth gravity also acts on the soil, making it denser and stronger than the same material under lunar or Martian gravity. The gravitational-offset test therefore over-predicts traction and under-predicts sinkage. In short, *Earth testing is biased in the optimistic direction*.

This finding has a consequence that, to our knowledge, has not been drawn out for learning-based control. If the dominant pre-flight validation method is biased in a known direction but by an unknown amount, then no controller can be fully calibrated before launch. Some of the adaptation must happen after landing, using the rover’s own experience. But after landing the rover has exactly one life. It cannot be reset to its starting pose, and a single over-confident drive into fluffy regolith can end the mission. The adaptation problem is therefore not ordinary fast adaptation; it is *fast adaptation under irreversibility, starting from a prior that is known to be optimistic*.

Meta-reinforcement learning (meta-RL) is the natural family of methods for the first half of that sentence. Gradient-based methods such as MAML \[15\] learn initial parameters that adapt in a few gradient steps. Context-based methods such as RL$^2$ \[16\], PEARL \[17\] and VariBAD \[18\] infer a latent task description from recent experience and condition the policy on it. Related adaptation architectures such as Rapid Motor Adaptation (RMA) \[19\] have enabled legged robots to adjust to new terrain within a fraction of a second. These methods share an assumption that is harmless in most benchmarks and harmful on a planet: adaptation proceeds by acting in the new environment, and evaluation averages over many episodes that each begin from a reset state. The rarely-reached tail of the experience distribution—the run that ends embedded to the axles—is averaged away.

Work on safety, irreversibility and single-life learning addresses the second half of the sentence, but largely in isolation from meta-learned adaptation and from physical terrain. Single-life RL \[22\] removes resets but does not learn which states are recoverable as a function of inferred dynamics. Recent theory on irreversible dynamics \[24\] and rollback-augmented RL \[25\] formalize the danger of non-undoable actions but are evaluated on abstract or toy domains. Constrained meta-RL \[26\] enforces constraints during adaptation but does not model irreversibility. In parallel, the rover community has developed slip and terrain learning \[10, 11, 8\], entrapment detection \[9\] and, very recently, RL-based all-terrain locomotion with real-rover transfer \[12\]. Gravity-aware learned control has been demonstrated for legged robots \[13\]. None of these, as far as our searches show, combines meta-learned adaptation, a learned model of recoverability, and an explicit treatment of the Earth-test bias.

### Proposal

We propose to study *one-life meta-RL with a learned recoverability margin* on ORION, a four-wheel skid-steer research rover built by the authors. The method has four parts:

1.  **Latent condition inference.** A context encoder estimates a posterior over the hidden conditions—gravity acting on the rover, soil state, and wheel health—from a short window of proprioceptive history that includes motor current, wheel speed, body velocity and attitude.

2.  **Recoverability margin.** A critic, meta-learned in simulation by deliberately driving into traps and testing a library of escape behaviors, predicts the probability that the rover can still return to a safe set from its current state under the inferred conditions.

3.  **Margin-gated adaptation.** The policy may explore to refine its estimate of the conditions only while the margin, including its uncertainty, stays above a threshold. Below it, the policy switches to conservative driving or to escape behaviors.

4.  **Wheels as instruments.** The action space includes short probing maneuvers, such as a single-wheel torque pulse, whose value is learned as the reduction in condition uncertainty they buy per unit of risk and energy.

Training randomizes *rover-side gravity* $g_r$ and *soil-side gravity* $g_s$ as separate variables. The classic Earth test corresponds to $g_r \approx g_{\text{target}}$, $g_s = g_\oplus$. The flight condition corresponds to $g_r = g_s = g_{\text{target}}$. A policy trained only on the former inherits the optimistic prior; a policy trained across the decoupled space can learn to be suspicious of it.

### Contributions

This paper is a research proposal. Its intended contributions, once the experiments are completed, are:

1.  A formulation of planetary rover adaptation as a *one-life, irreversibility-aware meta-RL problem*, with gravity decomposed into rover-side and soil-side components so that the Earth-test bias becomes an explicit part of the task distribution (Section 3).

2.  A method that meta-learns a recoverability margin conditioned on inferred dynamics and uses it to gate exploration and to trigger self-rescue (Section 4).

3.  A hardware protocol, with a $2\times2$ weight $\times$ soil-packing design, that can reproduce the direction of the Earth-test bias on a real rover in an ordinary sand bin and evaluate one-life adaptation under it (Section 6).

4.  A pre-registered evaluation plan: three falsifiable hypotheses, five baselines, metrics, trial counts, kill criteria and explicit threats to validity (Sections 7–10).

### What this paper does not claim

We report no results. We do not claim that aerated sand on Earth is equivalent to lunar regolith, that a skid-steer rover is representative of flight rovers with rocker-bogie or active suspension, or that the method has been tested at reduced gravity. These limitations are discussed in Section 9. Our novelty assessment is based on targeted searches rather than an exhaustive systematic review; a systematic review is the first task in the work plan (Section 10).

### Organization

Section 2 reviews related work. Section 3 formalizes the problem. Section 4 describes the method. Section 5 specifies the simulation environment and task distribution. Section 6 describes the ORION platform and hardware protocol. Section 7 gives hypotheses, baselines, metrics and the empty result tables. Section 9 discusses threats to validity. Section 10 gives the work plan and kill criteria, and Section 11 concludes.

## Background and Related Work

### Planetary Rover Mobility and Entrapment

Wheeled planetary rovers travel slowly and conservatively. Drives are planned on Earth from orbital and rover imagery, and onboard autonomy is mostly concerned with geometric hazards such as rocks and steps. Non-geometric hazards—soil that looks traversable but is not—are harder. They require inferring mechanical properties that are not visible. The Spirit and Opportunity entrapments \[2, 3\] **\[TODO: add JPL citation for Opportunity\]** are the best-known examples, but the general problem of high slip and sinkage on loose slopes and in drifts recurs across missions and in analog testing.

Two strands of work address non-geometric hazards. The first estimates terrain parameters online from wheel and body measurements. Iagnemma *et al.* \[6\] estimated cohesion and internal friction angle from wheel torque, sinkage and slip, with planetary rovers as the target application; Iagnemma and Dubowsky’s monograph \[7\] summarizes this line of work. Banerjee *et al.* \[8\] used Bayesian meta-learning (P-ALPaCA) to identify rover–terrain dynamics with a nominal model plus learned features, motivated by sand traps. These methods identify dynamics. They do not produce a control policy that reasons about irreversibility.

The second strand detects entrapment. Qiu \[9\] proposed a naive Bayes entrapment detector for planetary rovers. Detection is a necessary ingredient, but a detector that fires once the rover is already trapped arrives too late. Our recoverability margin is intended to fire *before* the boundary is crossed, and to be conditioned on the inferred soil and gravity.

### Terramechanics and Reduced-Gravity Testing

Classical terramechanics models wheel–soil interaction with semi-empirical pressure–sinkage and shear–displacement relations \[4, 5\]. Their parameters are fitted from bevameter and soil-bin measurements on Earth. Discrete-element and continuum granular models can capture effects outside the range of those fits, at much greater computational cost. Project Chrono \[31\] is an open-source multi-physics engine that includes deformable terrain models of both kinds.

Reduced gravity cannot be produced for long on Earth. Parabolic flights give seconds; offloading rigs carry part of the rover’s weight but leave the soil under full Earth gravity. The common gravitational-offset method builds a lighter test rover. Hu *et al.* \[1\] showed with Chrono-based continuum simulation, correlated against laboratory tests on early VIPER prototypes, that this method is misleading. It reproduces the rover’s weight but not the soil’s state under low gravity, and it is therefore optimistic. This is the central motivation of the present proposal. We are not aware of learning-based controllers that treat this bias as an explicit axis of their training distribution.

### Learning-Based Rover and Off-Road Control

Model-free RL has recently produced all-terrain locomotion for a planetary rover with actively articulated suspension. Bouton *et al.* \[12\] trained terrain-specialist agents in a terramechanics simulator, consolidated them into a single policy, and transferred it zero-shot to the physical ERNEST rover. Their work sets a realism bar for simulation and transfer; it does not vary gravity or meta-learn adaptation. Omi *et al.* \[10\] studied recurrent TD3 variants under evolving hidden wheel-slip disturbances on a simulated differential-drive rover, which is the closest learning-based competitor to our slip-adaptation component. MAGIC-VFM \[11\] meta-learns a residual dynamics model from visual foundation-model features and adapts its last layer online inside a composite adaptive controller, demonstrated on tracked and car-like vehicles with varying slip and slope. Stark *et al.* \[14\] present a model-based adaptive framework for planetary quadrupeds with field validation. Arm *et al.* \[13\] trained legged locomotion controllers with gravity-scaled, power-optimized rewards and validated them on hardware in simulated lunar gravity using a spring offload; their work shows that gravity is a meaningful axis for learned control, but for legs and without the soil-side effect.

### Meta-Reinforcement Learning

Meta-RL trains on a distribution of tasks so that a new task from the same distribution can be solved with little experience. Gradient-based methods, with MAML \[15\] as the canonical example, learn an initialization from which a few policy-gradient steps on the new task suffice. Recurrent or context-based methods treat adaptation as inference. RL$^2$ \[16\] uses a recurrent policy whose hidden state accumulates task information. PEARL \[17\] learns a probabilistic context variable with off-policy training. VariBAD \[18\] approximates Bayes-adaptive behavior by meta-learning a variational posterior over the task and conditioning the policy on that posterior, including its uncertainty. That property is central to our gating mechanism.

In robotics, Nagabandi *et al.* \[20\] meta-learned dynamics models that adapt online to changes such as a damaged leg or a new terrain. Al-Shedivat *et al.* \[21\] studied meta-learning for continuous adaptation in non-stationary and competitive environments. RMA \[19\] is not meta-RL in the strict sense. It trains a base policy with privileged environment information, then distills an adaptation module that estimates that information from proprioceptive history. It is, however, a strong practical baseline for terrain adaptation, and we include it.

### Safety, Irreversibility and Single-Life Learning

Constrained Markov decision processes \[27\] and constrained policy optimization \[28\] formalize safety as expected-cost constraints, which is not the same as avoiding a single irreversible event. Leave No Trace \[23\] learns a reset policy alongside the forward policy and aborts when the reset value predicts that the agent cannot return. It is the closest antecedent of our recoverability margin, but it is not meta-learned across dynamics and assumes a single environment. Single-life RL \[22\] formalizes completing a novel task in one episode without resets using prior experience, and proposes Q-weighted adversarial learning (QWALE) to steer back toward familiar states. Plaut *et al.* \[24\] give regret guarantees under irreversible dynamics when the agent can ask a mentor for help. Sorstkins *et al.* \[25\] add a learned reversibility signal and selective rollback to value-based RL. Cho and Sun \[26\] propose constrained meta-RL with differentiable convex programming to maintain safety constraints during adaptation.

### Gap Analysis

Table 1 positions the proposal against the closest work. The gap is not any single ingredient; each exists. The gap is their combination in a form that addresses a documented failure mode of a real engineering process.

|                              |                     |                 |           |              |             |             |
|:-----------------------------|:-------------------:|:---------------:|:---------:|:------------:|:-----------:|:-----------:|
| Work                         |    Meta-learned     | Irreversibility |  Gravity  |  Soil-side   | Mid-episode |    Real     |
|                              |     adaptation      |     modeled     |  varied   | gravity bias |   change    |  hardware   |
| Hu *et al.* 2025 \[1\]       |         no          |       no        |    yes    |     yes      |     no      |  lab tests  |
| Bouton *et al.* 2026 \[12\]  |         no          |       no        |    no     |      no      | not stated  |     yes     |
| Omi *et al.* 2023 \[10\]     | partial (recurrent) |       no        |    no     |      no      |     yes     |     no      |
| MAGIC-VFM 2024 \[11\]        |     yes (model)     |       no        |    no     |      no      | not stated  |     yes     |
| Arm *et al.* 2025 \[13\]     |         no          |       no        |    yes    |      no      |     no      |     yes     |
| Banerjee *et al.* 2020 \[8\] |     yes (model)     |       no        |    no     |      no      | not stated  | not stated  |
| RMA 2021 \[19\]              |       partial       |       no        |    no     |      no      |     yes     |     yes     |
| Single-life RL 2022 \[22\]   |   no (prior data)   |     partial     |    no     |      no      |     n/a     | not stated  |
| Leave No Trace 2018 \[23\]   |         no          |       yes       |    no     |      no      |     no      |     no      |
| Plaut *et al.* 2026 \[24\]   |         no          |  yes (theory)   |    no     |      no      |     n/a     |     no      |
| **This proposal**            |         yes         |       yes       | yes (sim) |     yes      |     yes     | yes (Earth) |

Positioning against the closest prior work, based on published abstracts and papers read for this proposal. “Partial” means the aspect is addressed indirectly or in a restricted form.



## Problem Formulation

### Hidden-Condition POMDP

We model rover driving as a partially observable Markov decision process whose transition dynamics depend on a hidden condition vector $c$. At each control step $t$ the rover is in state $s_t \in \mathcal{S}$. The state includes body pose and velocity, wheel angular velocities, wheel sinkage and the local soil deformation history. The rover receives an observation $o_t \in \mathcal{O}$ and applies an action $a_t \in \mathcal{A}$. The dynamics are $$s_{t+1} \sim P(\cdot \mid s_t, a_t; c_t), \qquad o_t \sim O(\cdot \mid s_t),$$ and a reward $r_t = R(s_t, a_t, s_{t+1})$ is defined in Section 3.6. The condition vector is $$c = \bigl(g_r,\; g_s,\; \boldsymbol{\theta}_{\text{soil}},\; \boldsymbol{h}\bigr),$$ where $g_r$ is the gravitational acceleration acting on the rover body, $g_s$ is the gravitational acceleration under which the soil was formed and is currently loaded, $\boldsymbol{\theta}_{\text{soil}}$ collects soil properties such as bulk density, cohesion, internal friction angle and particle size distribution, and $\boldsymbol{h} \in [0,1]^4$ is a wheel-health vector giving, per wheel, the fraction of nominal torque available. On a real planetary surface $g_r = g_s$. The separation exists to represent the Earth-test condition.

**Definition 1** (Earth-test bias). *Let $g_\oplus = 9.81~\text{m/s}^2$ and let $g_\star$ be a target gravity, e.g. $g_\text{Moon} = 1.62~\text{m/s}^2$ or $g_\text{Mars} = 3.71~\text{m/s}^2$. The *gravitational-offset condition* for $g_\star$ is $c^{\text{off}} = (g_\star, g_\oplus, \boldsymbol{\theta}_\text{soil}, \boldsymbol{h})$: the rover’s weight is scaled to the target, while the soil remains under Earth gravity. The *flight condition* is $c^{\text{flt}} = (g_\star, g_\star, \boldsymbol{\theta}_\text{soil}, \boldsymbol{h})$. The Earth-test bias of a performance measure $J$ is $\Delta J = J(c^{\text{off}}) - J(c^{\text{flt}})$.*



Hu *et al.* \[1\] provide evidence that $\Delta J > 0$ for traction-related measures: the offset condition is optimistic. Our hypothesis H1 (Section 7) tests whether a controller tuned under an Earth analog of $c^{\text{off}}$ also behaves optimistically under an Earth analog of $c^{\text{flt}}$.

### Non-Stationarity Within a Drive

During a single drive, the soil component and the wheel-health component may change. Gravity does not change within a mission. We model this as a piecewise-constant process: $$c_t = \begin{cases} c_{t-1} & \text{with probability } 1 - \lambda,\\ (g_r, g_s, \boldsymbol{\theta}'_\text{soil}, \boldsymbol{h}') & \text{with probability } \lambda, \end{cases}$$ where $(\boldsymbol{\theta}'_\text{soil}, \boldsymbol{h}')$ is drawn from a transition kernel that favors spatially plausible changes, such as firm soil to a loose drift, and monotone degradation of wheel health. In simulation, soil changes are tied to spatial patches rather than to time, so that driving into a patch causes the change. This makes the change observable only through its effect on the rover’s motion.

### Irreversible Set and Recoverability

Let $\mathcal{T} \subset \mathcal{S}$ denote the *trap set*: states from which the rover cannot make progress. A state is in $\mathcal{T}$ when, over a window of $T_\text{trap}$ seconds, body displacement remains below $d_\text{trap}$ while commanded wheel motion is non-zero. Being in $\mathcal{T}$ is not by itself permanent. We define the *irreversible set* with respect to a library of escape behaviors $\Pi_\text{esc}$:

**Definition 2** (Irreversible set). *Given a condition $c$, an escape library $\Pi_\text{esc}$, a horizon $H$ and a safe set $\mathcal{S}_\text{safe}$ (states with slip and sinkage below thresholds and non-zero progress), a state $s$ is *recoverable* if $$\max_{\pi \in \Pi_\text{esc}} \Pr\nolimits_{\pi, c}\bigl[\exists\, k \le H:\ s_{t+k} \in \mathcal{S}_\text{safe} \,\big|\, s_t = s\bigr] \ge 1 - \epsilon.$$ The irreversible set $\mathcal{I}_c$ is the set of states that are not recoverable.*



The definition is relative to $\Pi_\text{esc}$: a richer escape library shrinks $\mathcal{I}_c$. This is deliberate. It makes the irreversible set measurable in simulation and on hardware, where a state counts as irreversible if every behavior in the library fails within the horizon. On a real mission, “irreversible” would also include escape attempts so long that the mission must abandon them.

**Definition 3** (Recoverability margin). *The recoverability margin of state $s$ under condition $c$ is $$m_c(s) = \max_{\pi \in \Pi_\text{esc}} \Pr\nolimits_{\pi, c}\bigl[\exists\, k \le H:\ s_{t+k} \in \mathcal{S}_\text{safe} \,\big|\, s_t = s\bigr] \in [0,1].$$*



Because the rover does not know $c$, it must estimate a *belief-conditioned* margin $$\bar m(s, b) = \mathop{\mathrm{\mathbb{E}}}_{c \sim b}\bigl[m_c(s)\bigr], \qquad \underline{m}_\alpha(s, b) = \mathrm{Quantile}_\alpha\bigl\{m_c(s) : c \sim b\bigr\},$$ where $b$ is the rover’s belief over $c$. The lower quantile $\underline{m}_\alpha$ is what the gating mechanism uses. When the belief is wide, as after landing or after a soil change, the lower quantile is low even if the mean is comfortable. That is the formal reason a newly landed rover should be cautious.

### One-Life Objective

Standard meta-RL evaluates the expected return over tasks and episodes, $\mathop{\mathrm{\mathbb{E}}}_{c}\mathop{\mathrm{\mathbb{E}}}_{\tau \sim \pi, c}\bigl[\sum_t \gamma^t r_t\bigr]$, with resets between episodes. Under one-life deployment, the relevant quantity is the return of a single continuous run that ends either at the mission horizon $T$ or at the first entry into $\mathcal{I}_c$: $$J_\text{1L}(\pi; c) = \mathop{\mathrm{\mathbb{E}}}\left[\sum_{t=0}^{\min(T, \tau_\mathcal{I}) - 1} \gamma^t r_t\right], \qquad \tau_\mathcal{I} = \min\{t : s_t \in \mathcal{I}_{c_t}\}.
\label{eq:onelife}$$ Because every reward after $\tau_\mathcal{I}$ is lost, $J_\text{1L}$ penalizes irreversible events in proportion to the remaining mission value, not by a fixed cost. We additionally impose a chance constraint: $$\Pr\bigl[\tau_\mathcal{I} \le T\bigr] \le \delta.
\label{eq:chance}$$ Meta-training samples conditions $c \sim p_\text{train}(c)$, including resets, but the policy is trained and selected for its one-life performance.

### Observations and Actions on ORION

The observation available to the controller on hardware is $$o_t = \bigl(\boldsymbol{\omega}_t,\; \boldsymbol{i}_t,\; \boldsymbol{u}_{t-1},\; \boldsymbol{\nu}^\text{cmd}_t,\; \boldsymbol{\alpha}_t,\; \boldsymbol{\Omega}_t,\; \boldsymbol{\phi}_t,\; \hat{\boldsymbol{v}}_t\bigr),$$ with per-wheel angular velocity $\boldsymbol{\omega}_t \in \mathbb{R}^4$ from encoders, per-wheel motor current $\boldsymbol{i}_t \in \mathbb{R}^4$, the previous action $\boldsymbol{u}_{t-1}$, the commanded body twist $\boldsymbol{\nu}^\text{cmd}_t$, IMU linear acceleration $\boldsymbol{\alpha}_t$ and angular rate $\boldsymbol{\Omega}_t$, roll and pitch $\boldsymbol{\phi}_t$, and an estimate of body velocity $\hat{\boldsymbol{v}}_t$. The velocity estimate is the weakest term, since wheel odometry is corrupted precisely by slip. Options are discussed in Section 6. Motor current is the most informative signal for soil resistance: for a DC motor, torque is approximately proportional to armature current, so a rising current at falling wheel progress is a direct symptom of sinkage or bulldozing resistance.

The action is a vector of four wheel angular-velocity set-points, $\boldsymbol{u}_t \in [-\omega_\text{max}, \omega_\text{max}]^4$, executed by the existing `ros2_control` velocity interfaces at 100 Hz. The policy runs at 20 Hz. The action space is augmented with a discrete set of macro-actions: probe pulses (Section 4.5) and escape behaviors (Section 4.6).

### Reward

The task reward tracks a commanded heading and speed along a route, with penalties for energy and for excessive slip: $$r_t = -\bigl\|\boldsymbol{\nu}_t - \boldsymbol{\nu}^\text{cmd}_t\bigr\|_W \;-\; \beta_E \sum_{j=1}^{4} |\tau_{j,t}\,\omega_{j,t}|\,\Delta t \;-\; \beta_s \sum_{j=1}^{4} \max\bigl(0,\, \sigma_{j,t} - \bar\sigma\bigr) \;+\; \beta_p\, \Delta x_t,$$ where $\boldsymbol{\nu}_t$ is the measured body twist, $\tau_{j,t}$ and $\omega_{j,t}$ are wheel torque and speed, $\sigma_{j,t}$ is the longitudinal slip ratio of wheel $j$, $\bar\sigma$ is a tolerated slip level, and $\Delta x_t$ is progress along the route. The weights $\beta_E$, $\beta_s$, $\beta_p$ and the matrix $W$ are fixed before the main experiments (Section 7). There is no explicit trap penalty. Irreversible events are handled by the one-life truncation in (eq:onelife) and the constraint (eq:chance).

## Method

### Overview

Figure 1 shows the architecture. A context encoder maintains a variational posterior $q_\psi(z \mid h_t)$ over a latent condition variable $z$, given the history $h_t = (o_{0:t}, \boldsymbol{u}_{0:t-1})$. A margin network $M_\xi(s, z)$ predicts the recoverability margin. A driving policy $\pi_\theta(\boldsymbol{u} \mid o_t, q_\psi)$ and a mode selector act on the posterior and the margin. A probe module values short probing maneuvers. All components are meta-trained in simulation across the condition distribution, and only the encoder’s posterior update runs during deployment; no gradient steps are taken on the rover.

**Figure 1. Proposed architecture.**

```
 Proprioception ──► Context encoder q_ψ(z|h_t) ──(μ,Σ)──► Margin network M_ξ(s,z) → lower quantile m_α
 (ω, motor current i,      │            │                              │ margin quantile
  IMU, v̂)                  │            ▼                              ▼
       ▲                   │     Driving policy π_θ(u|o,q_ψ) ──u──► Mode selector ──► Wheel set-points
       │                   ▼            ▲                     (drive / cautious /    (ros2_control, 100 Hz)
       │            Probe value V_probe ┘                      probe / escape)            │
       │                                       Escape library Π_esc ──┘                   │
       └──────────────────────── rover + soil (physical loop) ◄───────────────────────────┘
```
The encoder posterior (mean μ, covariance Σ) feeds the policy, the margin network and the probe module. The mode selector gates exploration on the lower quantile of the margin.

### Context Encoder

The encoder follows the variational Bayes-adaptive approach of VariBAD \[18\]. A recurrent network (a single-layer GRU with 128 units, to be tuned) consumes $(o_t, \boldsymbol{u}_{t-1}, r_{t-1})$ and outputs the parameters of a diagonal Gaussian $q_\psi(z \mid h_t) = \mathcal{N}(\mu_t, \mathrm{diag}(\sigma^2_t))$ with $z \in \mathbb{R}^{d_z}$, $d_z = 8$. During meta-training the encoder is trained with a decoder that reconstructs future transitions and, as an auxiliary signal available only in simulation, the true condition $c$: $$\mathcal{L}_\text{enc} = \mathop{\mathrm{\mathbb{E}}}\Bigl[\sum_{k=1}^{K} \log p_\varphi\bigl(s_{t+k} \mid s_{t+k-1}, \boldsymbol{u}_{t+k-1}, z\bigr) + \lambda_c \log p_\varphi\bigl(c \mid z\bigr)\Bigr] - \beta_\text{KL}\,\mathrm{KL}\bigl(q_\psi(z \mid h_t)\,\|\,q_\psi(z \mid h_{t-1})\bigr).
\label{eq:enc}$$ The KL term uses the previous posterior as the prior, as in VariBAD, which encourages smooth belief updates. To allow fast re-adaptation after a soil change, we add a learned *forgetting gate*: when the decoder’s one-step prediction error rises above a learned threshold, the prior is widened by a factor $\kappa > 1$ before the update. This is a soft change-point mechanism. Its benefit is tested by ablation (Section 7).

The auxiliary term $\lambda_c \log p_\varphi(c \mid z)$ uses privileged simulator information in the spirit of RMA \[19\]. It is used only in training. The weight $\lambda_c$ is annealed toward zero so that $z$ is not forced to encode quantities that are unobservable from proprioception, such as $g_s$ when the rover is stationary.

### Recoverability Margin Network

The margin network $M_\xi(s, z) \in [0,1]$ is trained to predict the definition in Section 3, using labels generated in simulation by *escape rollouts*:

1.  During meta-training, the driving policy and a deliberately aggressive data-collection policy drive into hazardous regions, so that states near and inside $\mathcal{T}$ are visited often.

2.  At a sampled subset of states $s$, the simulator state is checkpointed. Each escape behavior in $\Pi_\text{esc}$ is executed from the checkpoint for $H$ steps under the true condition $c$.

3.  The label is $y(s, c) = \mathbb{1}\bigl[\exists\, \pi \in \Pi_\text{esc}: \text{rollout reaches } \mathcal{S}_\text{safe}\bigr]$. With $n$ stochastic repeats, the empirical fraction approximates $m_c(s)$.

$M_\xi$ is trained with binary cross-entropy against $y$, using as input the current state features available on the rover and a sample $z \sim q_\psi(z \mid h_t)$: $$\mathcal{L}_\text{marg} = -\mathop{\mathrm{\mathbb{E}}}\bigl[y \log M_\xi(s, z) + (1 - y) \log\bigl(1 - M_\xi(s, z)\bigr)\bigr].$$ At deployment, the belief-conditioned lower quantile $\underline{m}_\alpha(s, b)$ is estimated by Monte Carlo: draw $N_z$ samples $z^{(i)} \sim q_\psi$, evaluate $M_\xi(s, z^{(i)})$, and take the empirical $\alpha$-quantile. With $N_z = 32$ and a small MLP this costs well under a millisecond on the rover’s computer.

Two details matter. First, the escape rollouts make labels expensive: each labeled state costs $|\Pi_\text{esc}| \cdot n \cdot H$ simulation steps. We therefore label states with importance sampling toward the margin boundary, and fit a calibration layer on held-out conditions. Second, $M_\xi$ is evaluated as a classifier of *future* irreversibility, which is how hypothesis H3 is stated: it must predict entry into $\mathcal{I}$ at least $\Delta t_\text{lead}$ seconds before it happens.

### Margin-Gated Adaptation

The mode selector chooses among four modes using two thresholds $0 < m_\text{low} < m_\text{high} < 1$ on $\underline{m}_\alpha$:

- **Drive** ($\underline{m}_\alpha \ge m_\text{high}$): the policy acts freely, including information-seeking behavior learned in meta-training.

- **Cautious** ($m_\text{low} \le \underline{m}_\alpha < m_\text{high}$): actions are restricted to a speed and turn-rate envelope that shrinks linearly with the margin, and probing is allowed only if its predicted margin cost is small.

- **Escape** ($\underline{m}_\alpha < m_\text{low}$, or the trap detector fires): control passes to the escape library, which selects the behavior with the highest predicted success under the current posterior.

- **Probe**: a macro-action selected from Drive or Cautious when the probe module’s value exceeds its cost (Section 4.5).

The thresholds are not hand-set. They are tuned in simulation by Lagrangian relaxation of the chance constraint (eq:chance): we maximize $J_\text{1L}$ subject to $\Pr[\tau_\mathcal{I} \le T] \le \delta$ over a grid of $(m_\text{low}, m_\text{high}, \alpha)$, following the constrained-MDP view \[27, 28\]. Because the gating acts on a lower quantile of a belief-conditioned quantity, the rover is automatically more cautious when it knows less, without a separate exploration schedule.

The policy itself is trained with PPO \[29\] on the one-life objective, with episodes truncated at the first entry into $\mathcal{I}_c$, as determined by the simulator’s escape-rollout oracle at episode end. The gate is part of the environment during policy training, so the policy learns to act well within the envelope it will actually be given.

### Wheels as Instruments: Probing

A probe is a short macro-action, 0.5–1.5 s, that excites the wheel–soil interaction in an informative way. Candidates are: (i) a single-wheel forward torque pulse with the other wheels braked; (ii) a symmetric in-place wiggle of a few degrees; (iii) a brief reverse of all wheels by one wheel radius. Each probe has an energy cost and a small risk of reducing the margin. The probe value is the expected reduction in posterior entropy over the soil-related latent dimensions, discounted by its cost: $$V_\text{probe}(k) = \mathop{\mathrm{\mathbb{E}}}\bigl[\mathcal{H}(q_\psi(z \mid h_t)) - \mathcal{H}(q_\psi(z \mid h_t, \text{probe}_k))\bigr] - \eta_E E_k - \eta_m \mathop{\mathrm{\mathbb{E}}}\bigl[\underline{m}_\alpha(s_t) - \underline{m}_\alpha(s_{t+|k|})\bigr].$$ $V_\text{probe}$ is learned as a regression from the current posterior and state to the realized quantity, using probes executed during meta-training. This makes the classical idea of estimating terrain parameters from wheel measurements \[6\] an active, decision-theoretic part of the controller.

### Escape Library

The escape library $\Pi_\text{esc}$ serves two roles: it defines recoverability (Section 3), and it is executed in Escape mode. It contains parameterized behaviors feasible for a skid-steer rover:

1.  **Reverse along own track**: drive backwards at low speed along the recorded path, where the soil is already compacted.

2.  **Rocking**: alternate short forward and reverse motions at increasing amplitude to widen the rut.

3.  **Alternating wheel pulsing**: drive left and right sides, or diagonal wheel pairs, in alternation, to reduce simultaneous bulldozing.

4.  **Low-torque creep**: minimum torque that maintains motion, to avoid digging in.

5.  **Pivot-out**: a small yaw to place the wheels on undisturbed soil before reversing.

Each behavior has two to four continuous parameters, such as amplitude, period and torque limit. In Escape mode, a small context-conditioned policy selects the behavior and its parameters from $q_\psi$. The library is deliberately simple. The research question is whether a learned margin can keep the rover out of situations where the library fails, not whether a learned escape policy can outperform these behaviors. A learned escape policy is a possible extension.

### Training Against the Earth-Test Bias

The training condition distribution $p_\text{train}(c)$ samples $g_r$ and $g_s$ independently from $\{1.62, 3.71, 9.81\}$ m/s$^2$ plus continuous jitter, and samples soil and wheel-health parameters from the ranges in Section 5. Training on this *decoupled* distribution has two effects. First, the encoder learns features that separate the rover’s weight from the soil’s state, which the gravitational-offset test conflates. Second, the margin network learns that conditions which look benign from the rover’s weight alone can still be close to irreversible because of soil state.

As a diagnostic, we also train a *coupled-offset* policy only on $g_s = g_\oplus$, the condition all Earth testing provides, and evaluate it under $g_s = g_r$. The difference in one-life performance quantifies how much of the Earth-test bias a learned controller inherits when trained the conventional way.

### Meta-Training Procedure

Algorithm (alg:train) summarizes meta-training. Encoder, margin network, probe value and policy are trained jointly but with separate optimizers, and the margin labels are refreshed periodically as the driving policy changes the visited state distribution.

1.  **Require:** condition distribution $p_\text{train}(c)$; escape library $\Pi_\text{esc}$; thresholds grid $\mathcal{G}$.

2.  Initialize encoder $\psi$, decoder $\varphi$, margin $\xi$, probe value $\zeta$, policy $\theta$.

3.  **For** each iteration:

    1.  Sample a batch of conditions $c^{(b)} \sim p_\text{train}$ with patch-based soil switches.

    2.  For each $c^{(b)}$ in parallel: roll out the gated controller for up to $T$ steps; record $(o, u, r)$ and simulator checkpoints; insert exploratory segments from an aggressive data policy with probability $p_\text{agg}$.

    3.  Select checkpoints near the predicted margin boundary; run escape rollouts; label $y$.

    4.  Truncate each rollout at the first labeled irreversible state $\tau_\mathcal{I}$.

    5.  Update $\psi,\varphi$ with $\mathcal{L}_\text{enc}$; update $\xi$ with $\mathcal{L}_\text{marg}$; update $\zeta$ from realized probe information gain.

    6.  Update $\theta$ with PPO on truncated one-life rollouts, gate in the loop.

    7.  Every $N_\text{gate}$ iterations: re-tune $(m_\text{low}, m_\text{high}, \alpha)\in\mathcal{G}$ to maximize $J_\text{1L}$ subject to the chance constraint on validation conditions.



### Deployment

On the rover, only forward passes run: the encoder update at 20 Hz, the margin quantile with $N_z$ samples, the mode selector and the policy. There are no gradient updates. Adaptation is entirely through the posterior. This removes a failure mode of gradient-based meta-RL in this setting: MAML-style adaptation needs fresh rollouts in the new condition before it can update, and on a planet those rollouts are the risk. We include MAML-PPO as a baseline precisely to measure this difference (Section 7).

## Simulation Environment

### Design Requirements

The simulator must satisfy four requirements, in tension with each other:

1.  **Soil that responds to gravity.** The soil state must depend on $g_s$, separately from the rover’s weight. Otherwise the decoupled training distribution is meaningless.

2.  **Throughput.** PPO-based meta-training typically needs on the order of $10^7$–$10^8$ environment steps; the margin labels multiply this by the escape-rollout cost.

3.  **Checkpointing.** Escape rollouts require saving and restoring the full simulator state, including soil deformation.

4.  **Fidelity to ORION.** Geometry, mass distribution, wheel radius and motor limits must match the hardware closely enough for sim-to-sim and sim-to-real checks.

No single simulator satisfies all four on a laptop-class GPU. We therefore propose a two-tier design.

### Tier 1: Fast Training Simulator

Tier 1 couples a rigid-body simulator, MuJoCo \[30\], for the chassis and wheels with a custom per-wheel terramechanics force model implemented as an external force callback. The wheel–soil model follows the Bekker–Wong formulation \[4, 5\]. For a rigid wheel of radius $r$ and width $b$ with sinkage $z_w$, the normal pressure under the wheel at angle $\vartheta$ from the vertical is $$p(\vartheta) = \left(\frac{k_c}{b} + k_\phi\right) \bigl(r(\cos\vartheta - \cos\vartheta_1)\bigr)^{n},$$ and the shear stress follows the Janosi–Hanamoto relation $$\tau(\vartheta) = \bigl(c_\text{coh} + p(\vartheta)\tan\phi\bigr)\left(1 - e^{-j(\vartheta)/K}\right),$$ with cohesion $c_\text{coh}$, internal friction angle $\phi$, shear deformation modulus $K$, and shear displacement $j(\vartheta)$ that depends on the slip ratio. Integrating over the contact patch gives drawbar pull, normal load and resistive torque. The rover’s normal load scales with $g_r$.

**The soil-side gravity problem.** The Bekker–Wong parameters $(k_c, k_\phi, n, c_\text{coh}, \phi, K)$ are fitted from Earth measurements. Using them unchanged would build the gravitational-offset assumption into the training simulator, which is the error this project is meant to address. Tier 1 therefore makes the parameters functions of $g_s$ and bulk density through a correction model *fitted to Tier 2*: $$\boldsymbol{\theta}_\text{BW}(g_s, \rho_\text{bulk}, \boldsymbol{\theta}_\text{soil}) = f_\chi\bigl(g_s, \rho_\text{bulk}, \boldsymbol{\theta}_\text{soil}\bigr),$$ where $f_\chi$ is a small regression model trained on single-wheel tests run in Tier 2 across $g_s$. Tier 1 is thus a fast surrogate whose gravity dependence comes from the physics-based tier, not from Earth tests. The validity of this surrogate is itself a risk, and it is checked explicitly (Section 9).

Tier 1 also implements: (i) patch-based soil maps with smooth transitions; (ii) per-wheel torque caps and locked-wheel faults for $\boldsymbol{h}$; (iii) motor current computed from wheel torque through the motor constant and a winding-resistance model, with noise; (iv) IMU and encoder noise models; and (v) full state checkpoint and restore, including per-wheel rut depth.

### Tier 2: Physics-Based Evaluation Simulator

Tier 2 uses Project Chrono \[31\], the engine used in the study that motivates this work \[1\], with a deformable granular terrain model that represents the soil’s response to gravity directly. Tier 2 has three uses: (i) generating the single-wheel data used to fit $f_\chi$; (ii) evaluating trained policies under the offset condition and the flight condition, with $g_r$ and $g_s$ set as in the definition of Earth-test bias; and (iii) producing escape-rollout labels on a small held-out set, to measure the calibration error of the margin network trained on Tier 1 labels. Tier 2 is not used for policy training unless its throughput on the available hardware proves sufficient (Section 10).

### Tier 0: ROS 2 / Gazebo Integration Check

The existing ORION software stack runs on ROS 2 Jazzy \[32\] with Gazebo Harmonic and `ros2_control` \[33\], with a Xacro robot description, a differential-drive controller and a 2D LiDAR. Gazebo’s default contact model has no deformable soil, so it is not used for training. It serves as an integration test: the trained policy, exported to ONNX, runs as a ROS 2 node publishing wheel velocity commands through the same interfaces used on hardware. This catches interface, timing and frame errors before hardware trials.

### Rover Model

The simulated rover matches the ORION description used in the existing software (Table 2). Hardware-specific values that are not yet confirmed are marked.

| Parameter                                    | Value                                  | Source            |
|:---------------------------------------------|:---------------------------------------|:------------------|
| Chassis (L $\times$ W $\times$ H)            | $0.60 \times 0.45 \times 0.20$ m       | robot description |
| Ground clearance                             | 0.08 m                                 | robot description |
| Wheel radius / width                         | 0.12 m / 0.06 m                        | robot description |
| Wheel positions ($x$, $y$)                   | $\pm 0.25$ m, $\pm 0.26$ m             | robot description |
| Drive configuration                          | 4-wheel skid steer, independent motors | robot description |
| Model mass (chassis + wheels + mast + LiDAR) | 14.8 kg                                | robot description |
| Physical rover mass                          | \[TBD\]                                | to be measured    |
| Motor type, gear ratio, torque constant      | \[TBD\]                                | hardware audit    |
| Motor current sensing                        | \[TBD\]; required for H3               | hardware audit    |
| Encoder resolution                           | \[TBD\]                                | hardware audit    |
| IMU                                          | \[TBD\]                                | hardware audit    |
| Onboard computer                             | \[TBD\]                                | hardware audit    |

ORION model parameters. Geometric values are taken from the robot description in the project repository. Hardware values marked \[TBD\] must be measured on the physical rover.



### Condition Distribution and Task Split

Table 3 gives the planned condition ranges. Soil-parameter ranges are indicative and will be fixed after the Tier 2 calibration; they are chosen to span firm to very loose granular soil rather than to match any specific regolith.

| Variable                      | Symbol            | Training range                            | Held-out emphasis      |
|:------------------------------|:------------------|:------------------------------------------|:-----------------------|
| Rover-side gravity            | $g_r$             | $\{1.62, 3.71, 9.81\}$ m/s$^2$ $\pm 10\%$ | same set               |
| Soil-side gravity             | $g_s$             | $\{1.62, 3.71, 9.81\}$ m/s$^2$ $\pm 10\%$ | same set               |
| Bulk density (relative)       | $\rho_\text{rel}$ | 0.55–0.95 of maximum                      | extremes 0.50, 0.98    |
| Internal friction angle       | $\phi$            | $28^\circ$–$40^\circ$                     | $26^\circ$, $42^\circ$ |
| Cohesion                      | $c_\text{coh}$    | 0–1.0 kPa                                 | 1.5 kPa                |
| Slope                         | $\beta$           | $0^\circ$–$15^\circ$                      | $18^\circ$, $20^\circ$ |
| Wheel health (per wheel)      | $h_j$             | $\{1.0, 0.6, 0.3, 0\}$                    | two-wheel faults       |
| Soil patch switches per drive | —                 | 0–3                                       | 4–5                    |

Planned condition distribution. Training conditions are 35 fixed draws plus continuous jitter; 8 held-out conditions are disjoint by construction (see text).



The 35 training conditions and 8 held-out conditions are sampled so that held-out conditions lie at least partly outside the convex hull of the training conditions in the soil and slope dimensions. This tests extrapolation, not only interpolation. Leakage is checked by asserting that no held-out condition lies within a fixed normalized distance of any training condition. The split manifest, random seeds and generator code will be released with the paper.

### Throughput Gate

Before any meta-RL training, Tier 1 must reach a throughput that makes the experiment feasible on the available compute. We set the gate at 5,000 environment steps per second for 16 parallel environments on the development laptop. This is a planning target, not a measured value. If Tier 1 misses the gate, the fallback is to reduce the escape-label budget by labeling only states selected by an uncertainty criterion, and to rent GPU time for training.

## Hardware Platform and Experimental Protocol

### Platform

ORION is a four-wheel skid-steer rover with independently driven wheels, a sensor mast and a 2D LiDAR, controlled through ROS 2 Jazzy and `ros2_control`. The controller in this proposal uses only proprioception. The LiDAR is used for ground-truth logging only, as described below. The hardware audit in Phase 0 (Section 10) will fill the \[TBD\] entries of Table 2. Two properties are required:

- **Per-wheel motor current at $\ge 50$ Hz.** If the motor drivers do not report current, low-cost current-sense modules will be added in series with each motor. Without current sensing, hypothesis H3 and the probe module lose their main signal, and the study will be narrowed accordingly.

- **A stall-safe drive train.** Repeated high-current operation in sand risks overheating motors and drivers. Firmware current limits, a thermal cutoff and a physical emergency stop are mandatory before any trial.

**Ground-truth velocity.** Slip makes wheel odometry unreliable exactly when it matters. Ground-truth body velocity and displacement will be measured independently of the controller by an overhead camera tracking a fiducial marker on the rover, synchronized with ROS time. The controller’s own velocity estimate $\hat{\boldsymbol v}$ comes from fusing IMU and wheel odometry, and its error is logged as a secondary outcome.

### Test Bed

Trials take place in a sand bin, planned at about $3 \times 1.5$ m with at least 0.25 m soil depth so that the wheels cannot reach the bin floor. Two soils are planned: a fine dry quartz sand and a coarser sand–gravel mix. A third, sand–flour mix with very low bulk density is optional. Slopes up to $15^\circ$ are produced by tilting a section of the bin or by shaping the soil and checking the angle with an inclinometer.

### The $2 \times 2$ Design

The key experimental device reproduces the two conditions in the definition of Earth-test bias as closely as an Earth laboratory allows (Fig. 2).

**Figure 2. The 2×2 hardware design (rover weight × soil state).**

| Rover weight \ Soil | Compacted | Loose / aerated |
|---|---|---|
| **Full** (ballast added) | **A.** Normal Earth driving | **B.** Hard Earth case |
| **Reduced** (ballast removed) | **C.** The gravitational-offset test (analog of g_r = g⋆, g_s = g⊕) | **D.** Closest Earth analog of flight (analog of g_r = g_s = g⋆) |

Tune in C → deploy in D: that transfer is what H1 tests. Cell D is an analog, not lunar regolith (Section 9).

**Weight factor.** The full-weight condition is the rover with ballast added. The reduced-weight condition removes ballast. Because ORION cannot be made one sixth as heavy without removing essential hardware, the weight ratio will be the largest practical ratio, targeted at 0.5 or below, and reported exactly. The center of mass will be kept within a fixed tolerance between conditions, by placing ballast symmetrically.

**Soil factor.** Compacted beds are prepared by raking, leveling and a fixed number of passes of a weighted roller. Loose beds are prepared by sieving or air-fluffing the top layer to a fixed depth and leveling without compaction. Because weaker, low-density soil is the qualitative effect of reduced soil-side gravity reported in \[1\], the loose bed is used as an *analog*. It is not a reproduction of lunar conditions.

**Soil characterization.** Every prepared bed is characterized before the trial by (i) bulk density from a fixed-volume core sample and (ii) penetration resistance from a cone penetrometer at three fixed locations. Beds outside a pre-specified tolerance band are re-prepared. These measurements address the largest source of noise in soil-bin experiments, inconsistent preparation, and let the analysis use measured soil state as a covariate rather than only the nominal condition.

### Trial Procedure

Each trial is one continuous one-life run:

1.  Prepare and characterize the bed; place the rover at the start pose; record the cell, soil measurements, battery voltage and motor temperatures.

2.  Command a fixed route: straight 2 m, a $90^\circ$ arc, and a 1 m ascent of the slope section where applicable. The same route is used for every method within a cell.

3.  Run the controller under test with no human intervention until the route is completed, a trap is declared, or a time limit of 180 s is reached.

4.  A **trap** is declared when ground-truth displacement stays below 2 cm over 10 s with non-zero wheel commands. After a trap, the escape library is executed for up to 60 s. If no behavior frees the rover, the event is recorded as **irreversible** for that trial.

5.  The rover is extracted by hand if needed, the bed is re-prepared, and the next trial starts.

Method order is randomized within blocks so that bed-preparation drift and battery state are not confounded with method. The operator preparing the bed is blind to the next method where practical.

### Mid-Run Changes on Hardware

Soil changes within a run are produced spatially: the route crosses from a compacted section into a loose section at a position unknown to the controller. Wheel faults are injected in software by capping the commanded torque of one or two wheels at a random time. Both are logged with timestamps so that re-adaptation time can be measured.

### Sample Size

With four cells, five primary methods (Section 7) and a target of 20 runs per method per cell, the full study needs about 400 runs. At an estimated 10 minutes per run including bed preparation, this is about 67 hours of bin time. For H1, the planned comparison is between trap rates in cells C and D for a controller tuned in C. If the true trap rates are, for example, 10% and 40%, an exact calculation for a two-sided Fisher exact test at $\alpha = 0.05$ gives a power of about 0.49 with 20 runs per cell, 0.71 with 30 and 0.85 with 40. Twenty runs per cell is therefore under-powered for an effect of this size. For the H1 comparison we plan 40 runs in each of cells C and D, which adds 40 runs, about 7 hours of bin time. These trap rates are illustrative assumptions, not predictions, and the sample size will be revisited after the Phase 0 pilot.

### Safety

All trials have a physical emergency stop within reach of the operator, firmware current limits, a thermal cutoff and a maximum wheel speed. No trial is run unattended. Batteries are charged and stored according to the manufacturer’s guidance.

## Hypotheses, Baselines, Metrics and Result Tables

### Pre-Registered Hypotheses

**H1 (Earth-test bias reproduces on hardware).** A controller tuned in cell C (reduced weight, compacted soil) has a higher trap rate and a higher irreversible-event rate in cell D (reduced weight, loose soil) than in cell C, on the same route. *Falsified if* the difference in trap rate is not significant at $\alpha = 0.05$ over 40 runs in each of cells C and D. If H1 fails on our test bed, the Earth-test framing cannot be supported with our hardware, and the study pivots to the self-rescue framing described in Section 10.

**H2 (one-life safety at small cost).** In cell D and on the held-out simulation conditions, the proposed controller has fewer irreversible events than each baseline, with a loss of mean route speed no greater than $\Delta v_\text{max}$ relative to the fastest baseline. *Falsified if* the reduction relative to the best baseline is below $r_\text{min}$, or the speed loss exceeds $\Delta v_\text{max}$.

**H3 (early warning).** The margin network predicts entry into the irreversible set at least $\Delta t_\text{lead}$ seconds before it occurs, with area under the ROC curve at least $\text{AUC}_\text{min}$. *Falsified if* either threshold is missed on hardware trials in cells B and D.

**Freezing the thresholds.** Initial values are $r_\text{min} = 50\%$, $\Delta v_\text{max} = 15\%$, $\Delta t_\text{lead} = 2$ s and $\text{AUC}_\text{min} = 0.8$. These are first estimates. They will be revised once, after the baselines are measured in simulation and before any run of the proposed method on hardware, then frozen and published with a timestamp. Any later change will be reported as a deviation.

### Baselines

1.  **Tuned classical controller.** The existing differential-drive controller with gains and speed limits tuned by hand in cell C. This is the reference for H1 and represents current practice.

2.  **PPO with domain randomization (PPO-DR)** \[29\]: one policy trained on the full condition distribution, without a context encoder or margin.

3.  **RMA-style adaptation** \[19\]: a base policy with privileged condition input, plus an adaptation module that estimates it from history.

4.  **Recurrent off-policy RL** in the style of \[10\]: a recurrent TD3 agent, the closest published rover-slip method.

5.  **MAML-PPO** \[15\]: gradient-based meta-RL with $k$ adaptation steps, given rollouts collected under the new condition. On hardware those rollouts are themselves at risk, which is what this baseline measures.

A single-life RL baseline (QWALE \[22\]) will be added if implementation time allows. It requires a prior-data buffer and careful re-implementation, and it is budgeted as optional.

### Ablations

- **No gating**: the encoder and policy only, without the margin; this is a VariBAD-style policy.

- **Mean margin**: gating on $\bar m$ instead of the lower quantile $\underline{m}_\alpha$.

- **No probing**: probe macro-actions removed.

- **No forgetting gate**: the change-point widening in the encoder removed.

- **Coupled-offset training**: trained only on $g_s = g_\oplus$, evaluated on $g_s = g_r$.

- **No current sensing**: motor current removed from the observation.

### Metrics

- **Irreversible-event rate**: fraction of one-life runs that end in $\mathcal{I}$, the primary safety metric.

- **Trap rate**: fraction of runs with at least one trap, recovered or not.

- **Route completion** and **mean route speed**.

- **Re-adaptation time** after a mid-run change: time until speed-tracking error returns within 10% of its pre-change level.

- **Energy per meter**, from integrated electrical power.

- **Early-warning performance** of the margin: ROC-AUC and the lead-time distribution at a fixed false-alarm rate.

- **Calibration**: expected calibration error of $M_\xi$ against Tier 2 escape-rollout labels.

- **Compute**: training wall-clock time and on-board inference latency.

All simulation results will be reported over at least five training seeds, with 95% confidence intervals by bootstrap. Hardware results will report counts per cell with exact binomial intervals.

### Result Tables (Intentionally Empty)

Tables 4–8 define the results this study will produce. They are empty by design. No value will be entered that has not been measured under the protocol above.

| Cell                                      | Trap rate | Irreversible rate | Route completion | Mean speed (m/s) |
|:------------------------------------------|:---------:|:-----------------:|:----------------:|:----------------:|
| A. Full weight, compacted                 |     —     |         —         |        —         |        —         |
| B. Full weight, loose                     |     —     |         —         |        —         |        —         |
| C. Reduced weight, compacted (tuned here) |     —     |         —         |        —         |        —         |
| D. Reduced weight, loose                  |     —     |         —         |        —         |        —         |
| C vs. D, Fisher exact $p$                 |     —     |                   |                  |                  |

H1: Earth-test bias on hardware. Tuned classical controller, 20 runs in cells A and B, 40 in cells C and D. \[Pending data.\]



| Method        | Irreversible rate | Completion | Speed (m/s) | Re-adapt. time (s) | Energy (J/m) |
|:--------------|:-----------------:|:----------:|:-----------:|:------------------:|:------------:|
| PPO-DR        |         —         |     —      |      —      |         —          |      —       |
| RMA-style     |         —         |     —      |      —      |         —          |      —       |
| Recurrent TD3 |         —         |     —      |      —      |         —          |      —       |
| MAML-PPO      |         —         |     —      |      —      |         —          |      —       |
| **Proposed**  |         —         |     —      |      —      |         —          |      —       |

H2: One-life performance in simulation, held-out conditions, flight case $g_s = g_r$. Mean \[95% CI\] over 5 seeds. \[Pending data.\]



| Method          | Irreversible events / runs | Traps / runs | Mean speed (m/s) | Energy (J/m) |
|:----------------|:--------------------------:|:------------:|:----------------:|:------------:|
| Tuned classical |             —              |      —       |        —         |      —       |
| PPO-DR          |             —              |      —       |        —         |      —       |
| RMA-style       |             —              |      —       |        —         |      —       |
| MAML-PPO        |             —              |      —       |        —         |      —       |
| **Proposed**    |             —              |      —       |        —         |      —       |

H2 on hardware, cell D (reduced weight, loose soil), 20 runs per method. \[Pending data.\]



| Setting                   | ROC-AUC | Median lead time (s) | Calibration error |
|:--------------------------|:-------:|:--------------------:|:-----------------:|
| Simulation, Tier 1 labels |    —    |          —           |         —         |
| Simulation, Tier 2 labels |    —    |          —           |         —         |
| Hardware, cell B          |    —    |          —           |        n/a        |
| Hardware, cell D          |    —    |          —           |        n/a        |

H3: Early warning by the recoverability margin. \[Pending data.\]



| Variant                         | Irreversible rate | Speed (m/s) | Re-adapt. time (s) |
|:--------------------------------|:-----------------:|:-----------:|:------------------:|
| Full method                     |         —         |      —      |         —          |
| No gating                       |         —         |      —      |         —          |
| Mean margin instead of quantile |         —         |      —      |         —          |
| No probing                      |         —         |      —      |         —          |
| No forgetting gate              |         —         |      —      |         —          |
| Coupled-offset training         |         —         |      —      |         —          |
| No current sensing              |         —         |      —      |         —          |

Ablations in simulation, held-out conditions. \[Pending data.\]



### Outcomes and What Each Would Mean

Pre-registering interpretations protects against post-hoc storytelling.

- **H1 holds, H2 holds**: the central claim is supported on Earth hardware. The method mitigates a documented bias in rover qualification, within the limits of the analog.

- **H1 holds, H2 fails**: the bias is real on our test bed, but the learned margin does not beat simpler adaptation. That is still a publishable negative result about where learned recoverability helps.

- **H1 fails**: our analog does not reproduce the bias. The weight and soil manipulation is insufficient, and claims about gravity must be limited to simulation. The project continues with the self-rescue framing on loose soil and wheel faults.

- **H3 fails but H2 holds**: gating helps, but the margin is not a reliable standalone warning signal. That limits its use as an operator-facing alert.

## Design Rationale

This section explains the main design choices, and the alternatives we rejected, so that reviewers can challenge them before the experiments are run.

### Why Gate on a Lower Quantile of the Margin

Gating on the belief-mean margin $\bar m$ would let a rover with a wide, uncertain belief act as if it were in the average condition. Under the Earth-test bias, the average of an Earth-calibrated prior is exactly the wrong reference. The lower quantile gives a simple guarantee, stated under an idealized calibration assumption.

**Assumption 1** (Calibrated belief and margin). *At time $t$, the true condition satisfies $c_t \sim b_t$, and the margin network is exact: $M_\xi(s, z) = m_c(s)$ for the condition $c$ encoded by $z$.*



**Proposition 1**. *Under the assumption, if the gate admits an action in Drive or Cautious mode at state $s_t$, that is $\underline{m}_\alpha(s_t, b_t) \ge m_\text{low}$, then $$\Pr\bigl[m_{c_t}(s_t) < m_\text{low}\bigr] \le \alpha.$$*



*Proof.* By definition, $\underline{m}_\alpha$ is the $\alpha$-quantile of $m_c(s_t)$ under $c \sim b_t$. Under the assumption, $c_t$ is a draw from $b_t$, so the probability that $m_{c_t}(s_t)$ falls below its $\alpha$-quantile is at most $\alpha$. Since $m_\text{low} \le \underline{m}_\alpha$, the event $m_{c_t}(s_t) < m_\text{low}$ is contained in that event.



The proposition is deliberately modest. It holds per decision, not per mission, and it depends on calibration that the Earth-test bias is expected to violate at landing. Its purpose is to make explicit what the learned components must achieve. Calibration of the margin network is therefore a reported metric (Table 7), and the risk budget $\delta$ in (eq:chance) is enforced empirically by tuning the gate, not assumed from the proposition.

### Why Context-Based Rather Than Gradient-Based Adaptation On Board

Gradient-based meta-RL, such as MAML \[15\], adapts by collecting rollouts in the new condition and taking policy-gradient steps. On a rover this has three costs. First, the rollouts must be executed with the pre-adaptation policy, which under the Earth-test bias is the over-confident one; those are the drives most likely to end in a trap. Second, policy-gradient estimates from a handful of real rollouts are noisy, and a bad update cannot be undone if it leads into an irreversible state. Third, gradient computation on board adds compute and memory requirements to a power-limited platform. Context-based adaptation \[16, 17, 18\] updates only a belief, continuously, within a single drive, and the belief’s uncertainty is available to the gate. We keep MAML-PPO as a baseline because this argument is a hypothesis, not a fact. It may turn out that a single conservative gradient step is competitive, and the experiment will show whether it is.

### Why Proprioception Only

Vision can detect some hazardous soil before contact \[11\], and a deployed system should use it. We exclude it here for two reasons. First, the entrapment hazards that motivate this work were, by most accounts, not visually distinguishable at planning time \[3\]. Second, adding vision would make it impossible to tell whether improvements come from better perception or from better adaptation. The proposed method is complementary to visual terrain classification: a visual prior over soil type could initialize the belief $b_0$.

### Why an Explicit Escape Library

A fully learned escape policy is attractive, but it would make the definition of irreversibility depend on a learned component that changes during training, and the margin labels would drift with it. Fixing $\Pi_\text{esc}$ makes the irreversible set well defined, makes labels reproducible, and lets the hardware experiments use the same behaviors to decide whether a trap is permanent. Learning the escape policy is left for future work.

### Computational Budget for Margin Labels

The margin labels dominate the simulation cost. Each labeled state needs $|\Pi_\text{esc}| \cdot n$ escape rollouts of $H$ seconds. With $|\Pi_\text{esc}| = 5$, $n = 3$, $H = 30$ s and a 20 Hz control rate, one label costs $5 \cdot 3 \cdot 600 = 9{,}000$ control steps. Labeling $10^5$ states would cost $9 \times 10^8$ steps, or 50 hours at the Tier 1 throughput gate of 5,000 steps per second. That is too expensive to repeat at every label refresh. We therefore plan a budget of about $2 \times 10^4$ labels per refresh, about 10 hours, concentrated near the predicted margin boundary by importance sampling, and at most three refreshes. If Tier 1 misses its throughput gate, label budget is the first thing to cut, and its effect on margin calibration will be reported.

### An Illustrative Walk-Through

The following scenario is *illustrative*. It describes the intended behavior, not an observed result.

ORION begins a drive in a lunar-gravity simulation, on firm soil, with a belief that was shaped by meta-training to be suspicious of firm-looking ground. The lower-quantile margin starts moderate, so the gate allows Cautious mode. After a few meters, motor currents, slip and body acceleration agree with firm soil. The belief narrows, the margin quantile rises, and the gate opens Drive mode. At an unannounced position the route enters a loose patch. Currents rise on the front wheels while body velocity falls. The decoder’s prediction error jumps, the forgetting gate widens the prior, and the margin quantile drops sharply even before slip becomes large. The gate falls back to Cautious mode and reduces speed. The probe module estimates that a short reverse pulse would sharpen the soil estimate at low risk, so the rover executes it. The belief now favors loose soil, the margin falls toward $m_\text{low}$, and the policy steers toward the edge of the patch at low torque. If the margin had crossed $m_\text{low}$, the escape library would have taken over, starting with reversing along the rover’s own track. A baseline without the margin, by contrast, would be expected to continue at commanded speed until slip forced a trap. Whether this intended behavior actually appears is what Tables 5–7 will measure.

## Threats to Validity and Limitations

We list the threats we consider most likely to invalidate or narrow the conclusions, in order of severity, with the mitigation planned for each. This section was written by deliberately arguing against the proposal.

### Construct Validity

**T1: The training simulator can inherit the bias it is meant to remove.** Bekker–Wong parameters are fitted from Earth tests. If Tier 1 used them unchanged, the policy would learn an Earth-calibrated soil. *Mitigation:* Tier 1’s soil parameters are functions of $g_s$ fitted to the physics-based Tier 2 simulator, and every headline simulation result is reported on Tier 2. The surrogate’s error is reported as a calibration metric. If the surrogate error is large, training moves to Tier 2 at reduced scale, or the claims about gravity are restricted.

**T2: Loose sand on Earth is not low-gravity soil.** Lowering bulk density imitates one qualitative effect of low soil-side gravity, but grain contact forces in the loosened bed are still produced under Earth gravity. *Mitigation:* cell D is consistently called an *analog*; soil state is measured in every trial; and no claim of lunar or Martian performance is made from hardware data alone. The hardware experiment tests whether the *direction* of the bias appears and whether the controller handles it, not its magnitude on another planet.

**T3: The irreversible set depends on the escape library.** A trap counted as irreversible for our library might be escapable with a better behavior. *Mitigation:* the definition is explicit about this dependence; the same library is used for all methods, so comparisons are fair; and the library is reported in full.

### Internal Validity

**T4: Bed preparation drift.** Soil-bin results are notoriously sensitive to preparation. *Mitigation:* measured density and penetration resistance for every bed; tolerance bands; randomized method order within blocks; and soil state as a covariate in the analysis.

**T5: Battery state and motor temperature.** Available torque falls as the battery discharges and the motors heat. *Mitigation:* log voltage and temperature at the start of every run; impose limits outside which runs are postponed; and balance method order across battery cycles.

**T6: Threshold tuning on test data.** *Mitigation:* hypothesis thresholds and gate thresholds are fixed on simulation validation conditions and frozen before any hardware run of the proposed method.

### External Validity

**T7: Gravity is not tested on hardware.** The flagship motivation concerns the Moon and Mars, but no experiment runs at reduced gravity. *Mitigation:* gravity claims are limited to Tier 2 simulation; future work could use an offloading rig to vary rover weight continuously, although this does not address soil-side gravity.

**T8: A skid-steer rover is not a flight rover.** Spirit and Opportunity used rocker-bogie suspension; other designs use active suspension \[12\]. Skid steering also produces lateral scrubbing that is atypical for flight rovers. *Mitigation:* claims are stated for wheeled rovers on deformable soil; the method does not depend on the drive configuration except through the action space and the escape library.

**T9: Two stacked gaps.** Results pass through sim-to-sandbox and then sandbox-to-planet gaps. Only the first is measured. *Mitigation:* explicit statement; sim-to-sandbox error is reported for every metric that has both.

### Statistical and Practical Threats

**T10: Under-powered hardware comparisons.** About 400 runs is substantial lab work, and rare irreversible events make per-cell rates noisy. *Mitigation:* exact binomial intervals; extra runs for the H1 comparison; and pooling across cells only where pre-specified.

**T11: Novelty may be overstated.** Our positioning is based on targeted searches and on reading the abstracts and papers cited here, not on a systematic review. A closely related method may exist. *Mitigation:* a systematic review across robotics, machine-learning and terramechanics venues is the first task of Phase 0; contribution claims will be narrowed to whatever survives it.

**T12: The motivating result is a single study.** The Earth-test bias rests mainly on \[1\]. If later work qualifies it, the framing weakens, though the one-life recoverability method remains applicable to any deformable-terrain setting.

**T13: “One life” applies at deployment only.** Meta-training uses many simulated lives, with resets. This is standard and intended. The claim is about deployment behavior, and is stated that way throughout.

## Work Plan, Kill Criteria and Applications

### Phase 0: De-Risking (3 weeks)

Phase 0 tests the assumptions most likely to sink the project before the expensive work begins. Each step has a kill or pivot criterion.

| Week | Step                                                                                           | Kill / pivot criterion                                                                                                        |
|:-----|:-----------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------|
| 1    | Read \[1\] in full; systematic literature review across robotics, ML and terramechanics venues | Direct prior work on learned, dynamics-conditioned recoverability for rovers $\Rightarrow$ narrow the contribution claims     |
| 1    | Hardware audit: motors, drivers, encoders, IMU, computer, current sensing at $\ge 50$ Hz       | No current sensing and none can be added $\Rightarrow$ drop H3 and probing; narrow the claims                                 |
| 2    | H1 pilot with the tuned classical controller, 20 runs each in cells C and D                    | No difference in trap rate $\Rightarrow$ pivot to the self-rescue framing (loose soil and wheel faults, no Earth-test claim)  |
| 3    | Tier 1 throughput and Tier 2 single-wheel runtime on the development machine                   | Tier 1 below gate $\Rightarrow$ reduce label budget or rent compute; Tier 2 too slow $\Rightarrow$ use it for evaluation only |

Phase 0 steps and kill criteria.



### Phases 1–5

1.  **Phase 1 (weeks 4–6): Simulation.** Implement Tier 1 with the $g_s$-dependent soil surrogate fitted to Tier 2; implement the escape library; build the condition generator and the 35/8 split with leakage checks.

2.  **Phase 2 (weeks 7–9): Baselines.** Train PPO-DR, RMA-style, recurrent TD3 and MAML-PPO. Measure them on held-out conditions, then revise and freeze the hypothesis thresholds.

3.  **Phase 3 (weeks 10–13): Proposed method.** Train the encoder, margin network, probe value and gated policy; tune the gate thresholds by constrained optimization; run ablations; and evaluate on Tier 2.

4.  **Phase 4 (weeks 14–18): Hardware.** Tier 0 integration test in Gazebo; deploy as ROS 2 nodes; run the full $2\times2$ protocol with all methods.

5.  **Phase 5 (weeks 19–20): Analysis and release.** Statistical analysis as pre-registered; release the code, condition manifests, trained weights, raw logs and bed-characterization data.

The schedule assumes one primary researcher with access to the rover and a laptop-class GPU. It should be re-planned once the team size and deadline are known.

### Fallback Framing

If H1 fails in the Phase 0 pilot, the project keeps every technical component and changes its claim. It becomes *one-life self-rescue on deformable soil with wheel faults*: the rover detects approaching entrapment, probes the soil, and escapes, across soils and wheel faults it has not seen. The Earth-test bias then appears only as a simulation study. This fallback addresses the same mission problem with a smaller claim.

### Relation to the Original ARES Research Plan

This proposal refines an earlier ARES project plan for the same rover. That plan targeted MAML-PPO as the core algorithm, FOMAML and Reptile as lower-compute alternatives, PEARL and multi-task PPO as comparisons, a parameterized terrain generator over friction, slope, roughness, gravity and compliance, 35 training and 8 held-out terrain configurations, and a target of reaching 80% of scratch-policy performance within five adaptation updates. Table 10 maps that plan onto the present proposal and explains each change.

| Earlier plan                                                        | This proposal                                            | Reason                                                                                                                                              |
|:--------------------------------------------------------------------|:---------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------|
| MAML-PPO as core method                                             | Context-based meta-RL as core; MAML-PPO as baseline      | On-board gradient adaptation needs risky rollouts (Section 8) |
| FOMAML, Reptile, PEARL                                              | Dropped from the core study                              | Scope; they answer the same question as MAML-PPO or the context-based method                                                                        |
| Terrain generator (friction, slope, roughness, gravity, compliance) | Kept, with gravity split into $g_r$ and $g_s$            | Makes the Earth-test bias explicit                                                                                                                  |
| 35 train / 8 held-out                                               | Kept, with held-out conditions outside the training hull | Tests extrapolation, with a leakage check                                                                                                           |
| 80% within 5 updates                                                | Replaced by H1–H3                                        | One-life safety is the mission-relevant outcome; the earlier target was not falsifiable as stated                                                   |
| Sim-to-real analysis                                                | $2\times2$ hardware protocol                             | Gives a measurable sim-to-sandbox comparison                                                                                                        |

Mapping from the earlier ARES plan to this proposal.



### Applications Beyond Planetary Exploration

The capability of noticing that one is approaching an unrecoverable state, under uncertain dynamics and with no reset, applies wherever a ground robot works alone on deformable or changing terrain. Examples include agricultural robots on wet fields, robots for disaster response on rubble and mud, polar and desert field robots, and construction-site vehicles. In each, an operator is far away or unavailable, and getting stuck ends the task. The Earth-test-bias framing is specific to space, but the recoverability-gated adaptation is general.

## Conclusion

Rovers are qualified on Earth, and Earth testing of low-gravity mobility has been shown to be optimistic \[1\]. A rover must therefore finish adapting after it lands, with one life and no resets. This proposal frames that situation as one-life, irreversibility-aware meta-reinforcement learning. Gravity is split into rover-side and soil-side components so that the Earth-test bias becomes part of the task distribution. The proposed method meta-learns a recoverability margin conditioned on inferred dynamics, gates exploration on its lower quantile, and uses the wheels as instruments. The hardware protocol uses a $2\times2$ weight $\times$ soil-packing design that can test, on an Earth sand bin, whether the direction of the bias appears and whether the controller survives it.

We have stated what would count as failure for each hypothesis, which baselines must be beaten, and which limitations no Earth experiment can remove. The result tables are empty, and they will be filled only with measured data. We invite criticism of the design before the data exist, when it is cheapest to act on.

## Notation

| Symbol                                                         | Meaning                                                                                |
|:---------------------------------------------------------------|:---------------------------------------------------------------------------------------|
| $s_t, o_t, a_t$                                                | State, observation and action at step $t$                                              |
| $\boldsymbol{u}_t$                                             | Wheel angular-velocity set-points (4-vector)                                           |
| $c = (g_r, g_s, \boldsymbol\theta_\text{soil}, \boldsymbol h)$ | Hidden condition: rover-side gravity, soil-side gravity, soil parameters, wheel health |
| $g_\oplus, g_\star$                                            | Earth gravity; target body gravity                                                     |
| $c^\text{off}, c^\text{flt}$                                   | Gravitational-offset condition; flight condition                                       |
| $\Delta J$                                                     | Earth-test bias of performance measure $J$                                             |
| $\lambda$                                                      | Per-step probability of a condition switch (simulation)                                |
| $\mathcal{T}, \mathcal{I}_c, \mathcal{S}_\text{safe}$          | Trap set; irreversible set under $c$; safe set                                         |
| $\Pi_\text{esc}$, $H$                                          | Escape library; escape horizon                                                         |
| $m_c(s)$                                                       | Recoverability margin under known condition $c$                                        |
| $\bar m(s,b)$, $\underline m_\alpha(s,b)$                      | Belief-mean margin; lower $\alpha$-quantile margin                                     |
| $J_\text{1L}$, $\tau_\mathcal{I}$, $\delta$                    | One-life objective; first irreversible time; risk budget                               |
| $q_\psi(z \mid h_t)$                                           | Context encoder posterior over latent $z$                                              |
| $M_\xi(s,z)$                                                   | Margin network                                                                         |
| $\pi_\theta$                                                   | Driving policy                                                                         |
| $V_\text{probe}(k)$                                            | Value of probe macro-action $k$                                                        |
| $m_\text{low}, m_\text{high}$                                  | Gate thresholds                                                                        |
| $\boldsymbol\omega, \boldsymbol i$                             | Wheel angular velocities; motor currents                                               |
| $\sigma_j$                                                     | Longitudinal slip ratio of wheel $j$                                                   |
| $k_c, k_\phi, n, c_\text{coh}, \phi, K$                        | Bekker–Wong and Janosi–Hanamoto soil parameters                                        |
| $f_\chi$                                                       | Gravity-dependent soil-parameter surrogate fitted to Tier 2                            |

Notation used in this paper.



## Planned Hyperparameters

Table 12 lists starting values. They will be tuned on simulation validation conditions only, and the final values will be reported.

| Component   | Parameter                                     | Starting value                                                   |
|:------------|:----------------------------------------------|:-----------------------------------------------------------------|
| Control     | Policy rate / wheel loop rate                 | 20 Hz / 100 Hz                                                   |
| Encoder     | GRU width; latent dimension $d_z$             | 128; 8                                                           |
| Encoder     | Decoder horizon $K$; $\beta_\text{KL}$        | 10 steps; $10^{-2}$                                              |
| Encoder     | Privileged weight $\lambda_c$                 | 1.0, annealed to 0 over training                                 |
| Encoder     | Forgetting widening factor $\kappa$           | 4                                                                |
| Margin      | MLP size; samples $N_z$                       | $2 \times 128$; 32                                               |
| Margin      | Escape horizon $H$; repeats $n$               | 30 s simulated; 3                                                |
| Gate        | $(m_\text{low}, m_\text{high}, \alpha)$ grid  | $\{0.2,0.3,0.4\} \times \{0.6,0.7,0.8\} \times \{0.05,0.1,0.2\}$ |
| Gate        | Risk budget $\delta$                          | 0.05 per mission                                                 |
| Policy      | PPO clip; GAE $\lambda$; discount $\gamma$    | 0.2; 0.95; 0.995                                                 |
| Policy      | Rollout length; parallel environments         | 2048; 16                                                         |
| Probe       | Durations                                     | 0.5, 1.0, 1.5 s                                                  |
| Reward      | $\beta_E, \beta_s, \beta_p, \bar\sigma$       | set in Phase 2                                                   |
| Data policy | Aggressive-segment probability $p_\text{agg}$ | 0.2, annealed                                                    |

Planned starting hyperparameters (to be tuned on validation conditions).



## Escape Library Parameters

| Behavior                | Parameters                                              | Initial range                   |
|:------------------------|:--------------------------------------------------------|:--------------------------------|
| Reverse along own track | Speed; maximum distance                                 | 0.05–0.15 m/s; 0.3–1.0 m        |
| Rocking                 | Amplitude; period; number of cycles                     | 0.05–0.20 m; 1–4 s; 3–10        |
| Alternating pulsing     | Pairing (sides / diagonals); pulse duration; torque cap | discrete; 0.3–1.0 s; 40–100%    |
| Low-torque creep        | Torque cap; target speed                                | 20–60%; 0.02–0.08 m/s           |
| Pivot-out               | Yaw angle; then reverse distance                        | $5^\circ$–$20^\circ$; 0.2–0.6 m |

Escape behaviors and their parameters. Ranges are initial; they are bounded by motor limits measured in the hardware audit.



## Soil Bed Preparation and Characterization Procedure

The following procedure is executed before every hardware trial. Deviations are logged.

1.  **Loosen.** Rake the full bed to a depth of 0.20 m in two perpendicular passes.

2.  **Level.** Screed the surface with a straight edge to a flatness tolerance of $\pm 5$ mm.

3.  **Compacted beds only.** Apply $N_\text{roll}$ passes of a weighted roller, with $N_\text{roll}$ fixed during Phase 0 so that the target penetration resistance is reached.

4.  **Loose beds only.** Sieve or air-fluff the top 0.10 m, then screed lightly without applying load.

5.  **Slope sections.** Shape and verify the angle at three points with a digital inclinometer.

6.  **Measure bulk density.** Take a core of known volume at a fixed off-route location; weigh it; return the soil.

7.  **Measure penetration resistance.** Use a cone penetrometer at three fixed locations along the route, at 25, 50 and 100 mm depth.

8.  **Check tolerances.** If any measurement is outside its band, repeat from step 1.

9.  **Record.** Log the measurements, bed type, ambient temperature and humidity with the trial identifier.

## Data Logging and Release

All runs are recorded as ROS 2 bag files. Table 14 lists the planned topics. Each run’s bag is accompanied by a JSON metadata file with the trial identifier, cell, method, model checkpoint hash, bed measurements, battery voltage and motor temperatures at start, and the event log: trap declarations, escape attempts and their outcomes, injected faults, and soil-section crossings.

| Topic                         | Type                                          | Rate        |
|:------------------------------|:----------------------------------------------|:------------|
| `/joint_states`               | `sensor_msgs/JointState`                      | 100 Hz      |
| `/orion/motor_current`        | `std_msgs/Float32MultiArray` (4 currents)     | $\ge 50$ Hz |
| `/imu/data`                   | `sensor_msgs/Imu`                             | 100 Hz      |
| `/diff_drive_controller/odom` | `nav_msgs/Odometry`                           | 50 Hz       |
| `/orion/ground_truth_pose`    | `geometry_msgs/PoseStamped` (overhead camera) | 30 Hz       |
| `/orion/policy/latent`        | mean and variance of $q_\psi$                 | 20 Hz       |
| `/orion/policy/margin`        | $\bar m$, $\underline m_\alpha$               | 20 Hz       |
| `/orion/policy/mode`          | drive / cautious / probe / escape             | on change   |
| `/orion/wheel_cmd`            | wheel velocity set-points                     | 20 Hz       |
| `/scan`                       | `sensor_msgs/LaserScan` (logging only)        | 10 Hz       |

Planned logged topics. Topic names follow the existing ORION controller configuration where one exists.



**Release.** At publication we will release the simulation code, condition generator and split manifest with seeds, trained weights for all methods, the Tier 2 calibration data, all hardware bags and metadata, and the analysis notebooks that produce every table from raw data.

## Integration With the Existing ORION Software

The current ORION workspace contains three ROS 2 packages: a robot description in Xacro, a `ros2_control` configuration with a differential-drive controller, and bring-up launch files for Gazebo, SLAM, localization and navigation. A code review conducted while preparing this proposal found issues that must be fixed before Phase 4, because they affect the interfaces the learned controller relies on.

- The differential-drive controller in ROS 2 Jazzy accepts stamped twist commands on its own namespaced topic. Any node that publishes unstamped commands on `/cmd_vel` will not move the rover. The learned controller will command wheel velocities directly through a forward-command controller instead, bypassing the differential-drive kinematics. The differential-drive controller is retained only for the classical baseline.

- The description package’s build file installs directories that do not exist in the repository, which is likely to break a fresh build. This must be fixed before results can be reproduced from a clean clone.

- The skid-steer rover is modeled kinematically as a two-wheel differential drive with a fixed wheel separation. In turns, the wheels scrub sideways, so wheel-odometry yaw is expected to over-report rotation. This is one reason the learned controller does not rely on wheel odometry for velocity, and ground truth comes from the overhead camera.

- No motor current is currently published. A driver-level or sensor-level current interface must be added (Section 6).

## Planned Timeline

**Figure 3. Planned 20-week schedule (one primary researcher).**

| Weeks | Phase |
|---|---|
| 0–3 | Phase 0: de-risking → **go / pivot gate at week 3** |
| 3–6 | Phase 1: simulation |
| 6–9 | Phase 2: baselines → **hypothesis thresholds frozen at week 9** |
| 9–13 | Phase 3: proposed method |
| 13–18 | Phase 4: hardware trials |
| 18–20 | Phase 5: analysis, release |

The Phase 0 gate decides between the Earth-test-bias framing and the self-rescue fallback.

## Acknowledgment

\[Acknowledge supervisors, laboratory and funding sources.\]

## References

1.  W. Hu, P. Li, A. Rogg, A. Schepelmann, S. Chandler, K. Kamrin, and D. Negrut, “A study demonstrating that using gravitational offset to prepare extraterrestrial mobility missions is misleading,” *J. Field Robot.*, vol. 42, no. 7, pp. 3772–3794, 2025, doi: 10.1002/rob.22597.

2.  “NASA abandons escape attempts for stuck Mars rover,” *Space.com*, Jan. 2010. \[Online\]. Available: <https://www.space.com/7823-nasa-abandons-escape-attempts-stuck-mars-rover.html>

3.  “Spirit stuck in ‘insidious invisible rover trap’ on Mars,” *Space.com*, May 2009. \[Online\]. Available: <https://www.space.com/6740-spirit-stuck-insidious-invisible-rover-trap-mars.html>

4.  M. G. Bekker, *Introduction to Terrain-Vehicle Systems*. Ann Arbor, MI, USA: Univ. of Michigan Press, 1969.

5.  J. Y. Wong, *Theory of Ground Vehicles*, 4th ed. Hoboken, NJ, USA: Wiley, 2008.

6.  K. Iagnemma, S. Kang, H. Shibly, and S. Dubowsky, “Online terrain parameter estimation for wheeled mobile robots with application to planetary rovers,” *IEEE Trans. Robot.*, vol. 20, no. 5, pp. 921–927, Oct. 2004.

7.  K. Iagnemma and S. Dubowsky, *Mobile Robots in Rough Terrain*, ser. Springer Tracts in Advanced Robotics, vol. 12. Berlin, Germany: Springer, 2004.

8.  S. Banerjee, J. Harrison, P. M. Furlong, and M. Pavone, “Adaptive meta-learning for identification of rover-terrain dynamics,” 2020, *arXiv:2009.10191*.

9.  D. Qiu, “Naive Bayes entrapment detection for planetary rovers,” 2018, *arXiv:1801.10571*.

10. S. Omi, H.-S. Shin, N. Cho, A. Tsourdos, and M. A. Olivares-Mendez, “Robust recurrent reinforcement learning under evolving hidden disturbances with application to rover wheel slip,” 2023, *arXiv:2307.15931*.

11. E. S. Lupu, F. Xie, J. A. Preiss, J. Alindogan, M. Anderson, and S.-J. Chung, “MAGIC-VFM: Meta-learning adaptation for ground interaction control with visual foundation models,” 2024, *arXiv:2407.12304*.

12. A. Bouton, T. D. Hasseler, M. Paton, T. Brown, J. Levy, W. Reid, J. Martin, and H. Nayar, “Learning all-terrain locomotion for a planetary rover with actively articulated suspension,” 2026, *arXiv:2606.06790*.

13. P. Arm, O. Fischer, J. Church, A. Fuhrer, H. Kolvenbach, and M. Hutter, “Efficient learning-based control of a legged robot in lunar gravity,” 2025, *arXiv:2509.10128*.

14. F. Stark, R. Kumar, S. Vyas, H. Isermann, J. Haack, M. Popescu, J. Middelberg, D. Mronga, and F. Kirchner, “An adaptive hierarchical control framework for quadrupedal robots in planetary exploration,” 2025, *arXiv:2510.17249*.

15. C. Finn, P. Abbeel, and S. Levine, “Model-agnostic meta-learning for fast adaptation of deep networks,” in *Proc. 34th Int. Conf. Mach. Learn. (ICML)*, 2017, pp. 1126–1135.

16. Y. Duan, J. Schulman, X. Chen, P. L. Bartlett, I. Sutskever, and P. Abbeel, “RL$^2$: Fast reinforcement learning via slow reinforcement learning,” 2016, *arXiv:1611.02779*.

17. K. Rakelly, A. Zhou, D. Quillen, C. Finn, and S. Levine, “Efficient off-policy meta-reinforcement learning via probabilistic context variables,” in *Proc. 36th Int. Conf. Mach. Learn. (ICML)*, 2019, pp. 5331–5340.

18. L. Zintgraf, K. Shiarlis, M. Igl, S. Schulze, Y. Gal, K. Hofmann, and S. Whiteson, “VariBAD: A very good method for Bayes-adaptive deep RL via meta-learning,” in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2020.

19. A. Kumar, Z. Fu, D. Pathak, and J. Malik, “RMA: Rapid motor adaptation for legged robots,” in *Proc. Robot.: Sci. Syst. (RSS)*, 2021.

20. A. Nagabandi, I. Clavera, S. Liu, R. S. Fearing, P. Abbeel, S. Levine, and C. Finn, “Learning to adapt in dynamic, real-world environments through meta-reinforcement learning,” in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2019.

21. M. Al-Shedivat, T. Bansal, Y. Burda, I. Sutskever, I. Mordatch, and P. Abbeel, “Continuous adaptation via meta-learning in nonstationary and competitive environments,” in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2018.

22. A. S. Chen, A. Sharma, S. Levine, and C. Finn, “You only live once: Single-life reinforcement learning,” in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2022.

23. B. Eysenbach, S. Gu, J. Ibarz, and S. Levine, “Leave no trace: Learning to reset for safe and autonomous reinforcement learning,” in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2018.

24. B. Plaut, J. Liévano-Karim, H. Zhu, and S. Russell, “Safe learning under irreversible dynamics via asking for help,” *J. Mach. Learn. Res.*, vol. 27, no. 188, pp. 1–43, 2026.

25. A. Sorstkins, O. Tariq, and M. Bilal, “Learning to undo: Rollback-augmented reinforcement learning with reversibility signals,” 2025, *arXiv:2510.14503*.

26. M. Cho and C. Sun, “Constrained meta-reinforcement learning for adaptable safety guarantee with differentiable convex programming,” 2023, *arXiv:2312.10230*.

27. E. Altman, *Constrained Markov Decision Processes*. Boca Raton, FL, USA: Chapman & Hall/CRC, 1999.

28. J. Achiam, D. Held, A. Tamar, and P. Abbeel, “Constrained policy optimization,” in *Proc. 34th Int. Conf. Mach. Learn. (ICML)*, 2017, pp. 22–31.

29. J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” 2017, *arXiv:1707.06347*.

30. E. Todorov, T. Erez, and Y. Tassa, “MuJoCo: A physics engine for model-based control,” in *Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS)*, 2012, pp. 5026–5033.

31. A. Tasora *et al.*, “Chrono: An open source multi-physics dynamics engine,” in *High Performance Computing in Science and Engineering (HPCSE 2015)*, Lecture Notes in Computer Science, vol. 9611. Cham, Switzerland: Springer, 2016, pp. 19–49.

32. S. Macenski, T. Foote, B. Gerkey, C. Lalancette, and W. Woodall, “Robot Operating System 2: Design, architecture, and uses in the wild,” *Sci. Robot.*, vol. 7, no. 66, 2022, Art. no. eabm6074.

33. S. Chitta *et al.*, “ros_control: A generic and simple control framework for ROS,” *J. Open Source Softw.*, vol. 2, no. 20, p. 456, 2017.

[^1]: S. G is with the Department of Computer Science and Engineering (AI & ML), Maharaja Institute of Technology Mysore, Karnataka, India (e-mail: \[add\]).

[^2]: **Status of this manuscript:** research proposal with a pre-registered experimental protocol. No experimental results are reported. All result tables are intentionally empty and will be filled only with measured data.


---

# Part 2. Developments after the paper (2026-11-09)

## A. Round 7 candidate: tipping-point sensing (not yet adopted)

**Status:** this is a candidate feature. It depends on a 2-day sled-tow kill test that needs no RL. Do not build it into the policy until the user approves it after that test. The software interfaces should still support it: encoder and IMU logging at ≥100 Hz, and a perturbation-injection hook in the motor command path.

**Idea.** On loose soil, drawbar pull rises with slip, peaks, then falls, because more slip digs the wheel in (slip-sinkage). ISTVS 2025 tests on a tracked unit (Azuma *et al.*) show slip "sharply increases" once the drawbar-pull coefficient exceeds about 0.45. Steady driving is an equilibrium where pull equals resistance. Past the peak, no equilibrium exists, and the rover runs away into sinkage. That is a **fold bifurcation**. Near any fold, the system recovers ever more slowly from small perturbations (**critical slowing down**; Scheffer *et al.*, Nature 2009). ORION therefore periodically applies a small wheel-speed pulse and measures the **recovery rate λ** of body velocity. λ → 0 as the rover approaches entrapment.

**Why it matters for this paper.** In a simple model, λ = g_r · (dDP/ds at the operating slip) / (r·ω). The normalized rate λ/λ₀ (λ₀ measured on flat ground at the start) **cancels rover-side gravity exactly**. Soil-side effects are measured where the rover is, not predicted from Earth-fitted parameters. So this signal cannot inherit the Earth-test bias. It also addresses the paper's admitted weakness: Proposition 1 assumes calibration that landing breaks.

**Meta-RL role.** Recovery-rate features feed the margin/risk network next to the learned belief. The probe module learns when and how hard to pulse, trading information against risk.

**Toy-model evidence (not hardware):**
- Weak soil, lunar gravity: λ fell from 20/s to 1/s (−95%) as the slope rose from 0° to 15.5°. The true fold was at 15.7°. An Earth-calibrated model predicted a limit of 27.5°.
- Earth soil, lunar gravity: λ at 15.5° was still 47% of its flat-ground value, which correctly reads as "far from the limit".
- A simple noisy pulse-response estimator did **not** give a clean warning. Estimation is the main engineering risk.

**Known weaknesses:**
1. Recovery on firm ground takes 5–50 ms, too fast for the 20 Hz policy. It needs ≥100–200 Hz velocity sensing.
2. Sudden traps, such as driving straight into a loose patch, give no slowing-down warning. The signal covers gradual approaches only.
3. The toy model ignores sinkage dynamics and the motor loop.
4. Pulses cost energy and add slip.

**Kill test (about 2 days, Phase 0 week 2).** On the sand bin, the rover tows a sled through a spring scale, with load added stepwise until it stalls, in cells C and D. At each step it gives 10 pulses, and λ is fitted. **Pass:** normalized λ falls monotonically and is below 30% at least one load step before the stall, in both cells. **Kill:** no monotone fall, or the signal is lost in noise at the logging rate the hardware can reach.

**Proposed H4 (for the paper, if adopted):** "Normalized recovery rate predicts gradual-onset traps, and its trap threshold is the same in cells C and D." The 2×2 design already tests this invariance.

**Odds (team estimate, not data):** about 30–40% that the sled test shows a clean signal on real sand.

## B. Differences from `docs/SPEC.md` (engineering spec, later council review)

| Topic | Paper (Part 1) | SPEC.md (follow this for engineering) |
|---|---|---|
| Margin | "Recoverability margin" m_c(s): probability that an escape behavior succeeds | **"Entrapment risk"**: probability the rover gets stuck in the next 10 s. Reason: nothing on the real rover can label whether an escape *would have* worked, so the quantity is learned in simulation only, and on hardware only its predictions are checked |
| Simulator errors | g_s-dependent soil surrogate fitted to Tier 2 | The simulator **randomizes its own model errors**, including how gravity affects soil strength, so training on an Earth-based soil model does not reproduce the bias |
| Kill test size | H1 pilot 20 runs/cell (Table IX); main H1 test 40 runs in each of cells C and D (power ≈ 0.85) | **Reconciled (D-008):** two-stage pre-registered design. Stage 1 = Phase 0 pilot, 4 setups × 20 runs; early GO only at α = 0.01. Stage 2 = extend light-packed (C) and light-loose (D) to 40 runs each; final H1 test at α = 0.04. NO-GO only after stage 2 |
| Ground truth | Overhead camera | Overhead camera + **AprilTag** (approved). Without it, speed accuracy is only about 2–5 cm/s |

**Reconciled 2026-11-09 (DECISIONS.md D-008):** the paper's power argument is right: 20 runs per cell gives about 49% power for 10% vs 40% trap rates, so a 20-run NO-GO would be weak evidence. SPEC.md now uses the two-stage design above. Paper and SPEC agree on 40 runs in cells C and D for the H1 test.

## C. Prior art found after the paper was written (add to Section 2)

| Work | What it does | Overlap with ORION | Difference |
|---|---|---|---|
| Promit *et al.*, arXiv 2610.24827 (Oct 2026), "Escape Without a Recipe" | MPM simulation; a recurrent learned policy escapes granular entrapment (62.5% vs 7/32 for the best scripted controller); a frozen entrapment detector feeds a belief | Learned escape, entrapment belief | From the abstract: no early warning, no probing pulses, no gravity or Earth-test bias, simulation only. **Must be cited**, as it overlaps ORION's escape component |
| van Beers, Scheffer *et al.*, arXiv 2512.20868 (PNAS 2026) | Model-free early-warning signals of loss of control, from critical slowing down, on drones | Same principle as Round 7 | Drones; no wheels, soil, active pulses or meta-RL |
| van Beers *et al.*, arXiv 2609.18264 (2026) | Resilience indicators for autonomous control; quadrotor with progressively damaged blades | Same principle | Same differences |
| Albinsson *et al.*, Veh. Syst. Dyn. 2017 | Deliberate torque excitation of car wheels to estimate tire–road friction | Active excitation to find the grip limit | Cars on roads; tire-model fit; no recovery rate |
| Liang *et al.*, arXiv 2603.09073 (2026) | Automated vehicles excite high slip to estimate peak tire–road friction | Same | Same |
| Yoshida *et al.*, FSR 2003 | Terramechanics-based "traction margin" and "slip margin" for a lunar rover | Margin concept | Model-based, which is exactly what inherits the Earth bias |
| Shrivastava *et al.*, Sci. Robot. 2020; Karsai *et al.*, Adv. Intell. Syst. 2022 | Rovers remodel granular terrain to move or escape | Escape by terrain interaction | Rules out "terrain reshaping" as a novel ORION feature |
| Zhang *et al.*, arXiv 2604.02563 (2026); Liu *et al.*, arXiv 2602.18688 (2026) | Proprioceptive terrain-strength sensing from leg impacts; scout-rover terrain-strength maps | Probing soil with the body | Legged, not wheeled; no recoverability or bias |

**Novelty verdict (searches as of 2026-11-09):** no paper was found that combines (1) active wheel pulses with recovery-rate measurement on deformable soil, (2) the gravity-invariance argument against the Earth-test bias, and (3) use as the safety margin inside one-life meta-RL. Every ingredient exists separately. Pitch it as "the first gravity-invariant, measured entrapment margin for rovers", not as a new principle. A systematic review is still required.

## D. Related project artifacts

- LaTeX source and PDF of the paper: project files `orion-paper/` (`main.tex`, `sections/`, `main.pdf`, 26 pages, IEEEtran one-column 12pt).
- Concept council doc, with Rounds 0–7 and the investor teardown: https://claude.ai/artifact/WJqkycSK97cenFpAhgz2CR
- Repository and harness review: https://claude.ai/artifact/8iVeRDjnsHnVigUdQ6BPJ3
- Software build plan: https://claude.ai/artifact/MprgPQtt19TxtfGRTq1acN
- Engineering spec and logs: `docs/SPEC.md`, `docs/engineering-log/` (JOURNAL, MISTAKES, DECISIONS), `LESSONS.md`.
