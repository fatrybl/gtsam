# Fixing poses in smart projection factors (`fixPose`)

This note describes the mathematics behind the `fixPose` capability added to
`SmartProjectionPoseFactor`. It corresponds to **PR "B"** in the design sketched
by F. Dellaert in the GTSAM discussion on using smart factors with fixed-lag
smoothers: *"making a '1 pose smaller' factor, which has a baked-in pose along
with its Jacobian."* It is intentionally self-contained and testable in
isolation; wiring it into the `IncrementalFixedLagSmoother`/`BatchFixedLagSmoother`
marginalization loop is a separate, later step.

## 1. Background: the smart projection factor

A single landmark $\ell \in \mathbb{R}^3$ is observed in $m$ views. View $i$ has
camera pose $x_i \in SE(3)$ (calibration is fixed and shared), and produces a
2D measurement $z_i \in \mathbb{R}^2$. With the projection function
$h(x_i, \ell)$ and isotropic measurement covariance $\Sigma = \sigma^2 I_2$, the
joint reprojection cost is

$$
E(x_{1:m}, \ell) \;=\; \tfrac12 \sum_{i=1}^{m} \big\lVert h(x_i, \ell) - z_i \big\rVert_{\Sigma}^2 .
$$

A *smart* factor does not keep $\ell$ as a variable. Instead it **profiles the
landmark out** by minimization (exact for the linear-Gaussian model used at each
linearization), producing a factor that depends only on the poses:

$$
\phi(x_{1:m}) \;=\; \min_{\ell}\; E(x_{1:m}, \ell).
$$

### Linearization and Schur complement

Linearizing around $(\bar x_{1:m}, \bar\ell)$, where $\bar\ell$ is obtained by
triangulation, with increments $\delta x_i$ and $\delta\ell$,

$$
h(x_i,\ell) - z_i \;\approx\; -b_i + F_i\,\delta x_i + E_i\,\delta\ell,
\qquad b_i = z_i - h(\bar x_i, \bar\ell),
$$

with $F_i = \partial h/\partial x_i \in \mathbb{R}^{2\times 6}$ and
$E_i = \partial h/\partial \ell \in \mathbb{R}^{2\times 3}$. Stacking and
whitening by $\Sigma^{-1/2}$, and writing

$$
F = \operatorname{blkdiag}(F_1,\dots,F_m)\in\mathbb{R}^{2m\times 6m},\quad
E = \begin{bmatrix}E_1\\\vdots\\E_m\end{bmatrix}\in\mathbb{R}^{2m\times 3},\quad
b = \begin{bmatrix}b_1\\\vdots\\b_m\end{bmatrix},
$$

the linearized cost is $\tfrac12\lVert F\,\delta x + E\,\delta\ell - b\rVert^2$.
Minimizing over the point increment $\delta\ell$ gives the **Schur complement of
the landmark**. With $P = (E^\top E)^{-1}$ and the rank-deficient projector
$Q = I - E P E^\top$,

$$
\phi(\delta x) \;=\; \tfrac12\,\delta x^\top \underbrace{F^\top Q F}_{G}\,\delta x
\;-\; \delta x^\top \underbrace{F^\top Q\, b}_{g}
\;+\; \tfrac12\,\underbrace{b^\top Q\, b}_{f}.
$$

This is exactly the augmented Hessian $\begin{bmatrix} G & g \\ g^\top & f\end{bmatrix}$
returned by `CameraSet::SchurComplement`. Block $i$ of $G$ (and entry $i$ of
$g$) corresponds to pose $x_i$.

## 2. Fixing a pose = conditioning at a value

Suppose pose $x_1$ must leave the active estimation window of a fixed-lag
smoother. We want to:

* **keep** the information that measurement $z_1$ contributes (it still
  constrains the landmark, and hence the remaining poses), but
* **stop** depending on $x_1$ as an optimization variable.

We therefore **fix** $x_1$ at its current estimate $\hat x_1$ (its
first-/best-estimate value), i.e. we set $\delta x_1 = 0$. Let
$L = \{2,\dots,m\}$ be the *live* (still-variable) poses and $A = \{1\}$ the
*anchored* (fixed) poses. The new factor is, again, a smart factor:

$$
\phi_{\text{fix}}(x_L) \;=\; \min_{\ell} \tfrac12\Big[
\big\lVert h(\hat x_1, \ell) - z_1 \big\rVert_\Sigma^2
+ \sum_{i\in L} \big\lVert h(x_i, \ell) - z_i \big\rVert_\Sigma^2 \Big].
$$

The landmark is still triangulated and profiled out, now using **all** cameras
(the anchored one held at $\hat x_1$ plus the live ones). Only the *poses* in
$A$ are frozen.

### The reduced augmented Hessian

Partition the full augmented Hessian of $\phi$ (over all $m$ poses) into live
($L$) and anchored ($A$) blocks:

$$
\begin{bmatrix}
G_{LL} & G_{LA} & g_L \\
G_{AL} & G_{AA} & g_A \\
g_L^\top & g_A^\top & f
\end{bmatrix},
\qquad
\phi(\delta x) = \tfrac12 \begin{bmatrix}\delta x_L\\\delta x_A\end{bmatrix}^\top
\begin{bmatrix}G_{LL}&G_{LA}\\G_{AL}&G_{AA}\end{bmatrix}
\begin{bmatrix}\delta x_L\\\delta x_A\end{bmatrix}
- \begin{bmatrix}g_L\\g_A\end{bmatrix}^\top\!\!\begin{bmatrix}\delta x_L\\\delta x_A\end{bmatrix} + \tfrac12 f.
$$

Fixing the anchored poses at their linearization value means substituting
$\delta x_A = 0$. The cost as a function of the live poses is then simply

$$
\phi_{\text{fix}}(\delta x_L)
= \tfrac12\,\delta x_L^\top G_{LL}\,\delta x_L - g_L^\top \delta x_L + \tfrac12 f,
\qquad\Longleftrightarrow\qquad
\begin{bmatrix} G_{LL} & g_L \\ g_L^\top & f \end{bmatrix}.
$$

In other words: **delete the rows and columns of the anchored blocks and keep
everything else, including the scalar $f$.** No extra approximation is
introduced beyond the linearization already inherent in the Schur complement.
The constant $f = b^\top Q b$ retains the residual energy of the anchored view,
so the factor's error is preserved.

This is the precise sense of Dellaert's *"calculate the Jacobian and redefine the
error function to add that additional cost"*: the anchored view's reprojection
residual and its coupling to the landmark are baked into $G_{LL}, g_L, f$.

### Conditioning, not marginalization

We **condition** $x_1$ (fix it at a value), we do not **marginalize** it
(integrate over its uncertainty) inside the factor. This is deliberate:

> *"A smart factor by itself cannot marginalize a pose, because that pose might
> be connected to many smart factors."* — F. Dellaert

The pose is typically shared by odometry, IMU, and other smart factors. A single
global marginalization of $x_1$ is performed once by the smoother (the later
PR). Each smart factor only needs to drop its dependence on $x_1$ while
preserving the contribution of $z_1$ — exactly the conditioning above. Doing a
local marginalization inside every factor would double-count $x_1$'s prior.

### Re-triangulation

Because $\phi_{\text{fix}}$ stays nonlinear, on every relinearization the
landmark is re-triangulated from the anchored + live cameras and
$G_{LL}, g_L, f$ are recomputed at the current live estimate while the anchored
camera stays at $\hat x_1$. This is the property that distinguishes `fixPose`
from freezing the factor into a linear marginal, and it is the dominant accuracy
benefit: the surviving views keep contributing a *live* constraint instead of a
stale linearization.

### First-Estimate Jacobians (FEJ) — not implemented

> **Status: documented for context only. FEJ is *not* implemented and the code
> always linearizes the live poses at their current estimate.** When/if added it
> would be a separate, opt-in, default-off option (see "Accuracy and stability"
> below).

**What it is.** In a sliding-window estimator a variable $x_i$ is re-linearized
repeatedly as its estimate $\bar x_i$ drifts, and marginalization then bakes
those linearizations into a permanent prior. Systems with **unobservable
directions** (monocular scale/gauge; VIO global position + yaw) require, along
such a direction $n$, that the stacked Jacobian satisfy $J(\bar x)\,n(\bar x)=0$.
The nullspace $n$ depends on the linearization point, so if different factors
linearize the *same* $x_i$ at *different* points $\bar x_i$, they share no common
nullspace and the information matrix $\Lambda=\sum_k J_k^\top\Sigma_k^{-1}J_k$
gains rank in directions that should be unobservable — *spurious information*,
i.e. over-confidence and bias.

FEJ fixes this by evaluating **every Jacobian** w.r.t. a variable at one fixed
point — the variable's **first** estimate $\bar x_i^{(0)}$ — for the variable's
whole life, while still evaluating the **residual** $b_k=-r_k(\bar x_{\text{cur}})$
at the current estimate:

$$
r_k(\bar x_{\text{cur}}+\delta)\;\approx\; r_k(\bar x_{\text{cur}})\;+\;J_k\big|_{\bar x^{(0)}}\,\delta .
$$

Applied to our smart factor: the pose Jacobians $F_i$ and the point Jacobian
$E_i$ (and the triangulated $\bar\ell$) would use the first estimates
$\bar x_i^{(0)}$, while $b_i = z_i - h(\bar x_i^{\text{cur}},\bar\ell^{\text{cur}})$
stays at the current estimate. Note the **anchored** pose is already a
first-estimate quantity — it is frozen at $\hat x_1$ by construction — so FEJ is
just the extension of the same idea to the still-live poses, which keeps the
linearization consistent across the marginalization seam.

**Benefits.**
- Preserves the unobservable subspace ⇒ no spurious information gain ⇒
  **consistent** (not over-confident) covariance.
- Over long, low-observability trajectories this also improves **accuracy**, by
  removing the dominant systematic error of sliding-window VIO.
- Makes the frozen anchored block and the moving live blocks share one
  nullspace, so the marginalization seam stays consistent.

**Drawbacks.**
- The Jacobian is **stale** (evaluated at $\bar x^{(0)}$, not $\bar x_{\text{cur}}$),
  a worse local approximation; in well-observed, well-initialized, low-drift
  problems this can *reduce* accuracy versus plain relinearization.
- It **fights iSAM2's design**, whose accuracy relies on selective
  relinearization (the wildfire threshold). Pinning Jacobians undercuts that.
- It is only **partially effective** if applied inside the smart factor alone:
  true consistency needs *every* factor sharing those poses (odometry, IMU,
  priors) to use FEJ as well — it is really a smoother-wide policy.
- Requires storing and managing per-variable first estimates.

**Accuracy and stability (summary).** The ordering of impact is
`fixPose` (keep the factor nonlinear / re-triangulate) $\gg$ anchoring the
retiring pose at exactly the smoother's marginalization linearization point
$\gg$ FEJ. The first two are what make the result accurate and stable in the
common case and are always on; FEJ is insurance against long-horizon
over-confidence in the presence of unobservable directions, worth having as a
default-off switch but not as the default. See Huang–Mourikis–Roumeliotis
(OC-EKF) and Chen et al., *"FEJ2"*, IROS 2023.

## 3. Algorithm

Given a factor over keys $\{x_1,\dots,x_m\}$ with measurements $\{z_1,\dots,z_m\}$
and a request to fix $x_a$ at value $\hat x_a$:

```
fixPose(x_a, x̂_a):
  move (x_a, z_a) from the live set into the anchored set:
      anchored ← anchored ∪ {(x̂_a, z_a)}
      keys_    ← keys_    \ {x_a}
      measured_← measured_\ {z_a}
  return the modified (still nonlinear) smart factor
```

Linearization of an anchored factor (live keys $L$, anchored cameras built at
the stored $\hat x$ values):

```
linearizeFixed(values):
  cameras  ← cameras(values) ⊕ anchoredCameras      # live first, then anchored
  measured ← measured_       ⊕ anchoredMeasured
  ℓ ← triangulate(cameras, measured)
  if degenerate: return zero Hessian over L
  (F, E, b) ← jacobians(cameras, measured, ℓ)        # whiten; apply body_P_sensor chain rule
  P ← (EᵀE)⁻¹
  H_full ← SchurComplement(F, E, P, b)               # augmented, all m poses + b
  H_red  ← selectLiveBlocks(H_full, |L|, m)           # drop anchored rows/cols, keep f
  return RegularHessianFactor(L, H_red)
```

Ordering the anchored cameras **after** the live ones makes `selectLiveBlocks`
a trivial sub-block extraction: keep blocks $0,\dots,|L|-1$ and the trailing
information block.

## 4. Implementation map

| Math object | Code |
|---|---|
| anchored value $\hat x_a$ (world←body) | `SmartProjectionPoseFactor::fixedPoses_` |
| anchored measurement $z_a$ | `SmartProjectionPoseFactor::fixedMeasured_` |
| `fixPose(x_a, x̂_a)` | `SmartProjectionPoseFactor::fixPose` |
| anchored cameras | `SmartProjectionPoseFactor::fixedCameras` |
| $H_{\text{full}}$ | `CameraSet::SchurComplement<3, 6>` |
| `selectLiveBlocks` ($H_{\text{full}}\!\to\!H_{\text{red}}$) | `SmartProjectionPoseFactor::SelectLiveBlocks` |
| $\phi_{\text{fix}}$ linear factor | `SmartProjectionPoseFactor::createHessianFactorFixed` |

## 5. Smoother integration

A fixed-lag smoother uses `fixPose` instead of deleting every smart factor that
touches an out-of-window pose (which discards the landmark information of the
surviving views) or freezing it into a linear marginal (which loses
re-triangulation). For each retiring pose `x` it calls
`newFactor = oldFactor.fixPose(x, x̂)` and lets the *global* marginalization
remove `x` once, via the rest of the graph.

To keep the smoother independent of any SLAM type, the capability is exposed
through a small interface, `gtsam::FixableFactor`, with a single method

```cpp
NonlinearFactor::shared_ptr fixKeys(const KeyVector& keysToFix,
                                    const Values& values) const;
```

`SmartProjectionPoseFactor` implements it by chaining `fixPose` over the keys it
shares with `keysToFix` (returning `nullptr` if no live key remains).

### `BatchFixedLagSmoother`

Before its standard marginalization, `BatchFixedLagSmoother::marginalize`
runs a pass (`fixMarginalizedSmartFactors`) that, for every `FixableFactor`
which touches both a marginalized key and a surviving key, replaces it in place
with `fixKeys(marginalizeKeys, theta_)`. The retiring poses are anchored at
their current estimate $\hat x$, the factor stops depending on them, and the
remaining (odometry/IMU/prior) factors marginalize the pose globally. The
behavior can be toggled with `setFixSmartFactorsOnMarginalize(bool)` (default
on; only affects `FixableFactor` instances).

This is exactly the workflow Dellaert sketched:
`newFactor <- oldFactor.fixPose(pose)` followed by a single global
marginalization. Because the pose is *conditioned* (not marginalized) inside the
factor, the same pose can be shared across many smart factors without
double-counting.

### `IncrementalFixedLagSmoother`

The iSAM2-based smoother performs the same `fixKeys` replacement, but as a
factor add/remove inside the existing `ISAM2::update` call that precedes
`marginalizeLeaves`. `prepareFixedSmartFactors` scans the factors already in
iSAM2, and for every `FixableFactor` touching both a marginalizable key and a
surviving key it builds `fixKeys(marginalizableKeys, isam_.getLinearizationPoint())`,
appends it to the factors being added and the old factor's index to the factors
being removed. Anchoring at `isam_.getLinearizationPoint()` is deliberate: it is
exactly the point iSAM2 marginalizes about, so the factor's frozen pose and the
global marginal agree (a consistency requirement — see "Accuracy and stability"
above). The same `setFixSmartFactorsOnMarginalize(bool)` toggle applies (default
on).

Because the replacement smart factor no longer depends on the marginalizable
keys, it is untouched by the subsequent `marginalizeLeaves` and survives as a
smaller, still-nonlinear factor.
