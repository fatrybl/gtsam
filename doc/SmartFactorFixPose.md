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

### Re-triangulation and First-Estimate Jacobians (FEJ)

Because $\phi_{\text{fix}}$ stays nonlinear, on every relinearization the
landmark is re-triangulated from the anchored + live cameras and
$G_{LL}, g_L, f$ are recomputed at the current live estimate while the anchored
camera stays at $\hat x_1$. Optionally, to avoid the spurious information gain
caused by relinearizing about a moving estimate (an observability/consistency
issue in sliding-window estimators), the live Jacobians can also be evaluated at
their first estimates (FEJ). FEJ is orthogonal to the mechanism here and is left
as future work; see Chen et al., *"FEJ2"*, IROS 2023.

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

## 5. What this enables (next steps)

With `fixPose` available, a fixed-lag smoother can, instead of deleting every
smart factor that touches an out-of-window pose (which discards the landmark
information in the surviving views), call `newFactor = oldFactor.fixPose(x, x̂)`
for each aging pose and let the global marginalization remove the pose once.
That integration (the "PR 2" in the discussion) is deliberately *not* part of
this change.
