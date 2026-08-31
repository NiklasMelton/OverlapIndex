# Continuous OI oracle bottleneck diagnosis

This note records the completed frozen mechanism diagnostic. Its oracle views
use privileged generator state and are not deployable ranking methods.

## Identity and validation

- Implementation commit recorded by the runner: `76e9965`
- Protocol SHA-256:
  `a735bed8aff14d02583bc8a01dd518c8c9a995a1d69cf63308bab85545374904`
- Runner SHA-256:
  `4458c2095457ac25ec2728920773860f28c4000ec86d001f19a5ce88cdd4b20b`
- Raw-results SHA-256:
  `01f7627846d5ff00702fde0f68a1fa1f20b63b61f6dfc82f5668692290a740bb`
- Summary SHA-256:
  `a63d3746400a1b2f45363426c16390aa47fe78d063edf012cb34537cfd7f6726`
- Manifest SHA-256:
  `e6ed1d5a796e0a98897ab86d475fdd49384b1dd08dd75a70fb0e7452bfcf4eef`
- Complete grid: 15 panels, 240 candidate rows, and 15 matching atomic
  checkpoints.

All source hashes, terminal file hashes, checkpoint contents, and identities
were independently verified. Rebuilding the full summary from the checkpoint
panels matched exactly. The repository suite passes with 310 tests passed and
14 skipped; existing under-prototyped/multilabel runtime warnings are unchanged.

## Overall descriptive results

| View | Mean Spearman | Mean regret R2 | Exact best | Within 0.01 |
| --- | ---: | ---: | ---: | ---: |
| Linear probe | 0.508 | 0.146 | 0.333 | 0.600 |
| Observed, k=4 | 0.143 | **0.031** | 0.533 | 0.600 |
| Observed, k=8 | 0.238 | 0.048 | 0.467 | 0.733 |
| Oracle signal, k=4 | 0.420 | 0.077 | 0.333 | 0.533 |
| Oracle sufficient, k=4 | 0.410 | 0.076 | 0.400 | 0.533 |
| Oracle sufficient, k=8 | 0.386 | 0.071 | 0.200 | 0.533 |
| Oracle quadratic lift, k=4 | 0.535 | 0.047 | 0.200 | 0.733 |
| Oracle quadratic lift, k=8 | **0.578** | 0.060 | 0.400 | 0.733 |

The low observed-k4 regret is not evidence of a faithful ranking: its Spearman
was only 0.143. It repeatedly selected a favorable top candidate while poorly
ordering the rest of the panel.

## Bottleneck 1: nuisance geometry

The primary oracle estimand ranks signal quality within each fixed nuisance
level. Removing nuisance produced large gains:

| Family | Observed k4 quality rho | Oracle signal k4 quality rho |
| --- | ---: | ---: |
| Linear | 0.650 | 0.950 |
| Quadratic product | 0.450 | 0.983 |
| Radial quadratic | 0.100 | 0.483 |

For the product interaction, the minimal noisy `(u, v)` coordinates already
gave near-perfect quality ordering. A quadratic feature lift was not necessary
there. The earlier product failure was therefore primarily nuisance geometry,
not an intrinsic inability of prototypes to represent `u*v`.

## Bottleneck 2: second-order representation

Radial quadratic separation behaved differently. Removing nuisance raised
quality rho only to 0.417 in the minimal sufficient coordinates, while the
fixed quadratic lift raised it to 0.783 (0.800 with k=8). Its fixed-nuisance
selection regret fell from 0.235 to 0.074 at k=4.

This is evidence that a single global linear metric is insufficient for some
second-order relationships. A covariance-aware transform or scalable
second-order sketch is more promising than additional mean-based Fisher
projection.

## Bottleneck 3: prototype topology is conditional, not universal

Increasing prototypes from four to eight did not yield a uniform improvement:

- observed full-panel rho improved from 0.143 to 0.238, but regret worsened;
- lifted full-panel rho improved from 0.535 to 0.578, but regret worsened;
- quadratic-product quality ordering was already 0.983 and did not improve;
- radial lifted ordering improved slightly, while its selection regret worsened;
- sawtooth ordering improved modestly but remained weak.

Under-prototyping is therefore family-dependent. A globally larger `k` is not
the universal fix; an occupancy/topology-driven allocation would be the more
plausible future mechanism.

## Bottleneck 4: normalization is secondary

Mean Spearman based on final OI, negative raw actual loss, and negative loss
ratio was:

| View | Final OI | Negative actual loss | Negative loss ratio |
| --- | ---: | ---: | ---: |
| Observed k4 | 0.143 | 0.138 | 0.151 |
| Oracle signal k4 | 0.420 | 0.500 | 0.412 |
| Oracle sufficient k4 | 0.410 | 0.497 | 0.417 |
| Oracle quadratic lift k4 | 0.535 | 0.574 | 0.542 |
| Oracle quadratic lift k8 | 0.578 | 0.607 | 0.554 |

Null normalization discards some ordering once nuisance is removed, especially
for the oracle signal views, but it does not explain the observed-space
failure. It is a secondary calibration target after metric geometry.

## Periodic-family limitation

The sinusoid and sawtooth panels did not provide a valid strong-head stress
test. Mean best-envelope R2 was only about 0.20 for sinusoid and 0.003 for
sawtooth; linear Ridge remained the row winner in most cases. Their low OI
correlations cannot distinguish topology failure from an effectively flat,
weak reference envelope. These families must be redesigned before drawing a
periodic-geometry conclusion.

## Diagnosis and recommendation

There is no single bottleneck:

1. correlated nuisance obscures otherwise usable product and linear geometry;
2. radial relationships require second-order information absent from a global
   linear metric;
3. prototype capacity matters selectively rather than monotonically;
4. null normalization mildly reduces useful ordering in oracle spaces.

The strongest candidate for a more universal single transform is now a
regularized mean-plus-covariance sufficient-dimension reduction:

- retain residual scaling/deflation;
- use Fisher/SIR mean contrasts for linear directions;
- add SAVE-style conditional-covariance contrasts for symmetric and radial
  directions;
- keep the selected subspace available to FusedKMeans rather than replacing it
  with mean contrasts alone;
- compare fixed k=4 with a diagnostic adaptive-capacity policy, not blanket
  k=8.

The next frozen experiment should compare residual-only, Fisher, SAVE, and
Fisher+SAVE on the independent signal-by-nuisance grid. Periodic panels should
first be replaced with constructions where kNN or RBF achieves a substantial,
predeclared performance floor. Score-level hybridization is not yet necessary:
the oracle evidence supports testing covariance-aware geometry first.
