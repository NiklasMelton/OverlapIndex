# SmallNORB structured-pose discovery

This is the independent structured-dataset extension of the real annotation
screen. SmallNORB deliberately crosses object category, unseen object instance,
elevation, azimuth, and lighting. We use the official instance-disjoint split,
the left camera only, and three continuous pose targets.

The experiment uses five scout backbones. Five other backbones remain reserved.
It asks whether the pose target is recoverable, materially favors a nonlinear
head, has stable head-family selection, and produces enough backbone utility
spread to support a ranking experiment. It does not treat low accuracy as proof
of nonlinearity and it is not confirmation evidence.

Stages:

```bash
PYTHONPATH=. python -m experiments.continuous_smallnorb_head_discovery.prepare
PYTHONPATH=. python -m experiments.continuous_smallnorb_head_discovery.extract
PYTHONPATH=. python -m experiments.continuous_smallnorb_head_discovery.run
```

Every stage is non-overwriting and hash-binds the prior stage.

After discovery admitted only elevation, the separately frozen reserved-
backbone validation was run with the unchanged cohorts, heads, preprocessing,
and thresholds:

```bash
PYTHONPATH=. python -m experiments.continuous_smallnorb_head_discovery.extract_reserved
PYTHONPATH=. python -m experiments.continuous_smallnorb_head_discovery.validate_reserved
```

Its immutable design is `validation_protocol.json`; it binds the completed
discovery bytes and permits no azimuth or runner-up target.
