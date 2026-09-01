# Continuous bounded shortlist experiment

This frozen new-seed experiment tests a retrieval-and-reranking formulation
for a three-backbone continuous shortlist. Broad RFF retrieves six candidates
and preserves its complete winner group; SAVE+RFF reranks only the rest of that
pool. Candidates outside the pool cannot displace a retrieved candidate.

The experiment is synthetic and exploratory. It tests whether bounded
reranking retains broad RFF's periodic candidates while recovering SAVE+RFF's
linear and radial ordering gains. A pass permits a shared-computation
implementation experiment and real continuous-target validation only.

Run only after committing the frozen implementation:

```bash
CONTINUOUS_OI_COMMIT=$(git rev-parse HEAD) PYTHONPATH=. python -m \
  experiments.continuous_bounded_shortlist.run \
  --output artifacts/continuous_bounded_shortlist
```
