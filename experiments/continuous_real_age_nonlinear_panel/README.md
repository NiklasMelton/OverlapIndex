# Controlled nonlinear real-image panel

This developmental panel reuses the exact frozen UTKFace images and 16 embedding caches from `continuous_real_age_panel`, changing only the scalar target to

```text
abs(age - 44.5) / 44.5
```

The transform is symmetric around the midpoint of the frozen 0–89 age range and creates a controlled V-shaped relation over genuine image embeddings. The candidate roster, training/evaluation splits, PCA, head grids, continuous-OI scores, bounded shortlist rule, and seeds remain unchanged.

This is deliberately not described as untouched confirmation: its target was selected after learning that the original age panel was entirely linear-head dominated.

