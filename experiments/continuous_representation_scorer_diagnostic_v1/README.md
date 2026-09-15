# Continuous representation × scorer diagnostic V1

This frozen synthetic diagnostic asks whether continuous OI needs better
nuisance removal, an explicit nonlinear feature representation, or a different
ranking objective.

Each panel ranks six representations of the same continuous target: one
candidate and one weak control from each of the linear, product-quadratic, and
radial-quadratic families. The winning family reverses across scenarios.
Axis-aligned and 45-degree rotated forms test coordinate robustness. The
observed view contains either 14 or 510 nuisance dimensions; privileged
two-dimensional signal and degree-two lifted views isolate the failure stage.

Privileged views are diagnostic only. The experiment cannot promote a public
method or tune one from its outcomes.

The reference capacity and 1,200-row budget were calibrated before freezing on
provisional seed 1901. The terminal outcome grid uses disjoint seeds 2801,
3109, and 3413.
