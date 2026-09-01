from __future__ import annotations

import pytest

from experiments.continuous_real_age_nonlinear_panel import run
from experiments.continuous_real_age_panel import prepare


def test_frozen_v_target_is_symmetric_and_bounded() -> None:
    assert run.transformed_target(0) == 1.0
    assert run.transformed_target(89) == 1.0
    assert run.transformed_target(44) == run.transformed_target(45)
    assert run.transformed_target(44) == pytest.approx(0.5 / 44.5)
    with pytest.raises(ValueError):
        run.transformed_target(90)
    with pytest.raises(ValueError):
        run.transformed_target(44.0)


def test_protocol_and_lineage_are_exact() -> None:
    protocol = run._load_protocol()
    run._validate_lineage(protocol)
    assert tuple(protocol["replicate_seeds"]) == prepare.REPLICATE_SEEDS
    assert protocol["target"]["formula"] == "abs(age - 44.5) / 44.5"

