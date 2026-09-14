import json

import numpy as np
import torch

from experiments.continuous_intermediate_pooling_audit_v1 import extract, run


def test_protocol_freezes_reference_only_intermediate_grid():
    protocol = json.loads(extract.PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert protocol["scope"]["overlapindex_evaluated"] is False
    assert protocol["scope"]["reserved_backbones_evaluated"] is False
    assert len(protocol["targets"]) * len(protocol["backbones"]) * len(protocol["views"]) == 42
    assert len(protocol["targets"]) * len(protocol["backbones"]) * 3 * len(protocol["pooling"]) == 36
    assert len(run._rbf_grid(protocol)) == 504


def test_token_and_spatial_pooling_preserve_mean_and_dispersion():
    tokens = torch.tensor([[[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]]])
    token = extract._pool_tokens(tokens, torch)
    np.testing.assert_allclose(token["mean"], [[3.0, 6.0]])
    np.testing.assert_allclose(token["mean_std"], [[3.0, 6.0, np.sqrt(8.0 / 3.0), np.sqrt(32.0 / 3.0)]])
    spatial = extract._pool_spatial(tokens.transpose(1, 2).reshape(1, 2, 1, 3), torch)
    np.testing.assert_allclose(spatial["mean_std"], token["mean_std"])
