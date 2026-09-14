import copy
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import pytest
from scipy.io import savemat

from experiments.continuous_real_source_qualification_v1 import extract, prepare, run


def test_protocol_is_frozen_reference_only_and_exactly_sized():
    protocol = run.load_protocol()
    assert protocol["scope"]["reference_heads_only"] is True
    assert protocol["scope"]["overlapindex_evaluated"] is False
    assert protocol["scope"]["reserved_backbones_evaluated"] is False
    assert protocol["reference_library"]["consume_exactly_without_tuning"] is True
    assert tuple(protocol["reference_library"]["families"]) == run.HEADS
    assert tuple(protocol["backbones"]) == run.BACKBONES
    assert len(protocol["targets"]) == 4
    assert sum(len(row["draw_seeds"]) for row in protocol["targets"]) == 40
    assert protocol["expected_grid"]["candidate_draw_count"] == 200
    assert protocol["expected_grid"]["selected_image_count"] == 12000


def _rows(count=100):
    return [{"sample_id": f"s{index:04d}", "target": float(index)} for index in range(count)]


def test_internal_partition_is_deterministic_balanced_and_disjoint():
    left = prepare._partition_internal(_rows(200), training_rows=100, evaluation_rows=50, seed=17)
    right = prepare._partition_internal(_rows(200), training_rows=100, evaluation_rows=50, seed=17)
    assert left == right
    training = [row for row in left if row["split"] == "training"]
    evaluation = [row for row in left if row["split"] == "evaluation"]
    assert len(training) == 100 and len(evaluation) == 50
    assert {row["sample_id"] for row in training}.isdisjoint(row["sample_id"] for row in evaluation)
    assert [sum(row["target_bin"] == value for row in training) for value in range(10)] == [10] * 10
    assert [sum(row["target_bin"] == value for row in evaluation) for value in range(10)] == [5] * 10
    assert sorted(row["budget_rank"] for row in training) == list(range(100))


def _png16(values):
    handle = io.BytesIO()
    Image.fromarray(np.asarray(values, dtype=np.uint16)).save(handle, format="PNG")
    return handle.getvalue()


def _encode_sun_depth(millimetres):
    values = np.asarray(millimetres, dtype=np.uint16)
    return np.bitwise_or(np.left_shift(values, 3), np.right_shift(values, 13)).astype(np.uint16)


def test_sun_depth_decoder_recovers_log_median_and_coverage():
    encoded = _encode_sun_depth([[1000, 2000], [3000, 0]])
    target, coverage = prepare._decode_sun_depth(_png16(encoded))
    assert target == pytest.approx(np.log(2000 / 1000.0))
    assert coverage == pytest.approx(0.75)


def test_sun_official_split_reader_rejects_overlap(tmp_path):
    root = tmp_path / "sunrgbd"
    root.mkdir()
    buffer = io.BytesIO()
    savemat(buffer, {"alltrain": np.asarray(["/x/SUNRGBD/a"]), "alltest": np.asarray(["/x/SUNRGBD/a"])})
    with zipfile.ZipFile(root / "SUNRGBDtoolbox.zip", "w") as archive:
        archive.writestr("SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat", buffer.getvalue())
        archive.writestr("__MACOSX/SUNRGBDtoolbox/traintestSUNRGBD/._allsplit.mat", b"metadata")
    with pytest.raises(ValueError, match="empty or overlapping"):
        prepare._sun_split_directories(tmp_path)


def test_jhu_reader_uses_publisher_counts_and_native_splits(tmp_path):
    root = tmp_path / "jhu_crowd"
    root.mkdir()
    with zipfile.ZipFile(root / "jhu_crowd_v2.0.zip", "w") as archive:
        for split, count in (("train", 9), ("test", 99)):
            archive.writestr(f"jhu_crowd_v2.0/{split}/image_labels.txt", f"0001,{count},scene,0,0\n")
            archive.writestr(f"jhu_crowd_v2.0/{split}/images/0001.jpg", b"not-decoded-here")
    training, evaluation = prepare._jhu_rows(tmp_path)
    assert training[0]["target"] == pytest.approx(np.log1p(9))
    assert evaluation[0]["target"] == pytest.approx(np.log1p(99))
    assert training[0]["sample_id"] != evaluation[0]["sample_id"]


def test_summary_requires_three_independent_nonlinear_sources():
    protocol = run.load_protocol()
    screen = {"classification": "nonlinear_challenge_lead"}
    panels = [
        {"target_key": f"{source}::target", "dataset_id": source, "draw_count": 10, "screen": copy.deepcopy(screen)}
        for source in ("a", "b")
    ]
    assert run.summarize(protocol, panels)["decision"]["sufficient_real_nonlinear_diversity"] is False
    panels.append({"target_key": "c::target", "dataset_id": "c", "draw_count": 10, "screen": copy.deepcopy(screen)})
    assert run.summarize(protocol, panels)["decision"]["sufficient_real_nonlinear_diversity"] is True


def test_validate_grid_fails_closed_on_partial_checkpoint():
    with pytest.raises(ValueError, match="executed grid changed"):
        run.validate_grid(run.load_protocol(), {"panels": []})


def test_backbone_specs_are_exactly_the_frozen_scout_roster():
    protocol = run.load_protocol()
    specs = extract._protocol_specs(protocol)
    assert tuple(spec["backbone_id"] for spec in specs) == tuple(protocol["backbones"])
    assert all(spec["pretrained"] is True for spec in specs)


def test_archive_validation_fails_closed_on_hash_mismatch(tmp_path):
    protocol = copy.deepcopy(run.load_protocol())
    for name, relative in {
        "koniq_images": "koniq/koniq10k_512x384.zip",
        "koniq_scores": "koniq/koniq10k_scores_and_distributions.zip",
        "jhu": "jhu_crowd/jhu_crowd_v2.0.zip",
        "sunrgbd": "sunrgbd/SUNRGBD.zip",
        "sunrgbd_toolbox": "sunrgbd/SUNRGBDtoolbox.zip",
        "rsna_images": "rsna_bone_age/Bone_Age_Training_Set.zip",
        "rsna_annotations": "rsna_bone_age/Bone_Age_Training_Set_Annotations.zip",
    }.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        protocol["sources"]["archives"][name] = {"size_bytes": path.stat().st_size, "sha256": prepare.sha256_path(path)}
    protocol["sources"]["archives"]["jhu"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source archive identity changed: jhu"):
        prepare.validate_archives(tmp_path, protocol)


def test_qualified_head_library_ancestry_is_exact_and_passing():
    protocol = run.load_protocol()
    assert run.base.sha256_path(run.q3.PROTOCOL_PATH) == protocol["ancestry"]["qualification_v3_protocol_sha256"]
    terminal = run.q3.validate_artifacts(run.Q3_ROOT)
    assert terminal["decision"]["qualification_pass"] is True
