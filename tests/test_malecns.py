"""Tests for official MaleCNS v1.0 Feather ingestion."""

import pyarrow as pa
import pyarrow.feather as feather
import pytest

from neurofly.brain import MALECNS_DATASET_ID, Neuropil, load_malecns_v1_bulk
from neurofly.controllers import MaleCNSController


def _write_fixture_tables(tmp_path):
    weights_path = tmp_path / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
    annotations_path = tmp_path / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
    nt_path = tmp_path / "body-neurotransmitters-male-cns-v1.0.feather"

    feather.write_feather(
        pa.table(
            {
                "body_pre": pa.array([10001, 10002, 10003], type=pa.int64()),
                "body_post": pa.array([10002, 10003, 10001], type=pa.int64()),
                "weight": pa.array([10, 5, 2], type=pa.int64()),
            }
        ),
        weights_path,
    )

    feather.write_feather(
        pa.table(
            {
                "bodyId": pa.array([10001, 10002, 10003], type=pa.int64()),
                "instance": ["DNp01_R", "OCG01d_L", "CX01_R"],
                "type": ["DNp01", "OCG01d", "CX01"],
                "superclass": ["descending_neuron", "visual_projection", "central_complex"],
                "class": [None, None, None],
                "somaSide": ["R", "L", "R"],
                "somaLocation": pa.array(
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    type=pa.list_(pa.int64()),
                ),
                "status": ["Traced", "Traced", "Traced"],
            }
        ),
        annotations_path,
    )

    feather.write_feather(
        pa.table(
            {
                "body": pa.array([10001, 10002, 10003], type=pa.int64()),
                "consensus_nt": ["acetylcholine", "gaba", "dopamine"],
            }
        ),
        nt_path,
    )

    return weights_path, annotations_path, nt_path


def test_load_official_malecns_bulk_fixture(tmp_path):
    weights_path, annotations_path, nt_path = _write_fixture_tables(tmp_path)

    conn = load_malecns_v1_bulk(
        weights_path,
        annotations_path,
        nt_path,
        min_synapses=2,
        weight_transform="raw",
        device="cpu",
    )

    assert conn.metadata["dataset_id"] == MALECNS_DATASET_ID
    assert conn.num_neurons == 3
    assert conn.num_synapses == 3
    assert conn.neuron_names == ["DNp01_R", "OCG01d_L", "CX01_R"]

    assert conn.neuron_neuropils[conn.body_id_to_index[10001]] == Neuropil.DESCENDING_MOTOR
    assert conn.neuron_neuropils[conn.body_id_to_index[10002]] == Neuropil.OPTIC_LOBE_LEFT
    assert conn.neuron_neuropils[conn.body_id_to_index[10003]] == Neuropil.CENTRAL_COMPLEX

    dense = conn.weight_matrix.to_dense()
    pre = conn.body_id_to_index[10002]
    post = conn.body_id_to_index[10003]
    assert dense[post, pre].item() == pytest.approx(-5.0)
    assert conn.coordinates[pre].tolist() == [4.0, 5.0, 6.0]


def test_malecns_controller_reports_real_dataset(tmp_path):
    weights_path, annotations_path, nt_path = _write_fixture_tables(tmp_path)
    controller = MaleCNSController.from_bulk_files(
        weights_path,
        annotations_path,
        nt_path,
        weight_transform="log1p",
        device="cpu",
    )

    assert controller.is_official_malecns is True
    assert controller.name == "MaleCNS v1.0 Connectome"
    assert controller.get_brain_telemetry()["dataset_id"] == MALECNS_DATASET_ID
