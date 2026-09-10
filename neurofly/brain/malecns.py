"""MaleCNS v1.0 bulk-data ingestion.

This module loads the official HHMI Janelia MaleCNS flat-connectome Feather
exports into NeuroFly's sparse ``Connectome`` representation. The graph is
real MaleCNS connectivity; the neuron dynamics and sensory/motor transduction
remain NeuroFly simulation models and must not be confused with measured
membrane physiology.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from neurofly.brain.connectome import Connectome, Neuropil

MALECNS_DATASET_ID = "male-cns:v1.0"
MALECNS_RELEASE = "v1.0"
MALECNS_BULK_GS_BASE = "gs://flyem-male-cns/v1.0/connectome-data/flat-connectome"
MALECNS_BULK_HTTPS_BASE = (
    "https://storage.googleapis.com/flyem-male-cns/"
    "v1.0/connectome-data/flat-connectome"
)

WEIGHTS_FILENAME = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
ANNOTATIONS_FILENAME = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NEUROTRANSMITTERS_FILENAME = "body-neurotransmitters-male-cns-v1.0.feather"


def official_bulk_urls() -> Dict[str, str]:
    """Return canonical HTTPS download URLs for the core MaleCNS v1.0 tables."""
    return {
        "weights": f"{MALECNS_BULK_HTTPS_BASE}/{WEIGHTS_FILENAME}",
        "annotations": f"{MALECNS_BULK_HTTPS_BASE}/{ANNOTATIONS_FILENAME}",
        "neurotransmitters": f"{MALECNS_BULK_HTTPS_BASE}/{NEUROTRANSMITTERS_FILENAME}",
    }


def _require_pyarrow():
    try:
        import pyarrow.compute as pc
        import pyarrow.feather as feather
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "MaleCNS Feather loading requires pyarrow. "
            "Install with: pip install -e '.[malecns]'"
        ) from exc
    return feather, pc


def inspect_feather_schema(path: str | Path) -> List[str]:
    """Read only the Arrow schema names from a local Feather/IPC file."""
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "MaleCNS Feather inspection requires pyarrow. "
            "Install with: pip install -e '.[malecns]'"
        ) from exc

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"MaleCNS file not found: {path}")

    with pa.memory_map(str(path), "r") as source:
        reader = ipc.open_file(source)
        return list(reader.schema.names)


def _require_columns(columns: Sequence[str], required: Sequence[str], label: str) -> None:
    missing = [name for name in required if name not in columns]
    if missing:
        raise ValueError(
            f"Unexpected {label} schema; missing {missing}. "
            f"Available columns: {list(columns)}"
        )


def _annotation_neuropil(superclass: object, cell_class: object, neuron_type: object, side: object) -> Neuropil:
    """Map MaleCNS taxonomy to NeuroFly's coarse functional regions.

    This is intentionally a coarse adapter for simulation I/O. It is not an
    anatomical ROI assignment. Unknown cells remain ``INTERNEURON``.
    """
    text = " ".join(
        str(value).lower()
        for value in (superclass, cell_class, neuron_type)
        if value is not None and str(value).lower() not in {"nan", "none"}
    )
    side_text = str(side).upper() if side is not None else ""

    if "descending_neuron" in text or "descending neuron" in text or str(neuron_type).startswith("DN"):
        return Neuropil.DESCENDING_MOTOR

    if any(token in text for token in ("visual_projection", "optic", "lobula", "medulla", "lamina", "lptc")):
        return Neuropil.OPTIC_LOBE_LEFT if side_text == "L" else Neuropil.OPTIC_LOBE_RIGHT

    if any(token in text for token in ("olfactory", "antennal", "orn", "projection_neuron")):
        return Neuropil.ANTENNAL_LOBE_LEFT if side_text == "L" else Neuropil.ANTENNAL_LOBE_RIGHT

    if any(token in text for token in ("mushroom", "kenyon", "mushroom_body")):
        return Neuropil.MUSHROOM_BODY

    if any(token in text for token in ("central_complex", "central complex", "ellipsoid", "fan-shaped", "protocerebral bridge")):
        return Neuropil.CENTRAL_COMPLEX

    return Neuropil.INTERNEURON


def _parse_soma_location(value: object) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        arr = value
    elif isinstance(value, (list, tuple)):
        arr = np.asarray(value)
    else:
        return None
    if arr.size != 3:
        return None
    try:
        coords = arr.astype(np.float32, copy=False)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(coords)):
        return None
    return coords


def _load_annotations(
    path: Optional[str | Path],
    selected_body_ids: np.ndarray,
) -> Tuple[List[Neuropil], List[str], np.ndarray, Dict[str, object]]:
    n = int(selected_body_ids.size)
    neuropils = [Neuropil.INTERNEURON] * n
    names = [f"body_{int(body)}" for body in selected_body_ids]
    coordinates = np.zeros((n, 3), dtype=np.float32)
    meta: Dict[str, object] = {
        "annotation_mapping": "none",
        "coordinate_source": "unavailable",
    }

    if path is None:
        return neuropils, names, coordinates, meta

    feather, pc = _require_pyarrow()
    path = Path(path)
    columns = inspect_feather_schema(path)
    _require_columns(columns, ["bodyId"], "MaleCNS annotation")

    wanted = [
        col
        for col in (
            "bodyId",
            "instance",
            "type",
            "superclass",
            "class",
            "somaSide",
            "somaLocation",
            "status",
        )
        if col in columns
    ]
    table = feather.read_table(str(path), columns=wanted, memory_map=True)
    table = table.filter(pc.is_in(table["bodyId"], value_set=__import__("pyarrow").array(selected_body_ids)))
    frame = table.to_pandas()

    index_by_body = {int(body): i for i, body in enumerate(selected_body_ids)}
    soma_count = 0
    for row in frame.itertuples(index=False):
        body = int(getattr(row, "bodyId"))
        idx = index_by_body.get(body)
        if idx is None:
            continue

        instance = getattr(row, "instance", None)
        neuron_type = getattr(row, "type", None)
        if instance is not None and str(instance) not in {"", "nan", "None"}:
            names[idx] = str(instance)
        elif neuron_type is not None and str(neuron_type) not in {"", "nan", "None"}:
            names[idx] = str(neuron_type)

        neuropils[idx] = _annotation_neuropil(
            getattr(row, "superclass", None),
            getattr(row, "class", None),
            neuron_type,
            getattr(row, "somaSide", None),
        )

        soma = _parse_soma_location(getattr(row, "somaLocation", None))
        if soma is not None:
            coordinates[idx] = soma
            soma_count += 1

    meta.update(
        {
            "annotation_mapping": "coarse taxonomy adapter",
            "coordinate_source": "MaleCNS somaLocation when available; zeros otherwise",
            "annotated_neurons": int(len(frame)),
            "neurons_with_soma_coordinates": int(soma_count),
        }
    )
    return neuropils, names, coordinates, meta


def _load_presynaptic_polarity(
    path: Optional[str | Path],
    selected_body_ids: np.ndarray,
) -> Tuple[Dict[int, float], Dict[str, object]]:
    """Return a simple transmitter-derived presynaptic sign map.

    Only GABA is assigned an inhibitory sign. Other consensus transmitter
    labels remain positive because receptor-specific effects are not encoded
    by the aggregate body table.
    """
    if path is None:
        return {}, {"neurotransmitter_signing": "disabled"}

    feather, pc = _require_pyarrow()
    path = Path(path)
    columns = inspect_feather_schema(path)
    _require_columns(columns, ["body", "consensus_nt"], "MaleCNS neurotransmitter")

    table = feather.read_table(str(path), columns=["body", "consensus_nt"], memory_map=True)
    table = table.filter(pc.is_in(table["body"], value_set=__import__("pyarrow").array(selected_body_ids)))
    frame = table.to_pandas()

    signs: Dict[int, float] = {}
    gaba_count = 0
    for row in frame.itertuples(index=False):
        body = int(row.body)
        nt = str(row.consensus_nt).strip().lower()
        sign = -1.0 if nt == "gaba" else 1.0
        if sign < 0:
            gaba_count += 1
        signs[body] = sign

    return signs, {
        "neurotransmitter_signing": "consensus_nt; GABA inhibitory, all other labels positive",
        "neurotransmitter_rows": int(len(frame)),
        "gaba_neurons": int(gaba_count),
    }


def _transform_weights(raw: np.ndarray, transform: str, scale: float) -> np.ndarray:
    raw = raw.astype(np.float32, copy=False)
    if transform == "raw":
        out = raw
    elif transform == "sqrt":
        out = np.sqrt(raw)
    elif transform == "log1p":
        out = np.log1p(raw)
    elif transform == "binary":
        out = np.ones_like(raw)
    else:
        raise ValueError("weight_transform must be one of: raw, sqrt, log1p, binary")
    return out * np.float32(scale)


def load_malecns_v1_bulk(
    weights_path: str | Path,
    annotations_path: Optional[str | Path] = None,
    neurotransmitters_path: Optional[str | Path] = None,
    *,
    body_ids: Optional[Iterable[int]] = None,
    min_synapses: int = 1,
    weight_transform: str = "log1p",
    weight_scale: float = 1.0,
    device: Optional[str] = None,
) -> Connectome:
    """Load the official MaleCNS v1.0 segment-to-segment connection graph.

    Parameters
    ----------
    weights_path:
        Official ``connectome-weights-male-cns-v1.0-minconf-0.5.feather``.
    annotations_path:
        Optional official body-annotations table. Enables names, coarse
        functional-region mapping, and soma coordinates.
    neurotransmitters_path:
        Optional official aggregate transmitter table. Enables a conservative
        GABA-negative presynaptic sign model for simulation.
    body_ids:
        Optional subset. Both pre- and postsynaptic bodies must belong to this
        set. If omitted, the entire graph is loaded.
    min_synapses:
        Drop segment-to-segment edges with fewer than this many synapses.
    weight_transform:
        Convert synapse counts to simulation weights: raw/sqrt/log1p/binary.
        This transformation is a model choice; raw MaleCNS counts are retained
        in provenance metadata but are not membrane conductances.
    """
    if min_synapses < 1:
        raise ValueError("min_synapses must be >= 1")
    if weight_scale <= 0:
        raise ValueError("weight_scale must be > 0")

    feather, pc = _require_pyarrow()
    weights_path = Path(weights_path)
    columns = inspect_feather_schema(weights_path)
    _require_columns(columns, ["body_pre", "body_post", "weight"], "MaleCNS connectivity")

    table = feather.read_table(
        str(weights_path),
        columns=["body_pre", "body_post", "weight"],
        memory_map=True,
    )
    table = table.filter(pc.greater_equal(table["weight"], min_synapses))

    requested_ids: Optional[np.ndarray] = None
    if body_ids is not None:
        requested_ids = np.asarray(sorted({int(v) for v in body_ids}), dtype=np.int64)
        if requested_ids.size == 0:
            raise ValueError("body_ids cannot be empty")
        import pyarrow as pa

        values = pa.array(requested_ids)
        table = table.filter(
            pc.and_(
                pc.is_in(table["body_pre"], value_set=values),
                pc.is_in(table["body_post"], value_set=values),
            )
        )

    if table.num_rows == 0:
        raise ValueError("MaleCNS selection produced zero connectivity edges")

    body_pre = table["body_pre"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    body_post = table["body_post"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    raw_weight = table["weight"].to_numpy(zero_copy_only=False).astype(np.float32, copy=False)

    selected_body_ids = np.unique(np.concatenate((body_pre, body_post)))
    body_to_index = {int(body): idx for idx, body in enumerate(selected_body_ids)}

    pre_idx = np.fromiter(
        (body_to_index[int(body)] for body in body_pre),
        dtype=np.int64,
        count=len(body_pre),
    )
    post_idx = np.fromiter(
        (body_to_index[int(body)] for body in body_post),
        dtype=np.int64,
        count=len(body_post),
    )

    weights = _transform_weights(raw_weight, weight_transform, weight_scale)
    sign_map, nt_meta = _load_presynaptic_polarity(neurotransmitters_path, selected_body_ids)
    if sign_map:
        signs = np.fromiter(
            (sign_map.get(int(body), 1.0) for body in body_pre),
            dtype=np.float32,
            count=len(body_pre),
        )
        weights = weights * signs

    neuropils, names, coordinates, annotation_meta = _load_annotations(
        annotations_path,
        selected_body_ids,
    )

    synapse_indices = torch.from_numpy(np.vstack((post_idx, pre_idx))).long()
    synapse_weights = torch.from_numpy(weights.astype(np.float32, copy=False))

    metadata: Dict[str, object] = {
        "dataset_id": MALECNS_DATASET_ID,
        "release": MALECNS_RELEASE,
        "connectivity_source": str(weights_path),
        "annotations_source": str(annotations_path) if annotations_path is not None else None,
        "neurotransmitters_source": str(neurotransmitters_path) if neurotransmitters_path is not None else None,
        "edge_semantics": "official body_pre -> body_post synapse counts",
        "min_synapses": int(min_synapses),
        "weight_transform": weight_transform,
        "weight_scale": float(weight_scale),
        "requested_body_count": int(requested_ids.size) if requested_ids is not None else None,
        "loaded_body_count": int(selected_body_ids.size),
        "loaded_edge_count": int(len(weights)),
        **annotation_meta,
        **nt_meta,
    }

    return Connectome(
        num_neurons=int(selected_body_ids.size),
        synapse_indices=synapse_indices,
        synapse_weights=synapse_weights,
        neuron_neuropils=neuropils,
        coordinates=coordinates,
        neuron_names=names,
        body_ids=selected_body_ids,
        metadata=metadata,
        device=device,
    )
