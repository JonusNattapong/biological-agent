"""Connectome representation, neuropils, and sparse synaptic weight graph."""

from enum import Enum
from typing import Any, Dict, Iterable, List, Optional
import numpy as np
import torch


class Neuropil(str, Enum):
    """Major neuropils of the Drosophila brain."""
    OPTIC_LOBE_LEFT = "OPTIC_LOBE_LEFT"
    OPTIC_LOBE_RIGHT = "OPTIC_LOBE_RIGHT"
    ANTENNAL_LOBE_LEFT = "ANTENNAL_LOBE_LEFT"
    ANTENNAL_LOBE_RIGHT = "ANTENNAL_LOBE_RIGHT"
    CENTRAL_COMPLEX = "CENTRAL_COMPLEX"
    MUSHROOM_BODY = "MUSHROOM_BODY"
    DESCENDING_MOTOR = "DESCENDING_MOTOR"
    INTERNEURON = "INTERNEURON"


class Connectome:
    """Connectome graph holding neuron metadata and sparse synaptic weight matrix."""

    def __init__(
        self,
        num_neurons: int,
        synapse_indices: torch.Tensor,  # shape (2, num_synapses): [post, pre]
        synapse_weights: torch.Tensor,  # shape (num_synapses,)
        neuron_neuropils: List[Neuropil],
        coordinates: np.ndarray,  # shape (num_neurons, 3) in microns
        neuron_names: Optional[List[str]] = None,
        body_ids: Optional[Iterable[int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
    ):
        self.num_neurons = num_neurons
        self.neuron_neuropils = neuron_neuropils
        self.coordinates = coordinates
        self.neuron_names = neuron_names or [f"neuron_{i}" for i in range(num_neurons)]
        self.body_ids = np.asarray(
            list(body_ids) if body_ids is not None else np.arange(num_neurons),
            dtype=np.int64,
        )
        if self.body_ids.shape != (num_neurons,):
            raise ValueError("body_ids must contain exactly num_neurons entries")
        self.body_id_to_index = {int(body): i for i, body in enumerate(self.body_ids)}
        self.metadata: Dict[str, Any] = dict(metadata or {})
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Create PyTorch sparse COO tensor for synaptic weights
        synapse_indices = synapse_indices.to(self.device)
        synapse_weights = synapse_weights.to(torch.float32).to(self.device)
        self.weight_matrix = torch.sparse_coo_tensor(
            synapse_indices,
            synapse_weights,
            (num_neurons, num_neurons),
            device=self.device,
        ).coalesce()

        # Cache neuropil indices for fast slicing
        self.neuropil_map: Dict[Neuropil, torch.Tensor] = {}
        for np_type in Neuropil:
            indices = [i for i, np_val in enumerate(neuron_neuropils) if np_val == np_type]
            self.neuropil_map[np_type] = torch.tensor(indices, dtype=torch.long, device=self.device)

    def get_neuropil_indices(self, neuropil: Neuropil) -> torch.Tensor:
        """Get neuron indices belonging to a specific neuropil."""
        return self.neuropil_map.get(neuropil, torch.empty(0, dtype=torch.long, device=self.device))

    def get_body_indices(self, body_ids: Iterable[int]) -> torch.Tensor:
        """Map external connectome body IDs to local tensor indices."""
        indices = [
            self.body_id_to_index[int(body)]
            for body in body_ids
            if int(body) in self.body_id_to_index
        ]
        return torch.tensor(indices, dtype=torch.long, device=self.device)

    def propagate_spikes(self, spikes: torch.Tensor) -> torch.Tensor:
        """Propagate presynaptic spikes across the connectome graph.

        Calculates incoming synaptic current:
            I_syn = W @ spikes
        where W is the sparse (post, pre) matrix.

        Args:
            spikes: Tensor of shape (num_neurons,)

        Returns:
            synaptic_currents: Tensor of shape (num_neurons,)
        """
        # Spikes as column vector: (num_neurons, 1)
        spikes_col = spikes.unsqueeze(1)
        # Sparse matrix-vector product
        syn_curr = torch.sparse.mm(self.weight_matrix, spikes_col).squeeze(1)
        return syn_curr

    @property
    def num_synapses(self) -> int:
        """Total number of directed synaptic connections."""
        return self.weight_matrix._nnz()
