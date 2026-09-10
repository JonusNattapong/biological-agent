"""SensoryEncoder: Translates environmental stimuli into physiological currents for neuropils."""

from typing import Tuple
import torch

from neurofly.brain.connectome import Connectome, Neuropil
from neurofly.sensory.vision import VisualStimulus
from neurofly.sensory.olfaction import OlfactoryStimulus
from neurofly.sensory.mechanosensory import TactileStimulus


class SensoryEncoder:
    """Translates sensory inputs into physiological injected currents (I_inj) with spatial receptive fields."""

    def __init__(
        self,
        connectome: Connectome,
        visual_current_gain: float = 4.5,
        olfactory_current_gain: float = 5.0,
        tactile_current_gain: float = 8.0,
        looming_escape_gain: float = 10.0,
    ):
        self.connectome = connectome
        self.device = connectome.device

        self.v_gain = visual_current_gain
        self.o_gain = olfactory_current_gain
        self.t_gain = tactile_current_gain
        self.loom_gain = looming_escape_gain

        # Pre-fetch neuron indices
        self.left_optic_idx = connectome.get_neuropil_indices(Neuropil.OPTIC_LOBE_LEFT)
        self.right_optic_idx = connectome.get_neuropil_indices(Neuropil.OPTIC_LOBE_RIGHT)
        self.left_antennal_idx = connectome.get_neuropil_indices(Neuropil.ANTENNAL_LOBE_LEFT)
        self.right_antennal_idx = connectome.get_neuropil_indices(Neuropil.ANTENNAL_LOBE_RIGHT)
        self.mechanosensory_idx = connectome.get_neuropil_indices(Neuropil.INTERNEURON)

    def encode(
        self,
        visual: VisualStimulus,
        olfactory: OlfactoryStimulus,
        tactile: TactileStimulus,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute indices and injected current tensors with spatial receptive fields."""
        target_indices = []
        current_values = []

        # 1. Left Optic Lobe (Retinotopic & Looming stimulus)
        if self.left_optic_idx.numel() > 0:
            num_lo = self.left_optic_idx.numel()
            # Map ommatidia sectors across optic lobe neurons
            ommatidia = torch.tensor(visual.left_ommatidia, dtype=torch.float32, device=self.device)
            # Repeat to match neuron count
            reps = (num_lo // len(ommatidia)) + 1
            retinal_pattern = ommatidia.repeat(reps)[:num_lo]

            base_current = retinal_pattern * self.v_gain
            if visual.looming_threat_left > 0.05:
                base_current += visual.looming_threat_left * self.loom_gain

            active_mask = base_current > 0.3
            if active_mask.any():
                target_indices.append(self.left_optic_idx[active_mask])
                current_values.append(base_current[active_mask])

        # 2. Right Optic Lobe (Retinotopic & Looming stimulus)
        if self.right_optic_idx.numel() > 0:
            num_ro = self.right_optic_idx.numel()
            ommatidia = torch.tensor(visual.right_ommatidia, dtype=torch.float32, device=self.device)
            reps = (num_ro // len(ommatidia)) + 1
            retinal_pattern = ommatidia.repeat(reps)[:num_ro]

            base_current = retinal_pattern * self.v_gain
            if visual.looming_threat_right > 0.05:
                base_current += visual.looming_threat_right * self.loom_gain

            active_mask = base_current > 0.3
            if active_mask.any():
                target_indices.append(self.right_optic_idx[active_mask])
                current_values.append(base_current[active_mask])

        # 3. Left Antennal Lobe (Glomerular Odor Injection)
        if self.left_antennal_idx.numel() > 0 and olfactory.left_concentration > 0.02:
            num_la = self.left_antennal_idx.numel()
            # Glomerular diversity: subsets of PNs activate with varying sensitivity
            sensitivity = torch.linspace(0.4, 1.2, num_la, device=self.device)
            i_la = olfactory.left_concentration * self.o_gain * sensitivity
            target_indices.append(self.left_antennal_idx)
            current_values.append(i_la)

        # 4. Right Antennal Lobe (Glomerular Odor Injection)
        if self.right_antennal_idx.numel() > 0 and olfactory.right_concentration > 0.02:
            num_ra = self.right_antennal_idx.numel()
            sensitivity = torch.linspace(0.4, 1.2, num_ra, device=self.device)
            i_ra = olfactory.right_concentration * self.o_gain * sensitivity
            target_indices.append(self.right_antennal_idx)
            current_values.append(i_ra)

        # 5. Mechanosensation & Tactile Bristles
        if self.mechanosensory_idx.numel() > 0:
            i_touch = (
                tactile.head_collision * self.t_gain
                + tactile.body_contact * (self.t_gain * 0.4)
                + tactile.predator_contact * (self.t_gain * 1.2)
            )
            if i_touch > 0.2:
                target_indices.append(self.mechanosensory_idx)
                current_values.append(
                    torch.full_like(self.mechanosensory_idx, i_touch, dtype=torch.float32)
                )

        if target_indices:
            return torch.cat(target_indices), torch.cat(current_values)
        else:
            empty_idx = torch.empty(0, dtype=torch.long, device=self.device)
            empty_curr = torch.empty(0, dtype=torch.float32, device=self.device)
            return empty_idx, empty_curr
