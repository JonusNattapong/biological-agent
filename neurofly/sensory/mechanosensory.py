"""Mechanosensory touch and tactile feedback model."""

from dataclasses import dataclass


@dataclass
class TactileStimulus:
    """Tactile sensor readings from head and body bristles."""
    head_collision: float  # [0.0, 1.0] intensity of front collision
    body_contact: float  # [0.0, 1.0] intensity of lateral obstacle contact
    predator_contact: float  # [0.0, 1.0] threat touch


class Mechanosensation:
    """Mechanosensory model simulating campaniform sensilla and tactile bristles."""

    def perceive(
        self,
        wall_distance: float,
        collision_detected: bool,
        predator_distance: float,
        danger_radius: float = 12.0,
    ) -> TactileStimulus:
        """Compute tactile deflection signals."""
        head_col = 1.0 if collision_detected else 0.0

        # Close proximity to wall gives subtle whisker bristle feedback
        body_contact = 0.0
        if wall_distance < 8.0:
            body_contact = float(1.0 - (wall_distance / 8.0))

        pred_contact = 0.0
        if predator_distance < danger_radius:
            pred_contact = float(1.0 - (predator_distance / danger_radius))

        return TactileStimulus(
            head_collision=head_col,
            body_contact=body_contact,
            predator_contact=pred_contact,
        )
