"""Sensory modules and encoders for Drosophila perception."""

from neurofly.sensory.vision import CompoundEyeVision, VisualStimulus
from neurofly.sensory.olfaction import AntennaeOlfaction, OlfactoryStimulus
from neurofly.sensory.mechanosensory import Mechanosensation, TactileStimulus
from neurofly.sensory.encoder import SensoryEncoder

__all__ = [
    "CompoundEyeVision",
    "VisualStimulus",
    "AntennaeOlfaction",
    "OlfactoryStimulus",
    "Mechanosensation",
    "TactileStimulus",
    "SensoryEncoder",
]
