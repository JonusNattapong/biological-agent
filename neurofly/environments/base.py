"""Base interface for all NeuroFly simulation environments."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import numpy as np


class BaseEnvironment(ABC):
    """Abstract Environment interface."""

    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """Reset environment to initial state and return initial observation/sensory bundle."""
        pass

    @abstractmethod
    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Advance simulation by one tick.

        Returns:
            (observation, reward, done, info)
        """
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Return serializable world state for logging and web visualization."""
        pass
