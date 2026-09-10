"""Base interface for all agent controllers."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseController(ABC):
    """Abstract interface for benchmark agents/controllers."""

    @abstractmethod
    def reset(self):
        """Reset internal controller state."""
        pass

    @abstractmethod
    def act(self, obs: Dict[str, Any]) -> Any:
        """Decide action given sensory observation bundle."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Controller identifier."""
        pass
