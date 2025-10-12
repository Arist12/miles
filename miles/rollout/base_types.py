from dataclasses import dataclass
from typing import Optional, Any, Dict

from miles.utils.types import Sample


@dataclass
class RolloutFnCallOutput:
    samples: Optional[list[list[Sample]]] = None
    metrics: Optional[Dict[str, Any]] = None
