from dataclasses import dataclass
from typing import Any, Dict, Optional

from miles.utils.types import Sample


@dataclass
class RolloutFnCallOutput:
    samples: Optional[list[list[Sample]]] = None
    metrics: Optional[Dict[str, Any]] = None
