from dataclasses import dataclass
from typing import Any, Dict, Optional

from miles.utils.types import Sample


# TODO may make input dataclass too to allow extensibility
@dataclass
class RolloutFnCallOutput:
    samples: Optional[list[list[Sample]]] = None
    metrics: Optional[Dict[str, Any]] = None
