from dataclasses import dataclass
from typing import Optional

from miles.utils.types import Sample


@dataclass
class RolloutFnCallOutput:
    samples: Optional[list[list[Sample]]] = None
