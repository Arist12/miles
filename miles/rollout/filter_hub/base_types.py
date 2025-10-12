from dataclasses import dataclass
from typing import Any, Dict, Optional

from miles.utils.types import Sample


@dataclass
class DynamicFilterOutput:
    keep: bool
    reason: Optional[str] = None
