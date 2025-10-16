import torch
from miles.utils.types import Sample


def read_rollout(rollout_data_path: str):
    pack = torch.load(rollout_data_path)
    samples = [Sample.from_dict(s) for s in pack["samples"]]
    return dict(
        pack=pack,
        samples=samples,
    )
