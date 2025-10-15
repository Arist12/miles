from typing import Optional


class RewardFn:
    def __init__(self):
        TODO

    def __call__(self, args, sample, **kwargs):
        TODO


_REWARD_FN: Optional[RewardFn] = None


async def reward_fn(*args, **kwargs):
    global _REWARD_FN
    if _REWARD_FN is None:
        _REWARD_FN = RewardFn()
    return _REWARD_FN(*args, **kwargs)
