from miles.utils.wandb_utils import init_wandb_primary, init_wandb_secondary


def init_observability_primary(args):
    init_wandb_primary(args)


def init_observability_secondary(args):
    init_wandb_secondary(args)
