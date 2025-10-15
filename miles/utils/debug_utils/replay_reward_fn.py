from typing import Annotated

import torch
import typer


def main(
    rollout_data_path: Annotated[str, typer.Option()],
):
    data = torch.load(rollout_data_path)
    TODO


if __name__ == '__main__':
    typer.run(main)
