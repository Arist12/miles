from typing import Annotated

import typer


def main(
    prompt_data: Annotated[str, typer.Option()],
    n_samples_per_prompt: Annotated[int, typer.Option()],
    rollout_max_response_len: Annotated[int, typer.Option()],
    rollout_temperature: Annotated[float, typer.Option()],
    top_p: Annotated[float, typer.Option()],
):
    """
    Send prompts to SGLang with arguments in the same format as main Slime.

    Example usage:
    python -m miles.utils.debug_utils.send_to_sglang --prompt-data /root/datasets/aime-2024/aime-2024.jsonl --n-samples-per-prompt 16 --rollout-max-response-len 32768 --rollout-temperature 0.8 --top-p 0.7
    """
    TODO


if __name__ == '__main__':
    typer.run(main)
