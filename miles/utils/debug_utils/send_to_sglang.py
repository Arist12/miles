from typing import Annotated

import typer

from miles.utils.data import read_file


def main(
    prompt_data: Annotated[str, typer.Option()],
    url: Annotated[str, typer.Option()] = "http://localhost:30000",
    input_key: Annotated[str, typer.Option()] = "input",
    n_samples_per_prompt: Annotated[int, typer.Option()] = 1,
    rollout_max_response_len: Annotated[int, typer.Option()] = 1024,
    rollout_temperature: Annotated[float, typer.Option()] = 1.0,
    rollout_top_p: Annotated[float, typer.Option()] = 1.0,
):
    """
    Send prompts to SGLang with arguments in the same format as main Slime.

    Example usage:
    python -m miles.utils.debug_utils.send_to_sglang --prompt-data /root/datasets/aime-2024/aime-2024.jsonl --input-key prompt --n-samples-per-prompt 16 --rollout-max-response-len 32768 --rollout-temperature 0.8 --rollout-top-p 0.7
    """

    for row in read_file(prompt_data):
        input_messages = row[input_key]
        TODO


if __name__ == '__main__':
    typer.run(main)
