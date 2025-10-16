# Usage

```shell
# install dependencies
apt update && apt install -y docker-cli
pip install kimina-client

# prepare data
python examples/formal_math/single_round/prepare_data.py

# run
MILES_DATASET_TRANSFORM_ID=... python examples/formal_math/single_round/run.py
```
