# How to setup runner

### Step 1: Env

Write `.env` mimicking `.env.example`.
The token can be found at https://github.com/radixark/miles/settings/actions/runners/new?arch=x64&os=linux.

### Step 2: Run

```shell
(cd /data/tom/primary_synced/miles/tests/ci/github_runner && docker compose up -d)
```
