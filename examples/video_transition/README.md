# Shared skill evolution for streaming video transitions

This integration runs frozen LongLive 1.0 on official instruction streams and evolves one shared
`skills/transition/SKILL.md`. A vision-language controller observes generated frames and proposes a short prompt
schedule. Hermit's GEPA Evolver updates the skill from scored trajectories and selects candidates on validation.
The generator weights, controller model, scheduler, and evaluator remain fixed during an experiment.

The [research plan](../../docs/developer/streaming-video-harness-evolution-plan.md) describes the intended experiments.
The initial four-case pilot checks the integration; it does not establish a generalization result.

## Model services and environments

Use Hermit's Python 3.11+ environment for preparation, the controller API client, evaluation, and GEPA.
LongLive runs in a separate CUDA environment with its upstream dependencies, including compatible Torch,
Flash Attention, Diffusers, PEFT, and PyAV. The worker never installs into that environment.

```bash
python -m pip install -e '.[test]' -r requirements-gepa.txt
python -m pip install -r examples/video_transition/requirements.txt
```

Each model role has its own settings:

| Role | Configuration | Requirement |
|---|---|---|
| GEPA learner | `learner` | Text chat completions; the project example uses Fireworks GLM-5.3 |
| Online controller | `controller` | Chat completions accepting ordered image inputs |
| Fixed judge | `evaluator` | Chat completions accepting up to eight images and returning strict JSON |

GLM-5.3 and GLM-5.3 Flash are distinct model IDs. Verify image support with the chosen endpoint. The current
pilot uses `accounts/fireworks/models/glm-5p3` for learning and `accounts/fireworks/models/glm-5p3-flash` for vision.
This judge differs from SEGUE's InternVL3.5-38B; report it as a separate evaluation version.
The example uses Fireworks' low reasoning setting and JSON output mode for vision calls so the token budget
includes a final structured answer. Adjust these provider-specific settings when changing services.

Copy `runtime.example.json` to `runtime.local.json` and configure endpoints, model IDs, generator paths,
the orchestration interpreter, and output directories. The local file is ignored by Git. Provider keys are read
from environment variables or a repository-external `secrets_file` with mode 600. That file contains literal
`KEY=value` assignments, such as `FIREWORKS_API_KEY=...`; no shell commands are executed.

The CUDA worker requires the official LongLive source, Wan2.1-T2V-1.3B assets, `longlive_base.pt`, and `lora.pt`.
If the source is a checkout of the current LongLive monorepo, point `generator.source` to `LongLive1.0/`.
Set any required CUDA library paths in `generator.environment`. All runtime paths are machine-specific.

## Prepare official data and evaluation rules

```bash
python examples/video_transition/utils/prepare.py
python examples/video_transition/utils/fetch_rubrics.py
```

Preparation downloads `LongLive1.0/example/interactive_example.jsonl` at the commit pinned in `prepare.py`.
It preserves the captions and records the source URL, commit, and checksum. The 12 official streams each contain
six prompts. A deterministic split keeps eight source streams for training, two for validation, and two for test.
Every stream's prefixes and seeds stay in its source group.

The four 20-second pilot tasks use only formal training source groups. Their own 2/1/1 integration split is
separate from the formal held-out streams. Generated tasks and source metadata live under
`.demo/video-transition/data/`, outside tracked source files. This small official example set is not the proposed
60-stream research dataset.

The rubric extractor reads SEGUE's public paper, validates ten criteria for each of the eight metrics, and records
both source and rubric checksums. The resulting local cache is not bundled into the repository. Set `rubrics_path`
to that cache. Sampling follows the paper's Table 12; image transport uses native-resolution JPEG at quality 95.
Evaluation with another model or image preprocessor must use a separate protocol label.

## Check the Hermit integration

```bash
python examples/video_transition/utils/with_runtime.py \
  --runtime examples/video_transition/runtime.local.json \
  check --config examples/video_transition/gepa/smoke.yaml --no-probe
```

The wrapper loads private credentials without placing their values in CLI arguments. It binds Hermit's learner
to the runtime's `learner` settings. A check without probing validates configuration but makes no video or model
calls. A full probe runs the actual acting pipeline and evaluator and therefore needs the runtime resources.

## Generate and evaluate the baseline

Submit from a host with Slurm, selecting the account and GPU resource type available on that cluster:

```bash
python examples/video_transition/utils/submit.py \
  --runtime examples/video_transition/runtime.local.json \
  --mode baseline --gres gpu:h100:1 --account YOUR_ACCOUNT
```

Submission freezes Python source, the task file, and runtime settings into a new run directory, and updates the
local `baseline_root`. Keys are loaded from the private file in the job, rather than embedded in its script.
Each case saves an MP4, latent tensor, original task, actual prompt-switch positions, timing, and model provenance.
Source revision recording works even if Git is absent from the CUDA job image.

LongLive generates three latent frames per block. The first decoded block has a different pixel length from later
blocks. The worker records requested times and actual condition-change times separately; a requested switch at
10 seconds is applied at latent boundary 42, corresponding to 10.3125 seconds for this decoder and frame rate.

After a case finishes, evaluate its actual output:

```bash
python examples/video_transition/utils/evaluate_video.py \
  --config examples/video_transition/runtime.local.json \
  --manifest /path/to/baseline-run/longlive-07-pilot/manifest.json
```

The judge sees the original user targets, not the optimized intermediate prompts. It saves sampled frame times,
binary decisions, failed criteria, model usage, and a summary. Cached judgments are tied to protocol and frame
checksums. Errors remain errors rather than becoming invented scores. High-frequency updates with overlapping
five-second target windows require a separate protocol and are rejected by this initial evaluator.

## Validate the controlled loop and evolve

Before attributing any improvement to prompt planning, run `--mode identity` on a selected pilot task and compare
its saved latents with `--mode baseline` at the same seed. Identity mode performs raw prompt switches through the
controlled loop, without model API calls. It checks that adapting the loop preserves the generator's behavior.

Run `--mode controller` to measure the initial shared skill. The scheduler discards superseded plan suffixes and
ignores replies to older instructions. Planning runs asynchronously while generation continues; late responses
are recorded and cannot rewrite finished video. Each controller request includes only already observed frames and
the arrived instructions.

Once generation, timing, and judgments have been inspected, submit the two-training-task GEPA pilot:

```bash
python examples/video_transition/utils/submit.py \
  --runtime examples/video_transition/runtime.local.json \
  --mode evolve --gres gpu:h100:1 --account YOUR_ACCOUNT --time 02:00:00
```

The job trains and evaluates root and final adapters. The evolving file is explicitly declared in the template
and GEPA config; the pipeline loads it for every candidate. `gepa/run.yaml` uses the full 8/2/2 source-stream
split and explicit test tasks for seeds 0, 1, and 2, and should follow a successful pilot with an explicit resource
budget. Its `eval.repeats` stays at one: repeating a deterministic cached task would not create independent seeds.

The generator is a separate process for each uncached Hermit task. This pilot favors a simple verifiable boundary;
a persistent generator service can reduce model reload overhead later. Vision-judge calls and GPU time are recorded
separately from Hermit's learner budget. Use measured costs to set a bounded experiment size.

## Verify implementation changes

```bash
python -m pytest examples/video_transition/utils/tests
```

Tests cover source-group split isolation, frame timing, strict evaluator output validation, information available
to the controller, plan budgets, and cancellation of stale plans. Tests use fake clients and make no paid calls.

## References

- [Official LongLive source and examples](https://github.com/NVlabs/LongLive/tree/main/LongLive1.0).
- [SEGUE and OpenTrans-360](https://arxiv.org/abs/2609.38691), especially Tables 12 and 19–26.
- [Hermit task interface](../../docs/framework/task.md) and [GEPA](../../docs/methods/gepa.md).
