# Shared skill evolution for streaming video transitions

This integration runs frozen LongLive 1.0 on official instruction streams and evolves one shared
`skills/transition/SKILL.md`. A vision-language controller observes generated frames and proposes a short prompt
schedule. Hermit's GEPA Evolver updates the skill from scored trajectories and selects candidates on validation.
The generator weights, controller model, scheduler, and evaluator remain fixed during an experiment.

The [research plan](../../docs/developer/streaming-video-harness-evolution-plan.md) describes the intended experiments.
The initial four-case pilot checks the integration; it does not establish a generalization result.

## Current experiment: one prompt pair, offline

The first focused experiment uses the first two prompts from official source row 02: a poker player grips his
cards tensely, then flicks them onto the table and leans back in celebration. Each video is 20 seconds long and
the user instruction changes at 10 seconds. At that boundary, `planning_mode: offline` pauses generation until
the controller has observed preceding frames and returned its transition plan. API latency does not advance
video time. The generator then executes the plan and restores the original second user prompt.

```bash
python examples/video_transition/utils/prepare_two_prompt.py
python examples/video_transition/utils/with_runtime.py \
  --runtime examples/video_transition/runtime.two_prompt.local.json \
  check --config examples/video_transition/gepa/two_prompt.yaml --no-probe
python examples/video_transition/utils/submit.py \
  --runtime examples/video_transition/runtime.two_prompt.local.json \
  --mode controller \
  --tasks .demo/video-transition/two-prompt-offline/data/tasks.jsonl \
  --ids-file .demo/video-transition/two-prompt-offline/data/train.txt
```

Preparation reuses the verified seed-0 raw baseline from the earlier pilot and preserves its provenance. It
creates a separate experiment directory and private runtime settings. The evaluator uses Fireworks
[JSON Schema output constraints](https://docs.fireworks.ai/structured-responses/structured-response-formatting)
to require exactly ten binary `judgments` plus `sum`; client validation still checks their consistency. This
sampling configuration has a separate evaluation protocol label.
The judge's ten binary decisions define the score. A recovery layer computes their sum locally when the model's
redundant reported total is wrong. It verifies the protocol, rubric, and frame checksums, preserves the original
API response byte for byte, and marks the derived metric cache with the reported and computed totals. Completed
decisions are reused; the recovery makes no additional model calls for that metric. Invalid decision vectors and
other evaluation errors still fail explicitly.

`gepa/two_prompt.yaml` runs **four epochs / four training steps**. Training, validation, and final comparison use
seeds 0, 1, and 2 of the same pair. These are seed holdouts for debugging one pair; they do not test unseen-prompt
generalization. The original full-stream splits remain separate. After inspecting initial generation and scores:

```bash
python examples/video_transition/utils/submit.py \
  --runtime examples/video_transition/runtime.two_prompt.local.json \
  --mode evolve --gepa-config examples/video_transition/gepa/two_prompt.yaml \
  --time 24:00:00
```

The same step, video, judgment, and Slurm resume mechanisms below apply to the offline experiment.

To extend a finished four-step run to **40 total steps**, preserving its training state and candidate pool:

```bash
python examples/video_transition/utils/submit.py \
  --resume /absolute/path/to/evolve-run --steps 40 --always-propose
```

This archives the previous final report, adapter, final evaluation, and Git tag, then resumes at step 5 from the
last training checkpoint. Existing video and judgment caches remain available. Root evaluation is reusable;
final evaluation is run after the new target completes. `--always-propose` disables GEPA's perfect-parent skip
so every remaining step attempts a proposal; acceptance and validation selection use the existing rules.
For this one-training-task config, 40 steps correspond to 40 epochs. The allocation limit and saved
`--time-min` backfill policy continue to apply.

## Ten training transitions and four validation transitions

The pinned official interactive example contains twelve source scenes. This experiment selects ten distinct
adjacent prompt pairs from six scenes for training, four pairs from four other scenes for validation, and two
pairs from the remaining two scenes for testing. The four validation scenes are the boy on the lawn, the Rio
street celebration, the couple picking a flower, and the model in pink smoke. Every validation and test pair
uses seeds 0, 1, and 2, producing twelve validation tasks and six test tasks. No source scene crosses splits.

Each video still has two unchanged official prompts, a switch at ten seconds, and twenty seconds of video.
All candidates share one evolving skill. Four epochs over ten training tasks give forty steps, with a
reproducible shuffled order and proposals attempted even when a training parent scores perfectly.

```bash
python examples/video_transition/utils/prepare_ten_prompt.py
python examples/video_transition/utils/submit.py \
  --mode evolve --runtime examples/video_transition/runtime.ten_prompt.local.json \
  --gepa-config examples/video_transition/gepa/ten_prompt.yaml \
  --steps 40 --always-propose --time 24:00:00 --time-min 00:30:00
```

Preparation reuses the configured offline two-prompt runtime and pins the original prompt file checksum.
It preserves the previous one-pair experiment. Validation selects the final skill; the separate test scenes
are evaluated for the initial and selected final skills after training. Cached videos and API responses,
per-step checkpoints, and the 24-hour allocation resume mechanism apply to this run as well. The per-step
timeout is ninety minutes to accommodate twelve validation videos when a new candidate is accepted.

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
GEPA also needs working `git` and `tar` executables on the compute node. If the node image lacks Git, supply a
compatible binary and local-fetch helpers through `generator.environment.PATH` and `GIT_EXEC_PATH`.

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
Set `slurm.max_hours: 24` in runtime settings for this cluster. The wrapper reports that external allocation
limit alongside the four-epoch stopping condition; it does not invent a cumulative training-time or dollar cap.

## Generate and evaluate the baseline

Submit from a host with Slurm, selecting the account and GPU resource type available on that cluster:

```bash
python examples/video_transition/utils/submit.py \
  --runtime examples/video_transition/runtime.local.json \
  --mode baseline --gres gpu:h100:1 --account YOUR_ACCOUNT
```

Submission freezes the Hermit framework, task Python source, adapter, tasks, rubrics, and runtime settings into a new run directory, and updates the
local `baseline_root`. Keys are loaded from the private file in the job, rather than embedded in its script.
Each case saves an MP4, latent tensor, original task, actual prompt-switch positions, timing, and model provenance.
Source revision recording works even if Git is absent from the CUDA job image.

LongLive generates three latent frames per block. The first decoded block has a different pixel length from later
blocks. The worker records requested times and actual condition-change times separately; a requested switch at
10 seconds is applied at latent boundary 42, corresponding to 10.3125 seconds for this decoder and frame rate.

After a case finishes, evaluate its actual output:

```bash
python examples/video_transition/utils/repair_arithmetic.py \
  --runtime examples/video_transition/runtime.local.json \
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
  --mode evolve --gres gpu:h100:1 --account YOUR_ACCOUNT --time 24:00:00
```

Both GEPA configs request **four epochs**. The pilot has eight training steps (two tasks, one per step); the full
config has sixteen steps (eight tasks, two per step). Training ends after all four epochs. The former 40,000-token
and two-hour pilot caps have been removed; these configs have no cumulative money, token, or training-time cap.
Each Slurm allocation is at most 24 hours. The per-call timeout still detects a stuck Evolver call.

The job trains and evaluates root and final adapters. The evolving file is explicitly declared in the template
and GEPA config; the pipeline loads it for every candidate. `gepa/run.yaml` uses the full 8/2/2 source-stream
split and explicit test tasks for seeds 0, 1, and 2, and should follow a successful pilot with an explicit resource
budget. Its `eval.repeats` stays at one: repeating a deterministic cached task would not create independent seeds.

## Resume across Slurm allocations

Evolve submissions default to `24:00:00` and reject requests above the cluster's 24-hour limit. The batch script
requests `--requeue` and a `USR1` warning two minutes before the time limit. Its warning handler calls
`scontrol requeue` on the same job. Slurm terminates that allocation's processes and schedules another allocation
with the same experiment directory. A user cancellation does not trigger this handler. Ordinary execution errors
stop the job for inspection rather than creating an endless retry loop. Use `--no-auto-resume` to disable the
time-limit handler for a new experiment.

To use shorter scheduling gaps, add `--time-min 00:30:00` alongside `--time 24:00:00`. Slurm's
[backfill scheduler](https://slurm.schedmd.com/sbatch.html#OPT_time-min) may then assign a shorter allocation of at
least 30 minutes. The job immediately runs the experiment; its time warning and checkpoint recovery still apply.
An earlier start depends on available resources and queue priority.

Recovery uses several completion boundaries:

- **Training:** Hermit restores the last committed step's adapter, GEPA candidate pool, progress, and spending.
  Every step has a checkpoint tag. An unfinished step is replayed; its saved reflection response is reused, so a
  completed proposal call is not purchased again. Reflections from different steps have separate cache scopes.
- **Videos:** the task, generator identity, and video/latent checksums must match before reuse. Encoding and
  manifest publication are atomic. Interrupted or invalid artifacts are retained under `attempts/` and that video
  is regenerated from its seed. Recovery is at the video boundary, not at an individual denoising chunk.
- **Judgments:** completed metric decisions and valid saved API responses are reused using the protocol and frame
  checksums. A missing or interrupted metric is completed without repeating valid decisions.
- **Evaluation:** root and final are separate stages. A successful stage is skipped when a later stage is resumed.

To resume after an execution error or a cancelled allocation, fix its cause and submit the **same run directory**:

```bash
python examples/video_transition/utils/submit.py \
  --resume /absolute/path/to/evolve-run
```

This retains the frozen config and source, model bindings, data, adapter tree, caches, and job resources. It rejects
changed frozen files and duplicate submission of an already pending or active job. It never sets
`trainer.resume_mode=disable`. `progress.json` records the active stage and Slurm restart count;
`submissions/` records manual submissions and `job-complete.json` appears only after all stages succeed.
API spending after the last committed step can be absent from Hermit's restored budget; saved response usage
remains available in the reflection and evaluation caches.
The scheduling-only `--time-min` option is also accepted with `--resume` and saved in `allocation.json`, outside
the immutable experiment snapshot. Later manual submissions reuse that minimum while preserving the original
maximum allocation time and training settings.

The generator is a separate process for each uncached Hermit task. This pilot favors a simple verifiable boundary;
a persistent generator service can reduce model reload overhead later. Vision-judge calls and GPU time are recorded
separately from Hermit's learner budget. Use measured costs to set a bounded experiment size.

## Verify implementation changes

```bash
python -m pytest examples/video_transition/utils/tests
```

Tests cover source-group split isolation, frame timing, strict evaluator output validation, information available
to the controller, plan budgets, cancellation of stale plans, interrupted-file recovery, stage recovery, and the
Slurm warning handler. Tests use fake clients and make no paid calls. The core Trainer tests also interrupt and
resume four-epoch training, including a crash before the first checkpoint.

## References

- [Official LongLive source and examples](https://github.com/NVlabs/LongLive/tree/main/LongLive1.0).
- [SEGUE and OpenTrans-360](https://arxiv.org/abs/2609.38691), especially Tables 12 and 19–26.
- [Hermit task interface](../../docs/framework/task.md) and [GEPA](../../docs/methods/gepa.md).
