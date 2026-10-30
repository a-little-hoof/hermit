# Full-batch evolution with Astra max scoring

This experiment uses ten training transitions per GEPA update, four validation pairs with three seeds each,
and two separate test pairs with three seeds each. Source scenes stay disjoint between splits. With batch
size ten, forty updates correspond to forty epochs. Acting remains serial on one GPU.
Four independent metric requests may run concurrently. A single evolution call can take up to six hours;
verified video and judgment caches preserve progress if a shorter Slurm allocation ends during that call.

The evaluator is `gpt-6-astra`, reasoning effort `max`, using a signed-in Codex CLI. It receives only the
selected metric's ordered frames, fixed rubric, and the original user targets where the metric permits them.
The CLI uses native image attachments and JSON Schema output; shell, browsing, plugins, and other tools are
disabled for scoring. CLI failures do not become scores or completed evolution steps. API key fallback is
disabled. Usage is recorded, and monetary cost is unknown for this authentication route.

The original eight metrics and frame sampling stay fixed. Model, reasoning effort, CLI binary hash, client
implementation, rubric, task, and sampled frame bytes bind the judgment cache. Previous GLM or InternVL
judgments are not reused. Eligible completed videos can be linked into a new experiment after verifying the
unchanged acting code, task, model settings, skill, and video checksum; all are scored afresh with Astra.

The existing single-task-per-step run stays available as historical evidence. The new run starts with the
same initial shared skill and reevaluates its train and validation scores before selecting evolved policies.
Generator, controller, learner, and evaluation rubric remain separately configured.

After `prepare_ten_prompt.py` has created the source-grouped data, prepare this runtime from the previous
offline run. The source run's complete videos are optional reusable acting evidence, not initial judge scores.

```bash
python examples/video_transition/utils/prepare_astra.py --source-run /absolute/path/to/old-evolve-run
```

Run `evaluate_with_recovery` on one completed training video with the new runtime and a fresh destination to
check all eight metrics before allocating a GPU. Then submit using its `evaluation/summary.json`:

```bash
python examples/video_transition/utils/submit_astra.py \
  --runtime .demo/video-transition/ten-prompt-bs10-astra-max/runtime.local.json \
  --source-run /absolute/path/to/old-evolve-run \
  --preflight .demo/video-transition/ten-prompt-bs10-astra-max/preflight/evaluation/summary.json
```

Resume the frozen new run with `submit.py --resume /absolute/path/to/new-evolve-run`. The new experiment
retains the same 24-hour maximum allocation, thirty-minute backfill minimum, and per-step resume behavior.

Relevant official documentation: [Astra capabilities](https://developers.openai.com/api/docs/models/gpt-6-astra)
and [Codex CLI image and JSON Schema options](https://learn.chatgpt.com/docs/cli/reference).
