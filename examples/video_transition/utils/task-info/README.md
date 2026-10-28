# Shared skill for streaming video transitions

Each task is a complete user instruction stream. The frozen generator receives instructions at video-time events;
the fixed vision-language controller reads the actual generated state and the candidate transition skill.
One shared skill must work across different source streams. Only arrived instructions are visible to the controller.

Training feedback reports transition scores, failed criteria, sampled-frame times, and execution errors. Improve
general state observation, intent preservation, reachable preparation, prompt wording, and chunk-budget allocation.
Preserve skill frontmatter. Do not embed task-specific captions or alter the evaluator, targets, or model settings.

The four short official-LongLive prefixes are integration tasks drawn only from training source groups. Their
scores are not evidence of broad generalization. Formal validation and test groups remain separate.
