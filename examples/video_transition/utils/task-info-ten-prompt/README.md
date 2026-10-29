# Shared prompt-transition skill across ten training transitions

Evolve one shared skills/transition/SKILL.md for a frozen LongLive video generator and controller.
Each task has two original official user prompts, a requested switch at ten seconds, and twenty seconds of video.
The controller reads the generated prefix and plans before video generation resumes; planning is offline.

Ten training transitions come from six source scenes. Four validation transitions come from four other scenes,
each evaluated with seeds 0, 1, and 2 (twelve validation tasks). Two test transitions come from two further scenes,
also with seeds 0, 1, and 2 (six test tasks). Source scenes are disjoint across all three splits. Seeds within a
scene measure generation variability; they are not additional independent prompt pairs.

Four epochs over the ten training tasks produce forty training steps. The training order is reproducibly shuffled.
Only the shared skill document is learnable. Video model weights, acting code, scheduling, user targets, and
evaluator rules remain fixed. Optimize general rules rather than memorizing source captions or identifiers.
