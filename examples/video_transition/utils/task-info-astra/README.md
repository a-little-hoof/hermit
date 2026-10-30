# Full-batch shared prompt-transition skill

Each update evaluates all ten training transitions (six source scenes) and combines their trajectories and
feedback to revise one shared skills/transition/SKILL.md. Acceptance requires a higher average score on that
same training batch. Model weights, user instructions, acting code, and scheduling remain fixed.

Four validation prompt pairs from four separate source scenes use seeds 0, 1, and 2, making twelve validation
tasks. Two additional test pairs use the same three seeds, making six test tasks. No source scene crosses
splits. Test tasks are evaluated only for the initial and selected final adapters after training.

The fixed judge is GPT-6 Astra with max reasoning, invoked with the supplied ordered frames through the
signed-in Codex CLI. It uses the existing eight SEGUE metrics, original user targets, ten binary judgments
per metric, and unchanged frame sampling. A logo has no applicable human/animal subject preservation score.

Ten training tasks at batch size ten give one update per epoch. Forty epochs give forty updates.
Video generation uses a frozen LongLive model and waits for offline transition plans. Only the shared skill
document can evolve; do not memorize particular captions, source rows, or seed identifiers.
