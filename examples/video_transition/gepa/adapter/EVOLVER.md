# Evolve a shared video transition skill

Edit only `skills/transition/SKILL.md`. Preserve its YAML frontmatter and return a complete valid skill document.
Use scored trajectories and failed evaluation criteria to improve general planning rules. The same skill is used
across streams. Do not memorize individual prompts, source row identifiers, outputs, or seeds.

Explain how to preserve user intent while choosing reachable intermediate changes under a bounded chunk budget.
The runner handles instruction arrival and cancellation; the skill should improve planning from the observed state.
Do not change evaluation targets, model settings, executable code, or data splits.
