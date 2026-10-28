# Streaming video transition controller

The acting pipeline must read `skills/transition/SKILL.md` and include it in the controller's instructions.
The skill governs transition planning for the frozen video generator. Follow the latest arrived user instruction
and only the visual state already generated. Model settings, evaluator rules, and the runtime scheduler are fixed.
