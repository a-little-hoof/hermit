# Offline adaptation on one prompt pair

This debugging experiment uses exactly two original LongLive prompts from one source stream: a seated poker
player holds his cards tensely, then flicks them onto the table and leans back with his arms spread in celebration.
The generator pauses at the instruction change while the controller observes preceding frames and plans a short
transition. Video generation then continues. API latency does not consume video time in this protocol.

Training, validation, and test use the same prompt pair with seeds 0, 1, and 2 respectively. This tests adaptation
within one pair and variation in generated states. It does not measure generalization to unseen user prompts.
Four epochs mean four training steps. Only the shared skills/transition/SKILL.md is learnable; model weights,
controller code, scheduler, and evaluator are fixed. Judge feedback always refers to the original user targets.
