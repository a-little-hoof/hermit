---
name: streaming-video-transition
description: Plan short, visually grounded transitions toward the latest instruction for a frozen streaming video generator.
---

# Plan a video transition

Read the recent frames before using the previous prompt as a description of the current state. Identify what is
visibly happening, the persistent entities and their appearance, object contacts, ongoing motion, and camera direction.

Compare the latest user request with that observed state. Keep attributes the user has not requested to change.
Honor requested changes to identity, environment, or motion instead of retaining conflicting old details.

Identify the minimum visible preparation needed to reach the new event. Conclude an incompatible ongoing action,
release an occupied object or contact when necessary, establish the needed position or pose, and introduce missing
entities through a visible arrival. Omit preparation that is already satisfied by the frames.

Write each intermediate prompt as a concrete visible action. Preserve enough entity and setting description for
the generator to follow it. Avoid combining incompatible old and new states or asking for an unexplained teleport.

Use no more stages or generation chunks than the provided budget. Prefer fewer executable stages over several
conditions that expire before the generator can follow them. If no preparation is needed, return an empty plan.
The runtime restores the unchanged latest user prompt after the intermediate stages.
