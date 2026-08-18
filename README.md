# On-Practice-VLA

On-Practice-VLA is a research project focused on practice-driven iteration.

The project aims to build an embodied agent with foundational knowledge priors. The agent identifies capability gaps through real-world interaction, incrementally updates its existing knowledge representations through interaction feedback, post-practice review, and error correction, and abstracts stable and effective behavior patterns into reusable skills. This process forms a continually evolving learning loop.

The loop can be summarized as:

**Practice -> Review -> Correction -> Knowledge Update -> Skill Consolidation -> Practice**

## Key Research Directions

- Foundational knowledge priors
- Continual learning
- Online learning
- Interaction feedback and error analysis
- Incremental knowledge updates
- Skill abstraction and reuse
- Agent capability evolution

Unlike paradigms that rely on large-scale offline data for one-time training or repeatedly retrain the entire model to update its capabilities, this project emphasizes the gradual and long-term evolution of model capabilities through limited interaction data, online feedback, and skill abstraction.

## Version Roadmap

| Version | Stage | New Capability | Training / Learning Method | Input | Output | Expected Outcome | Example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v0.1.0 | CLIP Baseline | Preserve Simple-CLIP's image-text alignment capability | Image-text contrastive learning | Image + text | Image-text similarity | Enable image-text matching and retrieval, providing a representational basis for vision-language policy learning | Given `red block`, retrieve relevant images |
| v0.2.0 | Action Prediction Head | Add a discrete action prediction head on top of the joint image-text representation | Action classification | Image + instruction | Action class | Establish a mapping from vision-language representations to a discrete action space | “Pick up the red block” -> `pick` |
| v0.3.0 | Single-Step Behavior Cloning | Learn a single-step policy from demonstration trajectories | Behavior Cloning | Current image + instruction | Next action | Learn expert action decisions under a given state | The expert executes `move_left`; the model predicts `move_left` |
| v0.4.0 | Short-Horizon History Modeling | Introduce recent states and actions as contextual information | History-aware BC | Image + instruction + history | Next action | Use temporal context to reduce repetitive actions and improve policy correction | Predict `stop` after multiple consecutive `move_left` actions |
| v0.5.0 | Action Parameterization | Decompose actions into action types and parameters | Multi-head supervised learning | Image + instruction + history | Action type + parameters | Improve action-space expressiveness without enumerating every compound action | `move(direction=left, step=small)` |
| v0.6.0 | Action Chunk Prediction | Predict multiple consecutive actions in a single inference step | Short-sequence supervised learning | Image + instruction + history | Action chunk | Model short-horizon behavior and improve execution continuity and stability | “Approach the cup” -> `move -> align -> stop` |
| v0.7.0 | Interaction Trajectory Logging | Record observations, actions, and task outcomes | Logging only; no training | Execution process | Interaction trace | Establish a real interaction data foundation for subsequent online learning | Record a failed `pick` and the cause: “alignment incomplete” |
| v0.8.0 | Human-in-the-Loop Correction | Introduce human-corrected actions or short trajectories | Incremental fine-tuning on correction samples | Failure samples + human corrections | Corrected policy | Use limited human feedback to reduce the recurrence of similar errors | Correct a failed `pick` to `align -> pick` |
| v0.9.0 | Experience Replay | Combine new correction samples with historical demonstrations | Replay Buffer | Demos + successes + corrections | Updated policy | Mitigate catastrophic forgetting in incremental learning while preserving existing task capabilities | Retain the original red-block task after learning a new tabletop layout |
| v1.0.0 | Online Adaptation | Update the policy incrementally based on interaction feedback | Small-batch online fine-tuning | New interaction data + replay data | Environment-adapted policy | Improve adaptation to environmental changes without leaving the deployed model fixed | Recover performance after a camera-view change through several rounds of correction |
| v1.1.0 | Skill Abstraction | Abstract frequent action sequences into reusable skills | Skill-level BC | Image + instruction + available skills | Skill or low-level action | Improve compositional task execution and experience reuse | `approach + align + pick` -> `pick_object` |
| v1.2.0 | Few-Shot Transfer | Adapt to new tasks using a limited number of demonstrations or corrections | Few-Shot Adaptation | Limited new-task samples + replay | New-task policy | Improve adaptation efficiency and reduce the cost of task transfer | Execute “place the cup next to the plate” from a small number of demonstrations |
| v1.3.0 | Continual Evaluation | Track success rate, forgetting rate, and recovery speed | Automated evaluation + model rollback | Multi-task evaluation set | Model status report | Quantify continual learning performance and support rollback when performance degrades | Automatically roll back when a new version significantly reduces performance on previous tasks |
