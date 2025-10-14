# Categories description file 


## in short

Spatial reasoning, Kinematics, Forces, Contact, Material, Mass, Visibility, Event ordering, Global.

## core categories

### Spatial Reasoning
View‑invariant spatial relations and metrics: distances, nearest/left‑of/inside/containment, path/clearance. (Uses 1), but asks relational questions.)

### Kinematics
State derivatives: linear/angular velocity, acceleration, path length vs. displacement, turning rate. (No forces here.)

### Forces & Dynamics
Non‑contact dynamics: gravity vector/magnitude, net external force (ex‑impulses), momentum/energy changes along free motion.

### Contact & Collision
Contact events and outcomes: first‑contact time, contact duration, impulses, sticking vs. sliding, bounce/apex height, collision counts.

### Material & Deformation
Intrinsic/interfacial parameters and deformations: μ_s/μ_k, restitution e, Young’s E, Poisson ν, damping, yield σ_y; stresses/strains and deflections.

### Mass & Volume (added)
Mass m, volume V, density ρ, inertia tensor I; shape dimensions. (Keep ρ here to avoid coupling with 6).)

### Visibility & View Geometry
What the camera sees: per‑camera visibility flags, visible counts, occluder ID, visible fraction, reprojection error. (View‑dependent only.)

### Event Ordering & Temporal Reasoning
Temporal relations: before/after/during, event order, temporal sorting, phase labels.
Not included here: durations (e.g., “how long”)—those live in the owning category (5 for contact time, 3 for motion phases, etc.).

### Global Plausibility (meta‑checks)
Physical sanity constraints: non‑penetration, friction‑cone adherence, energy/momentum consistency for isolated subsystems. (Aggregates signals from 3–7.)


## Small note on the categories.

There are quite some overlap in temporal and spatial with all the rest of the categories, but we will keep them as categories as well.
Temporal reasoning -> anything that requires to reason about time (e.g., before, after, during, etc.), or a time sequence of events.
What is not comprised here is the measure of time (e.g., how long did it take to fall, etc.) which is included in the other categories (e.g., forces, collisions, etc.)

Spatial reasoning -> anything that requires to reason about space (e.g., distance, location, etc.)

Also, this should also be a bootstrap version. The following is the prompt to get codex to generate 10X variations of the questions in each category.

```bash 
You are an expert in generating diverse and creative questions for visual question answering (VQA) tasks. 
Your task is to generate 10 unique variations of the given question while maintaining its original intent and context. 
Ensure that each variation is distinct in wording and structure, but still clearly relates to the same underlying concept. 
Avoid using the same phrases or sentence structures as the original question. 
Each question has a question_ID e.g. F_VISIBILITY_COUNTING, when creating the variation 
make sure to append a suffix _1, _2, etc. to the question_ID to make it unique.

The file name is /vqa/simpler_extended.json and you should save it as /vqa/simpler_extended_variation.json
```

## Note for the code

So far we have the following attributes:
- <OBJECT> -> defines any unique object
    <OBJECT_1> -> if there are multiple objects, _N will define the ID of the object
- <OBJECT_TYPE> -> defines the category of objects (multiple unique <OBJECT>)
- <TIME> -> the timestamp we want to measure
- <MATERIAL> -> the material of the object
- <MASS> -> the mass of the object.
- <VOLUME> -> volume of the obejct
- <SCENE> -> could be the scene itself or any segmented part of the scene