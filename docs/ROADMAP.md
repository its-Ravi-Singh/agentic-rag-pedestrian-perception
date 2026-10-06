# Roadmap: Agentic + Retrieval-Augmented Pedestrian Perception

This repo starts from a PEFT-tuned ViT that recognises eight attributes of a driving scene (Reddi, Kusuma & Parvin, ACM JATS 2026):

- Pedestrian behaviour, on each pedestrian crop: Action, Look, Cross, Occlusion
- Scene context, on the full frame: Weather, Time of Day, Road Presence, Pedestrian Density

Each attribute has its own adapter on a frozen ViT-B/16 backbone. A hand-written scheduler decides which adapters to run on each frame.

The goal of this phase is to replace the hand-written scheduling with a learned, confidence-aware controller, and to train and evaluate the whole system on three datasets instead of one.

Target completion: first week of November 2026.

## Goals

1. **Learned adapter routing.** A small router reads a summary of the frame and decides which adapters to run. No fixed scene rules.
2. **Confidence check and escalation.** If the model is unsure about Look or Cross, it runs additional adapters and checks again. There is at most one extra round, so latency stays bounded.
3. **Scene memory (retrieval).** A FAISS index of previously seen frames and their labels. It is used to:
   - give the router a prior from similar scenes
   - blend nearest-neighbour labels into uncertain predictions
   - act as a cheap fallback when the model is unsure
4. **Three datasets.** JAAD, PIE and BDD100K, trained jointly and reported per dataset.
5. **Decision log.** For every frame: which adapters ran, their confidences, whether it escalated and why, plus a short readable summary.
6. **Same or better accuracy at lower cost** than running every adapter on every frame.

## Datasets

| Dataset | Attributes used | Split |
|---|---|---|
| JAAD | All 8 | Official video splits |
| PIE | Action, Look, Cross, Occlusion, Road Presence | Official set split (set01/02/04 train, set05/06 val, set03 test) |
| BDD100K | Weather, Time of Day, Pedestrian Density | Official train/val, with val used as test |

No single dataset other than JAAD has all eight labels. Each attribute is trained and tested on every dataset that annotates it. Attributes a dataset does not annotate are masked out of the loss and the metrics.

Label mappings live in `configs/`, not in code. For example, BDD100K `overcast` and `partly cloudy` map to `cloudy`. Density and road-width labels use the same cut-offs on every dataset.

All splits are by video or sequence, so frames from the same clip never appear in both train and test.

## What gets replaced

| Currently | Replaced by |
|---|---|
| Scene-type rules that pick a preset group of adapters | Learned router |
| Weather and Time of Day run once per video; Density every 30th frame | Router decides per frame, with the previous frame's results as an input |
| Fixed confidence and ROI padding values | Values selected on the validation split |
| Adapter type and paths chosen in code | `configs/adapters.yaml` |

## Method outline

1. **Feature cache.** Run every adapter once per dataset and store:
   - the frame embedding ([CLS])
   - each adapter's probabilities
   - per-adapter latency
   - the labels

   Router training, baselines and most experiments then run from this cache.
2. **Joint adapter fine-tuning.** Start each adapter from its JAAD checkpoint and fine-tune it on every dataset that has its label. Sampling is balanced across datasets.
3. **Target masks.** For each frame, find the smallest set of adapters that reproduces the full-model output. These sets are the router's training labels.
4. **Router.** A small MLP with eight independent sigmoid outputs, trained with BCE on the target masks.
5. **Calibration.** Temperature-scale each adapter on validation, so the confidence threshold means something.
6. **Escalation.** Fire the router's next-highest-scoring adapters and query the scene memory. One round at most.
7. **Explanations.** Template summaries built from the decision log. LLM-generated explanations grounded in traffic rules are a stretch goal.

## Evaluation

**Baselines:**
- all adapters on every frame
- random-k (same number of adapters as the router fires)
- the original rule-based scheduler
- an oracle router that always uses the target mask

**Metrics:**
- per-attribute accuracy, macro-F1 and calibration error
- routing accuracy against the target masks
- ms/frame, FPS and GFLOPs, broken down by pedestrian count
- all of the above reported per dataset

**Ablations:**
- without escalation
- without the confidence check
- threshold sweep
- router inputs
- scene-memory size and k
- leave-one-dataset-out

## Milestones

| Week | Dates | Work |
|---|---|---|
| 1 | Oct 6 - Oct 12 | Restructure the code into a package with configs; loaders for JAAD, PIE and BDD100K; shared multi-adapter backbone; JAAD feature cache and baseline numbers |
| 2 | Oct 13 - Oct 19 | Joint adapter fine-tuning; target masks; baselines; router, confidence check, escalation and decision log on JAAD |
| 3 | Oct 20 - Oct 26 | Feature caches for all three datasets; joint router; scene memory; per-dataset results |
| 4 | Oct 27 - Nov 2 | Live latency benchmark; ablations; explanation summaries; Gradio demo |
| 5 | Nov 3 - Nov 6 | Results tables in the README; report; cleanup and release |

## Planned layout

```
configs/        dataset, label schema, adapter and router configs
src/apr/
  data/         JAAD, PIE, BDD100K loaders and label mapping
  perception/   shared ViT backbone with adapters, detector, calibration
  agent/        router, confidence check, escalation, decision log
  rag/          scene memory and retrieval
  baselines/    all-adapters, random-k, rule-based, oracle
  eval/         metrics, latency benchmark, plots
scripts/        caching, target masks, training, benchmarks
app/            Gradio demo
legacy/         original step 1-3 scripts
```

## Future work

- A router trained with offline RL or preference weighting (reward = accuracy minus a latency cost)
- More datasets (TITAN, PSI, ACDC)
- Full LLM-based explanations
