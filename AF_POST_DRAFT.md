# Entity-Recognition Feature Steering in Gemma 2 2B: Hallucination Induction and Cross-Lingual Circuit Asymmetry

## Summary

Entity-recognition features in Gemma 2 2B can induce hallucination when boosted onto unknown entities at calibrated strength — basketball probability for a fictional "Michael Batkin" rises from 3.4% to 12.5% at 2× activation, becoming the top prediction. Applying the same technique to Arabic reveals a circuit asymmetry: Arabic entity steering works at 1× (كرة rises from 18.7% to 28.0%) but breaks at 2× (drops to 5.2%), while English still improves at 2×. When Arabic circuits are pushed past their operating range, they produce cross-lingual leakage — German, Spanish, Polish, and Dutch tokens emerge — suggesting non-English features are entangled in the representation space.

## Background

Anthropic's [Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) identified a hallucination inhibition circuit in Claude 3.5 Haiku. Ferrando et al. ([Do I Know This Entity?](https://arxiv.org/abs/2411.14257), ICLR 2025 Oral) found corresponding SAE latents in Gemma 2 2B with causal interventions — but tested only in English. Behavioral research shows non-English prompts bypass model safety at alarming rates ([79% for GPT-4](https://arxiv.org/html/2310.06474v3), [Arabic transliteration increases unsafe responses in Claude 3 Sonnet](https://arxiv.org/html/2406.18725v1)). The mechanistic explanation for this gap has been open. I used [circuit-tracer](https://github.com/safety-research/circuit-tracer) with GemmaScope transcoders to test whether entity-recognition circuits operate comparably in English and Arabic.

## Setup

- **Model:** Gemma 2 2B base with GemmaScope per-layer transcoders
- **Tool:** [circuit-tracer](https://github.com/safety-research/circuit-tracer) (attribution graphs + `model.feature_intervention()`)
- **Compute:** NVIDIA A100-SXM4-80GB
- **Prompt format:** Completion style — "Michael Jordan plays the sport of" (matches [circuit-tracer Gemma demo](https://github.com/safety-research/circuit-tracer/blob/main/demos/gemma_demo.ipynb) and Ferrando et al.)
- **Code:** [github.com/Hashem-Al-Qurashi/mech-interp-hallucination-circuit](https://github.com/Hashem-Al-Qurashi/mech-interp-hallucination-circuit)

## Finding 1: Entity Features Induce Hallucination at Calibrated Strength

4,697 transcoder features are unique to "Michael Jordan plays the sport of" (not active for "Michael Batkin plays the sport of"). Boosting them onto the Batkin prompt:

| Multiplier | P(basketball) for Batkin | Top prediction | vs baseline (3.4%) |
|---|---|---|---|
| Baseline | 3.4% | golf (12.1%) | — |
| **1×** | **8.7%** | golf (14.4%) | **2.6× increase** |
| **2×** | **12.5%** | **basketball** | **3.7× increase** |
| 5× | 1.6% | course (28.3%) | destroyed |
| 10× | 0.6% | course (23.3%) | destroyed |

At 2×, basketball becomes the top prediction for a fictional person the model has never seen. At 5×+, the representation space collapses into generic tokens.

**Important control:** Ablating all 4,697 Jordan-only features drops basketball from 72.8% to 42.3% — but a random set of 4,697 features produces a comparable drop (avg 29.2%, ratio ≈ 1.0×). This means the features operate as a **distributed set** rather than a sparse circuit. The ablation result is general disruption, not entity-specific. The boost result is more specific — entity features at 2× make basketball the top prediction, which random features do not.

## Finding 2: Arabic Entity Steering Shows Circuit Asymmetry

The same boost technique applied to Arabic — using Arabic Jordan-only features on the Arabic Batkin prompt:

| Multiplier | P(كرة) for Arabic Batkin | Top prediction | English comparison at same multiplier |
|---|---|---|---|
| Baseline | 18.7% | كرة (ball) | basketball 3.4% |
| **1×** | **28.0%** | **كرة** | basketball 8.7% |
| **2×** | **5.2%** | ضد (against) | **basketball 12.5% (still works)** |
| 3× | 0.7% | gegen (German) | — |

At 1×, Arabic steering works — كرة increases from 18.7% to 28.0% (1.5× improvement). At 2×, where English circuits are still improving (3.7×), the Arabic circuit breaks. كرة drops to 5.2% and the model shifts to "ضد" (against).

I'm ~70% confident this asymmetry reflects weaker Arabic entity representations rather than a fundamental architectural difference. The Arabic circuit exists and can be steered — it just tolerates a narrower intervention range before collapsing. This is a single entity pair (Jordan/Batkin) — I tested two additional Arabic pairs (Obama, Eiffel Tower) but their target tokens were morphological fragments or incorrect answers, making direct comparison unreliable. More Arabic entity pairs with clean target tokens are needed.

## Finding 3: Cross-Lingual Leakage Under Stress

When Arabic circuits are pushed past their operating range (3×+), the model doesn't produce Arabic gibberish. It produces tokens from other languages:

| Arabic 3× output | Arabic 5× output |
|---|---|
| gegen (German) | dla (Polish) |
| دور (Arabic) | voor (Dutch) |
| في (Arabic) | pentru (Romanian) |
| contra (Spanish) | the (English) |

This cross-lingual leakage suggests Arabic features in the transcoder space are **entangled with other non-English language representations**. When the Arabic circuit is stressed beyond its operating range, probability mass scatters to features shared across low-resource languages rather than staying within Arabic.

## What Didn't Work

**1. Sledgehammer ablation.** Ablating all 4,697 Jordan-only features produces a similar drop as ablating 4,697 random features (ratio ≈ 1.0×). Large-scale ablation is general disruption, not circuit identification.

**2. Shared entity features.** 106 features are active for all three tested known entities (Jordan, Obama, Einstein) and none of the unknowns. Ablating only these 106 produces a 1.8% drop — *less* than random (7.0% avg, ratio 0.26×). These shared features are general-purpose, not the entity-recognition circuit. The circuit appears to be distributed rather than concentrated.

## Implications

**For multilingual safety:** Entity-recognition circuits in Gemma 2 2B operate within a narrower range for Arabic than English. Safety evaluations using activation steering or probing may overestimate robustness when calibrated on English circuits.

**For intervention methods:** The difference between 1-2× (meaningful steering) and 5-10× (representation destruction) is the difference between a useful tool and noise. Calibration is non-negotiable for feature intervention experiments.

**For circuit structure:** The negative ablation results suggest entity recognition in Gemma 2 2B is a distributed computation across thousands of features, not a sparse identifiable circuit. Whether this reflects a real difference from Claude 3.5 Haiku (where Anthropic found identifiable features) or a limitation of per-layer transcoders vs. cross-layer transcoders is an open question.

## Limitations

- **Single model.** Gemma 2 2B base only. Replication on Llama 3.1 1B and Qwen3 4B planned.
- **Base model.** No explicit refusal behavior. I'm studying entity-recognition confidence, not refusal circuits.
- **N=1 clean Arabic entity.** The circuit asymmetry is demonstrated for one clean entity pair (Jordan). More Arabic pairs with interpretable target tokens needed.
- **Arabic training data confound.** The narrower operating range may reflect limited Arabic training data rather than circuit architecture. Testing on models with strong Arabic support would control for this.
- **Distributed circuit.** The negative ablation results mean I haven't identified specific causal features — only demonstrated distributed feature sufficiency for hallucination induction.

Open questions:
1. Does the Arabic circuit asymmetry appear in instruction-tuned models?
2. Is cross-lingual leakage specific to Arabic, or general for all low-resource languages?
3. Can feature importance ranking (by adjacency weight) produce sparse circuits that pass the random control?

## Reproduction

- **Code:** [github.com/Hashem-Al-Qurashi/mech-interp-hallucination-circuit](https://github.com/Hashem-Al-Qurashi/mech-interp-hallucination-circuit)
- **Notebook:** `circuit_tracer_setup.ipynb` (33 cells, 5 experimental phases, all outputs saved)
- **Runtime:** ~3-4 hours on A100 for all phases
