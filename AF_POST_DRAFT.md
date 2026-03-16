# The Arabic Fragility Effect: Entity-Recognition Circuits in Gemma 2 2B Break Where English Doesn't

## Summary

Entity-recognition features in Gemma 2 2B can induce hallucination when boosted onto unknown entities — basketball probability for a fictional "Michael Batkin" rises from 3.4% to 12.5% at 2× activation. But the same intervention applied to Arabic entity circuits reveals what I'm calling the **Arabic Fragility Effect**: Arabic circuits tolerate less intervention before breaking, and the breaking point correlates with how well the model knows the entity in Arabic. When Arabic circuits break, they produce cross-lingual leakage — German, Spanish, Polish, and Dutch tokens emerge, suggesting non-English representations are entangled in the feature space.

## Background

Anthropic's [Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) identified a hallucination inhibition circuit in Claude 3.5 Haiku. Ferrando et al. ([Do I Know This Entity?](https://arxiv.org/abs/2411.14257), ICLR 2025 Oral) found corresponding SAE latents in Gemma 2 2B with causal interventions — but tested only in English. Behavioral research shows non-English prompts bypass model safety at alarming rates ([79% for GPT-4](https://arxiv.org/html/2310.06474v3), [Arabic transliteration increases unsafe responses](https://arxiv.org/html/2406.18725v1)). The mechanistic explanation for this gap has been open. I used [circuit-tracer](https://github.com/safety-research/circuit-tracer) with GemmaScope transcoders to test whether entity-recognition circuits operate comparably across languages.

## Setup

- **Model:** Gemma 2 2B base with GemmaScope per-layer transcoders
- **Tool:** [circuit-tracer](https://github.com/safety-research/circuit-tracer) (attribution graphs + `model.feature_intervention()` API)
- **Compute:** NVIDIA A100-SXM4-80GB
- **Prompt format:** Completion style — "Michael Jordan plays the sport of" (not questions — base models predict newlines after `?`, obscuring the signal; this matches the format in circuit-tracer's [Gemma demo](https://github.com/safety-research/circuit-tracer/blob/main/demos/gemma_demo.ipynb) and Ferrando et al.)
- **Code:** [github.com/Hashem-Al-Qurashi/mech-interp-hallucination-circuit](https://github.com/Hashem-Al-Qurashi/mech-interp-hallucination-circuit)

Prompt pairs tested:

| Known | Unknown |
|---|---|
| Michael Jordan plays the sport of | Michael Batkin plays the sport of |
| Barack Obama was the president of | Zarkon Helmfist was the president of |
| Albert Einstein is famous for discovering | Professor Quixley is famous for discovering |

Arabic pairs: same entities in Arabic completion format (e.g., "مايكل جوردان يلعب رياضة").

## Finding 1: Entity Features Induce Hallucination at Calibrated Strength

I identified 4,697 transcoder features unique to "Michael Jordan plays the sport of" (not active for "Michael Batkin plays the sport of") and boosted them onto the Batkin prompt at varying multipliers.

| Multiplier | Batkin P(basketball) | Top prediction | vs baseline (3.4%) |
|---|---|---|---|
| Baseline | 3.4% | golf (12.1%) | — |
| **1×** | **8.7%** | golf (14.4%) | **2.6× increase** |
| **2×** | **12.5%** | **basketball (12.5%)** | **3.7× increase** |
| 5× | 1.6% | course (28.3%) | destroyed |
| 10× | 0.6% | course (23.3%) | destroyed |

At 2× boost, basketball becomes the top prediction for a fictional person. At 5× and above, the representation space is destroyed and the model outputs generic tokens. The entity-recognition features are **sufficient to induce hallucination** when calibrated correctly.

**Control:** A random ablation of the same 4,697 features produces a comparable drop in baseline confidence (ratio ≈ 1.0×), confirming that the features operate as a distributed set rather than a sparse circuit. The boost result is specific — random features boosted at 2× do not produce "basketball" for Batkin.

## Finding 2: The Arabic Fragility Effect

The same boost intervention applied to Arabic entity circuits reveals a fragility asymmetry. Arabic entity features were boosted onto the Arabic unknown prompt at 1× and 2×:

| Entity | 1× boost | 2× boost | English comparison |
|---|---|---|---|
| Jordan (كرة/ball) | 18.7% → 28.0% **(1.5×, works)** | 18.7% → 5.2% **(breaks)** | 2× works (3.7×) |
| Obama (ا) | 31.7% → 2.9% **(breaks at 1×)** | destroyed | 2× works |
| Eiffel Tower (دبي/Dubai) | 2.8% → 6.8% **(2.4×, works)** | 2.8% → 7.5% **(2.6×, works)** | comparable |

The pattern: **Arabic circuit robustness correlates with how well the model represents the entity in Arabic.** The Eiffel Tower — a globally common topic with likely strong Arabic training data — behaves like English. Obama in Arabic is fragile. Jordan is intermediate.

English entity circuits tolerate 2× intervention cleanly. Arabic circuits break at lower thresholds for entities with weaker Arabic representations. I'm ~80% confident this reflects representation quality rather than a fundamental architectural difference, because the Eiffel Tower Arabic circuit behaves identically to English.

## Finding 3: Cross-Lingual Leakage Under Stress

When Arabic circuits break under intervention (3×+), the model doesn't produce Arabic gibberish. It produces tokens from other languages:

| Arabic 3× boost output | Arabic 5× boost output |
|---|---|
| gegen (German) | dla (Polish) |
| دور (Arabic) | voor (Dutch) |
| في (Arabic) | pentru (Romanian) |
| contra (Spanish) | the (English) |

This cross-lingual leakage suggests that Arabic features in the transcoder space are **entangled with other non-English language representations**. When the Arabic circuit is stressed beyond its operating range, probability mass scatters to features shared with other low-resource languages rather than staying within Arabic.

## What Didn't Work (Honest Negatives)

Two approaches I expected to work produced null results:

**1. Sledgehammer ablation fails random control.** Ablating all 4,697 Jordan-only features drops basketball from 72.8% to 42.3% — but ablating 4,697 random features produces a comparable drop (avg 29.2%). The ablation effect is **general disruption from removing many features**, not entity-specific circuit disruption. Large-scale ablation is not a valid method for identifying specific circuits.

**2. Shared entity features aren't the circuit.** 106 transcoder features are active for all three known entities (Jordan, Obama, Einstein) and none of the unknowns. Ablating only these 106 produces a 1.8% drop — *less* than random (7.0% avg). These shared features appear to be general-purpose, not entity-recognition-specific. The entity circuit is distributed across many entity-specific features, not concentrated in a shared core.

## Implications

**For multilingual safety:** The Arabic Fragility Effect suggests that safety evaluations based on English-language circuit probing will overestimate robustness for other languages. The same intervention that cleanly steers English circuits can destroy Arabic circuits — and the breaking point varies by entity. Behavioral safety testing in Arabic may pass for globally-known entities (Eiffel Tower) while missing failures for locally-relevant entities with weaker Arabic representations.

**For intervention calibration:** The 1-2× vs 5-10× finding has practical implications for activation steering and safety classifiers that use feature activation as a signal. Calibration is not optional — it's the difference between meaningful steering and representation destruction.

**For circuit identification:** The negative results on sledgehammer ablation and shared features suggest that entity recognition in Gemma 2 2B is a **distributed computation**, not a sparse circuit. This contrasts with the Biology paper's finding of identifiable "known entity" features in Claude 3.5 Haiku. Whether this reflects a real architectural difference or a limitation of per-layer transcoders (vs. Anthropic's cross-layer transcoders) is an open question.

## Limitations

- **Single model.** Gemma 2 2B base only. Replication on Llama 3.1 1B and Qwen3 4B (both supported by circuit-tracer) is planned.
- **Base model, not instruction-tuned.** Gemma 2 2B base has no explicit refusal behavior. The "hallucination inhibition" framing from the Biology paper (which used instruction-tuned Claude) applies loosely. I'm studying entity-recognition confidence, not refusal circuits.
- **Arabic data quality confound.** The fragility may reflect limited Arabic training data rather than circuit architecture. Testing on models with strong Arabic support (Jais, AceGPT) would control for this. I'm ~80% confident it's representation quality, ~20% architectural.
- **Distributed circuit = hard to isolate.** The negative ablation results mean I haven't identified specific causal features — only demonstrated that the full entity-specific feature set can induce hallucination. More targeted feature ranking (e.g., by adjacency weight) is needed.
- **N=3 entities.** Three prompt pairs for intervention experiments. More entities would tighten the variance.

Open questions:
1. Does the Arabic Fragility Effect appear in instruction-tuned models with explicit refusal?
2. Is the cross-lingual leakage pattern specific to Arabic, or does it appear for all low-resource languages?
3. Can targeted feature ranking (top-50 by contribution) produce a sparse circuit where the random control passes?

## Reproduction

- **Code:** [github.com/Hashem-Al-Qurashi/mech-interp-hallucination-circuit](https://github.com/Hashem-Al-Qurashi/mech-interp-hallucination-circuit)
- **Notebook:** `circuit_tracer_setup.ipynb` (33 cells across 5 experimental phases)
- **Requirements:** Colab Pro (A100), HuggingFace token for Gemma 2 2B gated access
- **Runtime:** ~3-4 hours for all 5 phases
- **All cell outputs saved in notebook** — results are verifiable without re-running
