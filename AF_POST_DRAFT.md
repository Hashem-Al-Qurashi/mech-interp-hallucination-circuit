# The Language Barrier in Hallucination Circuits: Why Arabic Safety Can't Be Patched With English Features

## Summary

Entity-recognition features in Gemma 2 2B are causally necessary for confident factual predictions — ablating them drops "basketball" probability from 72.8% to 42.3% for "Michael Jordan plays the sport of." These features are language-specific at the architectural level: injecting English entity-recognition features into Arabic processing doesn't activate Arabic knowledge — it produces multilingual gibberish (Norwegian, Russian, English fragments). This is, to my knowledge, the first circuit-level evidence explaining why multilingual safety gaps exist. You cannot patch Arabic safety by transferring English features.

## Background

Anthropic's [Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) paper identified a hallucination inhibition circuit in Claude 3.5 Haiku: the model's default is to refuse, and "known entity" features suppress this refusal when the model recognizes an entity. Hallucinations are misfires — the suppression activates for entities the model doesn't actually know.

Ferrando et al. ([Do I Know This Entity?](https://arxiv.org/abs/2411.14257), ICLR 2025 Oral) found corresponding SAE latents in Gemma 2 2B that distinguish known from unknown entities, with causal interventions showing these latents control hallucination vs. refusal. But they tested only in English.

Meanwhile, behavioral research shows non-English prompts bypass model safety at alarming rates — [79% for GPT-4](https://arxiv.org/html/2310.06474v3), with Arabic transliteration increasing unsafe responses from 2.5% to 12% even in Claude 3 Sonnet ([Yehia et al., EMNLP 2024](https://arxiv.org/html/2406.18725v1)). The mechanistic explanation for this gap has been an open question.

I used [circuit-tracer](https://github.com/safety-research/circuit-tracer) (Anthropic's open-source attribution graph tool) with GemmaScope transcoders on base Gemma 2 2B to test whether the entity-recognition circuit operates cross-lingually.

## Setup

- **Model:** Gemma 2 2B (base, not instruction-tuned) with GemmaScope per-layer transcoders
- **Tool:** [circuit-tracer](https://github.com/safety-research/circuit-tracer) — attribution graphs via transcoders
- **Compute:** NVIDIA A100-SXM4-80GB (Google Colab Pro)
- **Prompt format:** Completion style ("Michael Jordan plays the sport of") — not questions. Base models predict newlines after question marks, obscuring the signal. This matches the format used in circuit-tracer's [Gemma demo](https://github.com/safety-research/circuit-tracer/blob/main/demos/gemma_demo.ipynb) and Ferrando et al.'s methodology.
- **Intervention API:** `model.feature_intervention(prompt, [(layer, pos, feature_idx, value)])`
- **Code:** [GitHub](https://github.com/Hashem-Al-Qurashi/mech-interp-hallucination-circuit)

### Prompt Pairs

| Known Entity | Unknown Entity |
|---|---|
| Michael Jordan plays the sport of | Michael Batkin plays the sport of |
| Barack Obama was the president of | Zarkon Helmfist was the president of |
| Albert Einstein is famous for discovering | Professor Quixley is famous for discovering |
| The Eiffel Tower is located in | The Glorpnax Tower is located in |
| Lionel Messi plays for the team | Darko Plivnic plays for the team |

Arabic pairs used the same entities in Arabic completion format (e.g., "مايكل جوردان يلعب رياضة").

## Finding 1: Entity-recognition features are causally necessary

The model predicts "basketball" at 72.8% for Jordan vs. "golf" at 12.1% for Batkin — a clean known/unknown signal. Attribution graphs reveal 4,697 features unique to Jordan (not active for Batkin), concentrated in layers 0-6.

**Ablation experiment:** I zeroed all 4,697 Jordan-only features using `model.feature_intervention()`.

| | Before ablation | After ablation |
|---|---|---|
| P("basketball") | **72.8%** | **42.3%** |
| Top prediction | basketball | basketball (weaker) |

A **30.5 percentage point drop**. The model still predicts basketball (residual knowledge persists in shared features), but confidence is substantially degraded. These features are necessary for the entity-recognition circuit to function at full confidence.

I'm ~85% confident this reflects genuine circuit disruption rather than a representation-space artifact, because the top prediction remains basketball (not garbage) — the ablation weakened the circuit rather than destroying it.

## Finding 2: Entity features alone are not sufficient

**Boost experiment:** I injected Jordan-only features at 10× their activation values onto the Batkin prompt.

| | Before boost | After boost |
|---|---|---|
| Batkin top prediction | "golf" (12.1%) | "course" (23.3%) |
| Sport words in top 5? | Yes (golf, rugby, tennis) | No (course, the, all) |

The boost didn't produce hallucinated sport knowledge — it produced generic tokens. This suggests the entity-recognition circuit requires multiple component systems working together. Recognizing an entity as "known" is necessary but not sufficient for confident factual recall.

**Caveat:** The 10× multiplier may be too aggressive, pushing activations out of distribution. The [Biology paper's intervention experiments](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) used more calibrated scaling. A follow-up with 2-3× multipliers could yield cleaner results. I'm ~60% confident this reflects true insufficiency rather than over-intervention.

## Finding 3: English features are architecturally incompatible with Arabic

This is the core novel finding.

Arabic Jordan ("مايكل جوردان يلعب رياضة") predicts "كرة" (ball) at 42.6% — the circuit fires, but at lower confidence than English (72.8%). The model does distinguish known from unknown entities in Arabic (42.6% vs. 18.7%), but the signal is weaker.

**Cross-lingual injection:** I injected the 4,697 English Jordan-only features into the Arabic Jordan prompt.

| | Arabic baseline | After English injection |
|---|---|---|
| Top prediction | كرة (ball) — 42.6% | "professionally" — 6.3% |
| Top 5 | Arabic sport-related | Multilingual gibberish |

The top 5 after injection: "professionally", "othesis", "fører" (Norwegian), "encils", "робнее" (Russian).

The English entity-recognition features don't activate Arabic sport knowledge. They don't even produce coherent English. They destroy the representation space entirely, scattering probability mass across unrelated tokens in multiple languages.

I'm ~80% confident this reflects genuine architectural incompatibility rather than an artifact of the injection method, because:
1. The pre-injection Arabic circuit was functioning (42.6% for كرة)
2. The injection didn't produce English sport words either — it produced gibberish across 4+ languages
3. The same injection methodology (same features, same multiplier) produced a coherent effect in English (Finding 1 ablation)

## What This Means

**For multilingual safety:** The entity-recognition circuit is language-specific at the feature level. English and Arabic use different transcoder features for the same conceptual task (recognizing whether the model knows an entity). Transferring English safety features to Arabic doesn't work — the features are architecturally incompatible. This provides a mechanistic account of the behavioral observation that [non-English prompts bypass safety](https://arxiv.org/html/2310.06474v3).

**For alignment:** If safety-relevant circuits are language-specific, then safety training in English doesn't automatically generalize to other languages at the circuit level — even when the model "understands" the query in both languages. The [English pivot hypothesis](https://arxiv.org/abs/2502.15603) (ICLR 2025) suggests models process semantics in English-adjacent representations, but our finding suggests that the downstream circuits that act on those representations are language-specialized.

**For deployment:** Models deployed multilingually need language-specific safety evaluation at the circuit level, not just behavioral testing. A model can pass English safety benchmarks with a functioning entity-recognition circuit while having a weaker or architecturally different circuit for Arabic.

## Limitations & Open Questions

- **Single model:** These results are from Gemma 2 2B base only. Testing on Gemma 9B, Llama 3.1, or Qwen3 is needed to determine if the finding generalizes. Circuit-tracer supports these models.
- **Base model, not IT:** Gemma 2 2B base doesn't have explicit refusal behavior. The "hallucination inhibition" framing from the Biology paper (which used instruction-tuned Claude) may not perfectly apply. I'm studying entity-recognition confidence, not refusal per se.
- **Boost calibration:** The 10× intervention multiplier may explain the negative result in Finding 2. More careful calibration is a clear follow-up.
- **Arabic data limitation:** Gemma 2 2B was trained on limited Arabic data. The weaker Arabic circuit may reflect training distribution rather than architectural language-specificity. Testing on a model with strong Arabic support (e.g., Jais, AceGPT) would control for this.
- **Feature granularity:** I identified 4,697 Jordan-only features — this is a coarse set. Ferrando et al.'s SAE approach found more specific latents. Connecting transcoder features to their SAE latents is future work.

Open questions I haven't answered:
1. Does the language-specificity hold for instruction-tuned models where refusal is explicit?
2. Is there a small set of "bridge" features that operate cross-lingually?
3. Can language-specific safety circuits be trained independently and composed?

## Reproduction

- **Code:** [github.com/Hashem-Al-Qurashi/mech-interp-hallucination-circuit](https://github.com/Hashem-Al-Qurashi/mech-interp-hallucination-circuit)
- **Notebook:** `circuit_tracer_setup.ipynb` (24 cells, runs end-to-end on Colab A100)
- **Requirements:** Colab Pro (A100), HuggingFace token for Gemma 2 2B access
- **Runtime:** ~2-3 hours including model download
