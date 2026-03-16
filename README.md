# Hallucination Inhibition Circuits in Gemma 2 2B

Replicating Anthropic's hallucination inhibition circuit on open-weight Gemma 2 2B using circuit-tracer, with cross-lingual (Arabic) analysis.

## Key Findings

1. **Entity-recognition features are causally necessary** — ablating Jordan-only features drops "basketball" prediction from 72.8% to 42.3% (30pp drop)
2. **Entity features are NOT sufficient alone** — boosting them onto unknown entities produces garbage, not hallucinated facts. The circuit requires multiple components.
3. **English features are architecturally incompatible with Arabic** — injecting English entity features into Arabic processing produces multilingual gibberish (Norwegian, Russian, English fragments), not Arabic sport knowledge. First mechanistic evidence for cross-lingual safety gaps.

## Paper

Based on: [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) (Anthropic, 2025)

Builds on: [Do I Know This Entity?](https://arxiv.org/abs/2411.14257) (Ferrando et al., ICLR 2025)

## Setup

```bash
# Google Colab with A100 GPU (Colab Pro)
pip install "numpy==2.0.2"
pip install git+https://github.com/safety-research/circuit-tracer.git
pip install "numpy==2.0.2"
# Then: Runtime > Restart session
```

## Files

- `circuit_tracer_setup.ipynb` — Full experiment notebook (24 cells, 3 phases)
- `RESULTS.md` — Verified results with confidence levels
- `RESEARCH_CHECKLIST.md` — Quality gates for reproducibility

## Method

- **Model:** Gemma 2 2B (base) with GemmaScope transcoders
- **Tool:** [circuit-tracer](https://github.com/safety-research/circuit-tracer) (Anthropic open-source)
- **Prompt format:** Completion ("Michael Jordan plays the sport of") — not questions
- **Intervention:** `model.feature_intervention()` for zero-ablation, boosting, cross-lingual injection

## Author

Quraish — [Alignment Forum profile coming soon]

## License

MIT
