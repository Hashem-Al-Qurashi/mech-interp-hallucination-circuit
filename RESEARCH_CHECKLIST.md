# Research Quality Checklist

Run this BEFORE every experiment session. No exceptions.

## Before Designing Experiments

- [ ] **Prior work search** — Has anyone done this exact thing? Search AF, LW, arXiv, ICLR/NeurIPS for: model + method + phenomenon
- [ ] **What's the gap?** — Write one sentence: "Nobody has done X because Y." If you can't, the gap doesn't exist.
- [ ] **What's the novel contribution?** — Not "we ran circuit-tracer on Gemma." That's replication. The contribution is the NEW thing you learn.

## Before Running Experiments

- [ ] **What's the causal question?** — "If I intervene on X, does Y change?" If there's no intervention, it's descriptive, not research.
- [ ] **What would disprove my hypothesis?** — If nothing can disprove it, it's not a hypothesis.
- [ ] **One deep or many shallow?** — Choose depth. One prompt pair with full ablation > 20 pairs with counting.

## Before Claiming Findings

- [ ] **Rigor level check:**
  - Level 1 (Descriptive): "X activates here" — NOT a finding, it's an observation
  - Level 2 (Interpretive): "X represents concept Y" — Needs evidence
  - Level 3 (Causal): "Intervening on X causes Y" — THIS is the minimum for an AF post
  - Level 4 (Mechanistic): "X inhibits Y via pathway Z" — This is real research
  - Level 5 (Generalizable): "This holds across models/tasks" — This is a paper
- [ ] **Would Neel Nanda respond to a DM about this?** — If no, it's not ready.
- [ ] **Does this survive the question: "So what?"** — What does this mean for AI safety?

## The 3 Questions That Would Have Caught Our Mistake

1. "Has the ICLR 2025 'Do I Know This Entity?' paper already done this?"
2. "Is counting features the same as understanding the circuit?"
3. "Can I causally prove that these features CAUSE hallucination, or am I just observing correlations?"
