/* Paper-reported results: fine-tuning, merging, and pruning tables. Values are similarity scores, not accuracy. */
window.ATTNDIFF_RESULTS = {
  fine: {
    title: "01 / Fine-tuning", source: "Table 1 · Llama-2-7B derivatives",
    description: "Six downstream models span 5 million to 700 billion fine-tuning tokens. Does a fingerprint survive substantial domain adaptation?",
    takeaway: "AttnDiff retains similarity above 0.985 across all six derivatives. REEF is also strong, with a slightly higher mean (0.9966 vs. 0.9945).",
    rows: [
      ["Llama-2-Finance · 5M tokens", 0.9989, 0.9950],
      ["Vicuna-1.5 · 370M tokens", 0.9986, 0.9985],
      ["WizardMath · 1.8B tokens", 0.9985, 0.9979],
      ["Chinese-LLaMA-2 · 13B tokens", 0.9963, 0.9974],
      ["CodeLLaMA · 500B tokens", 0.9890, 0.9947],
      ["Llemma · 700B tokens", 0.9856, 0.9962]
    ]
  },
  merge: {
    title: "02 / Model merging", source: "Table 2 · each merged suspect compared with its parents",
    description: "Test both weight-space merging (Evollm-jp-7B) and distribution-level merging across heterogeneous architectures (FuseLLM-7B).",
    takeaway: "AttnDiff scores 0.9561–0.9996 for weight-merge parents. For FuseLLM, similarity to OpenLLaMA and MPT is lower (0.7953 / 0.7851), but exceeds REEF on both comparisons.",
    rows: [
      ["Evollm ← Shisa-gamma", 0.9726, 0.9635],
      ["Evollm ← WizardMath", 0.9561, 0.9526],
      ["Evollm ← Abel", 0.9996, 0.9374],
      ["FuseLLM ← Llama-2", 0.9962, 0.9996],
      ["FuseLLM ← OpenLLaMA-2", 0.7953, 0.6713],
      ["FuseLLM ← MPT", 0.7851, 0.6200]
    ]
  },
  prune: {
    title: "03 / Model pruning", source: "Table 3 · structured and unstructured pruning",
    description: "Change both the weights and the architecture: six Sheared-LLaMA checkpoints test structural compression and recovery; three sparse models test unstructured pruning.",
    takeaway: "AttnDiff maintains at least 0.9879 similarity across these nine suspects, including models with different depths and head counts.",
    rows: [
      ["Sheared 1.3B · pruned", 0.9879, 0.9368],
      ["Sheared 1.3B · recovered", 0.9938, 0.9676],
      ["Sheared 1.3B · ShareGPT", 0.9903, 0.9710],
      ["Sheared 2.7B · pruned", 0.9929, 0.9278],
      ["Sheared 2.7B · recovered", 0.9952, 0.9701],
      ["Sheared 2.7B · ShareGPT", 0.9936, 0.9991],
      ["Sparse-Llama-2", 0.9996, 0.9985],
      ["Wanda-Llama-2", 0.9927, 0.9986],
      ["GBLM-Llama-2", 0.9995, 0.9991]
    ]
  }
};
