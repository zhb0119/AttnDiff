<div align="center">

# AttnDiff
### Attention-based Differential Fingerprinting for Large Language Models

**NeurIPS 2026**

[![Paper](https://img.shields.io/badge/arXiv-2604.05502-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.05502)
[![Project Page](https://img.shields.io/badge/Project-Page-216557?logo=googlechrome&logoColor=white)](https://zhb0119.github.io/AttnDiff/)
[![Video](https://img.shields.io/badge/▶_Video-20_seconds-E7AB46?labelColor=273D39)](https://zhb0119.github.io/AttnDiff/#demo)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-216557)](LICENSE)

**Haobo Zhang**<sup>1,3,*</sup> · **Zhenhua Xu**<sup>2,3,*</sup> · **Junxian Li**<sup>4</sup><br>
**Shangfeng Sheng**<sup>5</sup> · **Dezhang Kong**<sup>2,3,†</sup> · **Meng Han**<sup>2,3,†</sup>

<sup>1</sup> Renmin University of China<br>
<sup>2</sup> Zhejiang University · <sup>3</sup> Binjiang Institute of Zhejiang University<br>
<sup>4</sup> Shanghai Jiao Tong University · <sup>5</sup> University of Science and Technology of China

<sup>*</sup> Equal contribution &nbsp; <sup>†</sup> Corresponding authors<br>
[haobo0119@gmail.com](mailto:haobo0119@gmail.com) · [xuzhenhua0326@zju.edu.cn](mailto:xuzhenhua0326@zju.edu.cn) · [mhan@zju.edu.cn](mailto:mhan@zju.edu.cn)

**Trace model provenance through differential attention.**

<a href="docs/assets/teaser.png"><img src="docs/assets/teaser.webp" width="100%" alt="AttnDiff traces the provenance of transformed models by comparing independently extracted reference and suspect attention fingerprints."></a>

<sub>Model families, fingerprint patterns, and the highlighted match are illustrative.</sub>

[Overview](#overview) · [Video](#video) · [Method](#method) · [Quick Start](#quick-start) · [Results](#results) · [Citation](#citation)

</div>

<a id="overview"></a>

## 🔎 Overview

AttnDiff is a **data-efficient, white-box** framework for verifying whether a suspect LLM derives from a reference model. It probes minimally edited prompt pairs that induce semantic conflicts, captures the resulting **attention differences**, summarizes them with **spectral descriptors**, and compares fingerprints using **centered linear CKA**.

- **Intrinsic fingerprints:** no watermark embedding, model training, or parameter modification.
- **Compact probes:** the paper defaults to 60 prompt pairs across six domains, with the top 3 singular values per layer/head.
- **Flexible comparison:** models can have different layer/head counts. CKA compares their geometry over the same ordered probes.
- **Broad evaluation:** fine-tuning, PPO/DPO, merging, pruning, distillation, quantization, cross-family comparisons, and a fixed-budget adaptive suppression pilot.

<a id="video"></a>

## 🎬 Video

<div align="center">
<a href="https://zhb0119.github.io/AttnDiff/#demo"><img src="docs/assets/video/preview.gif" width="800" alt="Animated preview: released models, transformations, semantic-conflict probes, and attention fingerprint comparison. Click to watch the narrated video."></a>

**[▶ Video](https://zhb0119.github.io/AttnDiff/#demo)** · [Download MP4](https://raw.githubusercontent.com/zhb0119/AttnDiff/main/docs/assets/video/AttnDiff_story_20s_narrated.mp4)

<sub>English narration · English and Chinese subtitles on the project page</sub>
</div>

<a id="method"></a>

## 🧬 Method

<a href="docs/assets/pipeline.png"><img src="docs/assets/pipeline.webp" width="100%" alt="The five-step AttnDiff workflow: prompt pairs, causal attention, aligned attention differences, top-K singular values, and centered CKA comparison."></a>

1. **Construct paired probes.** A single-word pivot creates a controlled semantic conflict: “parallel lines **never** intersect” → “parallel lines **always** intersect.”
2. **Extract attention.** Collect post-softmax causal self-attention maps across all layers and heads.
3. **Compute differences.** In the paper, adaptive average pooling aligns each pair to the smaller token length, then `ΔA = Pool(A_corrupted) − Pool(A_original)`.
4. **Build spectral fingerprints.** Keep the top `K = 3` singular values per layer/head and stack `M` probe descriptors into `F ∈ ℝ^(M × LHK)`.
5. **Compare models.** Independently extract both fingerprints using the **same probe identities and ordering**, then compare their centered `M × M` Gram matrices with linear CKA.

The method requires internal attention activations. Text-only APIs are outside the evaluated setting. Similarity provides evidence of shared lineage; decision thresholds should be calibrated for the model pool and verification setting.

<a id="quick-start"></a>

## 🚀 Quick Start

### Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
git clone https://github.com/zhb0119/AttnDiff.git
cd AttnDiff
uv sync
```

Alternatively, use `python -m pip install -e .`. Run commands from the repository root. Multiline examples use Bash syntax; in PowerShell, use a single line or replace `\` with a backtick.

### Compare the included example fingerprints

```bash
uv run attndiff-compare \
  --base output/comput_W/fingerprint_Llama-2-7B.json \
  --dir output/comput_W \
  --cka linear
```

This compares stored fingerprints without loading model weights or extracting new attentions. These are example artifacts, not a complete reproduction of the paper experiments.

### Extract spectral fingerprints

Use a Hugging Face model identifier or local checkpoint path:

```bash
uv run attndiff-compute \
  --model_name /path/to/your/model \
  --attn_device cuda:0 \
  --mode diff \
  --svd_k 3 \
  --out output/spectral/fingerprint_your_model.json
```

The extractor reads `dataset/dataset.json` and can reuse attention JSONs under `output/attention/`. Use separate attention paths when changing probes so that old cached attentions are not reused.

To start from pre-extracted attention maps:

```bash
uv run attndiff-compute \
  --original output/attention/model_att_origin.json \
  --corrupted output/attention/model_att_perturb.json \
  --mode diff \
  --svd_k 3 \
  --out output/spectral/fingerprint_model.json
```

After extracting reference and suspect fingerprints into the same directory:

```bash
uv run attndiff-compare \
  --base output/spectral/fingerprint_reference.json \
  --dir output/spectral \
  --cka linear
```

Keep the same representation and ordered probe set across compared files. Use `--layer 1` for an optional layer-specific comparison (1-based).

| Extraction option | Purpose |
| --- | --- |
| `--model_name` | Model identifier or local checkpoint path |
| `--original`, `--corrupted` | Pre-extracted attention JSON files |
| `--mode diff` | Corrupted-minus-original attention |
| `--svd_k 3` | Enable top-3 singular-value descriptors |
| `--attn_device` | Extraction device, e.g. `cuda:0` |
| `--out` | Fingerprint output path |

For batch runs, configure model paths and device in [`scripts/batch_compute.sh`](scripts/batch_compute.sh), and add `--svd_k 3` to its extraction command to use spectral descriptors.

### Probe format and reproduction notes

The input is a JSON list:

```json
[
  {
    "id": 1,
    "topic": "Mathematics",
    "original": "In Euclidean geometry, two distinct parallel lines never intersect, no matter how far they are extended.",
    "corrupted": "In Euclidean geometry, two distinct parallel lines always intersect, no matter how far they are extended."
  }
]
```

- The checked-in `dataset/dataset.json` contains **30 pairs across five domains**. The paper defaults to **60 pairs across six domains**: Code, Math, Economics, Medicine, Daily QA, and Safe Alignment (10 pairs each).
- Set `--svd_k 3` explicitly: the current CLI does not enable spectral descriptors by default. Do not mix these outputs with legacy scalar fingerprints.
- For unequal token lengths, the current CLI's SVD path **crops** maps to the common minimum size; the paper method specifies **adaptive average pooling**. Exact reproduction requires resolving this implementation difference. Equal-length pairs do not need spatial alignment.

See the paper's implementation appendix for the protocol and [`dataset/README.md`](dataset/README.md) for the schema.

<a id="results"></a>

## 📊 Results

The following are **paper-reported results**, not measurements produced by the quick-start example.

| Evaluation | Reported result |
| --- | --- |
| Fine-tuning | CKA **0.9856–0.9989** across six Llama-2-7B derivatives |
| Pruning | CKA **≥ 0.9879** across nine structured/unstructured pruned suspects |
| Weight-space merging | CKA **0.9561–0.9996** to the evaluated parents |
| Discriminability | **0.9883 AUROC / 0.9881 AUPR**, on 36 related + 33 unrelated suspects |
| Adaptive suppression | Refreshed-probe CKA **0.988–0.991**, under 20–80% exposure and a fixed 96-step LoRA budget |

The adaptive study covers one related Gemma-2 model pair. Its findings support private, refreshed probes within that setting; they do not establish robustness to unrestricted adaptive attacks. AUROC/AUPR are ranking metrics, not accuracy percentages.

**[Explore interactive results and ablations →](https://zhb0119.github.io/AttnDiff/#results)**<br>
**[Browse model families and checkpoints →](docs/model-zoo.md)**

## 📁 Repository

```text
AttnDiff/
├── src/attndiff/       # Fingerprint extraction, similarity, and CLI
├── dataset/           # Example paired probes
├── output/comput_W/   # Included example fingerprints
├── scripts/           # Batch processing
├── tools/             # Model merging and pruning utilities
├── examples/          # Usage examples
├── tests/             # Algorithm tests
├── docs/              # GitHub Pages source and paper media
├── CITATION.cff       # GitHub citation metadata
└── pyproject.toml     # Package configuration
```

Project-page maintenance and local preview: [`docs/README.md`](docs/README.md).

<a id="citation"></a>

## 📝 Citation

```bibtex
@misc{zhang2026attndiffattentionbaseddifferentialfingerprinting,
  title={AttnDiff: Attention-based Differential Fingerprinting for Large Language Models},
  author={Haobo Zhang and Zhenhua Xu and Junxian Li and Shangfeng Sheng and Dezhang Kong and Meng Han},
  year={2026},
  eprint={2604.05502},
  archivePrefix={arXiv},
  primaryClass={cs.CR},
  url={https://arxiv.org/abs/2604.05502}
}
```

## License and Acknowledgments

Code is released under the [MIT License](LICENSE). Built with [PyTorch](https://pytorch.org/), [Hugging Face Transformers](https://huggingface.co/transformers), and [uv](https://github.com/astral-sh/uv).
