# Model directory

[← README](../README.md) · [Search the interactive model directory](https://zhb0119.github.io/AttnDiff/#models)

This directory preserves the repository's model taxonomy and checkpoint links. Different experiments use different subsets; the 36-related / 33-unrelated discriminability pool is a specific benchmark, not the size of this entire directory.

## Model Taxonomy


The following table categorizes all models used in experiments by their manipulation type:

| Category | Type | Base Model | Derivative Models |
|----------|------|------------|-------------------|
| **Fine-tuning** | Instruction | Llama-2-7B | Llama-2-finance-7b, Vicuna-1.5-7b, WizardMath-7b, Chinese-LLaMA-2-7b, CodeLLaMA-7b, Llemma-7b |
| **Preference Opt.** | PPO/DPO | Llama-2-7B | llama-2-7b-ppo-v0.1-reward, llama-2-7b-ppo-lora-v0.1, tulu-2-dpo-7b, llama2-7b-dpo |
| **Model Merging** | Weight | Shisa-gamma-7b-v1, WizardMath-7b-1.1, Abel-7b-002 | Evollm-jp-7b |
| | Distribution | Llama-2-7B, OpenLLaMA-2-7b, mpt-7b | Fusellm-7b |
| **Model Pruning** | Structured | Llama-2-7B | Sheared-llama-1.3b, Sheared-llama-1.3b-pruned, Sheared-llama-1.3b-sharegpt, Sheared-llama-2.7b, Sheared-llama-2.7b-pruned |
| | Unstructured | Llama-2-7B | Sparse-llama-2-7b, Wanda-llama-2-7b, GBLM-llama-2-7b |
| **Ablation** | Related | Llama-2-7B | CodeLlama-7b, Llama-2-finance-7B, Vicuna-7B-v1.5, Chinese-LLama-2-7B, WizardMath-7B-V1.0, llemma_7b, Sheared-LLaMA-1.3B, Sheared-LLaMA-1.3B-Pruned, Sheared-LLaMA-1.3B-ShareGPT, Sheared-LLaMA-2.7B, Sheared-LLaMA-2.7B-Pruned, Sheared-LLaMA-2.7B-ShareGPT, Sparse-llama-2-7b, Wanda-llama-2-7b, GBLM-llama-2-7b |
| | Unrelated | Llama-2-7B | Llama3-8B, mpt-7b, Qwen2.5-1.5B, Qwen2.5-3B, Qwen2.5-7B, Qwen2.5-14B, Qwen2.5-Math-7B, gemma-2-2b, Gemma-7B-it, Yi-6B |
| **Pilot** | Discovery | Llama-2-7B | Llama-2-7B, CodeLlama-7b-hf, WizardMath-7B-V1.0, llemma_7b, Qwen2.5-7B |
| **Model Distillation** | Reasoning | Llama-3.1-8B, Qwen2.5-7B, Qwen2.5-14B | Llama-3.1-8B-Instruct-Open-R1-Distill, Qwen2.5-7B-Open-R1-Distill, DeepSeek-R1-Distill-Qwen-14B |
| | Logit-based | Llama-2-7B | llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta2 |
| **MoE** | Mixtral | Mixtral-8x7B | Instruct_Mixtral-8x7B-v0.1_Dolly15K, Nous-Hermes-2-Mixtral-8x7B-DPO, openbuddy-mixtral-8x7b-v15.4 |
| **Cross-Family** | Qwen2.5 | Qwen2.5-7B | Qwen2.5-Coder-1.5B, Qwen2.5-Math-1.5B, Qwen2.5-1.5B-Instruct |
| | | Qwen2.5-14B | Qwen2.5-14B-Instruct, oxy-1-small, Qwen2.5-14B-Gutenberg-Instruct-Slerpeno |
| | Gemma-2 | gemma-2-2b | gemma-2-2b-neogenes-ita, gemma-2-baku-2b, gemma2-2b-merged |
| | Mistral | Mistral-7B-v0.3 | AQUA-7B, spellcheck-mistral-7b, Mistral-7B-Instruct-demi-merge-v0.3-7B |
| **Quantization** | GPTQ | Qwen2.5-7B | Qwen2.5-7B-Instruct-GPTQ-Int8, Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 |
| | | Llama-2-7B | TheBloke/Llama-2-7B-Chat-GPTQ |
| | | Llama-3.1-8B | iqbalamo93/Meta-Llama-3.1-8B-Instruct-GPTQ-Q_8, DaraV/LLaMA-3.1-8B-Instruct-INT4-GPTQ |
| | | Mistral-7B-v0.3 | RedHatAI/Mistral-7B-Instruct-v0.3-GPTQ-4bit |

## Model Repository Links

The following table provides Hugging Face repository links for key models used in experiments:

| Category | Model Name | Repository URL |
|----------|------------|----------------|
| **Fine-tuning** | Llama-2-7b-ppo-lora | [renyiyu/llama-2-7b-ppo-lora-v0.1](https://huggingface.co/renyiyu/llama-2-7b-ppo-lora-v0.1) |
| | Tulu-2-dpo-7b | [allenai/tulu-2-dpo-7b](https://huggingface.co/allenai/tulu-2-dpo-7b) |
| | Llama2-7b-dpo | [mncai/llama2-7b-dpo-v1](https://huggingface.co/mncai/llama2-7b-dpo-v1) |
| **Model Merging** | Evollm-jp-7b | [SakanaAI/EvoLLM-JP-v1-7B](https://huggingface.co/SakanaAI/EvoLLM-JP-v1-7B) |
| | Fusellm-7b | [Wanfq/FuseLLM-7B](https://huggingface.co/Wanfq/FuseLLM-7B) |
| **Model Pruning** | Sheared-llama-1.3b | [princeton-nlp/Sheared-LLaMA-1.3B](https://huggingface.co/princeton-nlp/Sheared-LLaMA-1.3B) |
| | Sheared-llama-2.7b | [princeton-nlp/Sheared-LLaMA-2.7B](https://huggingface.co/princeton-nlp/Sheared-LLaMA-2.7B) |
| **Distillation** | Llama-3.1-8B-Open-R1-Distill | [asas-ai/Llama-3.1-8B-Instruct-Open-R1-Distill](https://huggingface.co/asas-ai/Llama-3.1-8B-Instruct-Open-R1-Distill) |
| | Qwen2.5-7B-Open-R1-Distill | [erickrus/Qwen2.5-7B-Open-R1-Distill](https://huggingface.co/erickrus/Qwen2.5-7B-Open-R1-Distill) |
| | DeepSeek-R1-Distill-Qwen-14B | [deepseek-ai/DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B) |
| | Llama-2-7b-logit-watermark | [cygu/llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta2](https://huggingface.co/cygu/llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta2) |
| **MoE** | Instruct_Mixtral-8x7B-Dolly15K | [Brillibits/Instruct_Mixtral-8x7B-v0.1_Dolly15K](https://huggingface.co/Brillibits/Instruct_Mixtral-8x7B-v0.1_Dolly15K) |
| | Nous-Hermes-2-Mixtral-8x7B-DPO | [NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO](https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO) |
| | Openbuddy-mixtral-8x7b-v15.4 | [openbuddy/openbuddy-mixtral-8x7b-v15.4](https://huggingface.co/openbuddy/openbuddy-mixtral-8x7b-v15.4) |
| **Cross-Family** | Qwen2.5-Coder-1.5B | [Qwen/Qwen2.5-Coder-1.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B) |
| | Qwen2.5-Math-1.5B | [Qwen/Qwen2.5-Math-1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B) |
| | Qwen2.5-1.5B-Instruct | [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |
| | Qwen2.5-14B-Instruct | [Qwen/Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) |
| | Oxy-1-small | [oxyapi/oxy-1-small](https://huggingface.co/oxyapi/oxy-1-small) |
| | Qwen2.5-14B-Gutenberg-Instruct | [v000000/Qwen2.5-14B-Gutenberg-Instruct-Slerpeno](https://huggingface.co/v000000/Qwen2.5-14B-Gutenberg-Instruct-Slerpeno) |
| | Gemma-2-2b-neogenesis-ita | [anakin87/gemma-2-2b-neogenesis-ita](https://huggingface.co/anakin87/gemma-2-2b-neogenesis-ita) |
| | Gemma-2-baku-2b | [rinna/gemma-2-baku-2b](https://huggingface.co/rinna/gemma-2-baku-2b) |
| | Gemma2-2b-merged | [vonjack/gemma2-2b-merged](https://huggingface.co/vonjack/gemma2-2b-merged) |
| | AQUA-7B | [KurmaAI/AQUA-7B](https://huggingface.co/KurmaAI/AQUA-7B) |
| | Spellcheck-mistral-7b | [openfoodfacts/spellcheck-mistral-7b](https://huggingface.co/openfoodfacts/spellcheck-mistral-7b) |
| | Mistral-7B-Instruct-demi-merge | [grimjim/Mistral-7B-Instruct-demi-merge-v0.3-7B](https://huggingface.co/grimjim/Mistral-7B-Instruct-demi-merge-v0.3-7B) |
| **Quantization** | Qwen2.5-7B-Instruct-GPTQ-Int8 | [Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8) |
| | Qwen2.5-7B-Instruct-GPTQ-Int4 | [Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4) |
| | Llama-2-7B-Chat-GPTQ | [TheBloke/Llama-2-7B-Chat-GPTQ](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GPTQ) |
| | Meta-Llama-3.1-8B-GPTQ-Q_8 | [iqbalamo93/Meta-Llama-3.1-8B-Instruct-GPTQ-Q_8](https://huggingface.co/iqbalamo93/Meta-Llama-3.1-8B-Instruct-GPTQ-Q_8) |
| | LLaMA-3.1-8B-INT4-GPTQ | [DaraV/LLaMA-3.1-8B-Instruct-INT4-GPTQ](https://huggingface.co/DaraV/LLaMA-3.1-8B-Instruct-INT4-GPTQ) |
| | Mistral-7B-v0.3-GPTQ-4bit | [RedHatAI/Mistral-7B-Instruct-v0.3-GPTQ-4bit](https://huggingface.co/RedHatAI/Mistral-7B-Instruct-v0.3-GPTQ-4bit) |
