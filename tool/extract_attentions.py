#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


def load_model_and_tokenizer(model_name: str, device: str):
    """Load tokenizer and model onto the specified device.

    By default, this uses HuggingFace AutoModel/AutoTokenizer.from_pretrained(model_name).
    """

    model_path = Path(model_name)

    if model_path.is_dir():
        gptq_config_candidates = [
            "quantize_config.json",
            "quant_config.json",
            "gptq_config.json",
        ]
        has_gptq_config = any((model_path / fname).exists() for fname in gptq_config_candidates)
        if has_gptq_config or "gptq" in model_name.lower():
            try:
                from auto_gptq import AutoGPTQForCausalLM  # type: ignore
                from auto_gptq.modeling._base import BaseQuantizeConfig  # type: ignore
            except ImportError as e:
                raise ImportError(
                    "Detected a GPTQ model directory but auto-gptq is not installed. Please install it: pip install auto-gptq"
                ) from e

            quantize_config_dict = None
            for fname in gptq_config_candidates:
                cfg_path = model_path / fname
                if cfg_path.exists():
                    with cfg_path.open("r", encoding="utf-8") as f:
                        quantize_config_dict = json.load(f)
                    break

            if quantize_config_dict is None:
                cfg_path = model_path / "config.json"
                if cfg_path.exists():
                    with cfg_path.open("r", encoding="utf-8") as f:
                        cfg_json = json.load(f)
                    quantize_config_dict = cfg_json.get("quantization_config")

            if not isinstance(quantize_config_dict, dict):
                raise FileNotFoundError(
                    "Detected a GPTQ model directory but could not find a quantization config in "
                    "quantize_config.json / quant_config.json / gptq_config.json / config.json::quantization_config"
                )

            field_names = {f.name for f in BaseQuantizeConfig.__dataclass_fields__.values()}
            synonyms = {"w_bit": "bits", "q_group_size": "group_size"}
            filtered_args = {}
            for k, v in quantize_config_dict.items():
                if k in synonyms and synonyms[k] in field_names:
                    filtered_args[synonyms[k]] = v
                elif k in field_names:
                    filtered_args[k] = v
            quantize_config = BaseQuantizeConfig(**filtered_args)

            model_basename = None
            for p in model_path.glob("*.safetensors"):
                name = p.name
                if "-00001-of-" in name:
                    model_basename = name.split("-00001-of-")[0]
                    break
            if model_basename is None and (model_path / "model.safetensors").exists():
                model_basename = "model"

            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoGPTQForCausalLM.from_quantized(
                model_name_or_path=model_name,
                device=device,
                trust_remote_code=True,
                use_safetensors=True,
                model_basename=model_basename,
                quantize_config=quantize_config,
            )
            if hasattr(model, "set_attn_implementation"):
                try:
                    model.set_attn_implementation("eager")
                except Exception:
                    pass
            if hasattr(model, "to"):
                try:
                    model.to(device)
                except Exception:
                    pass
            model.eval()
            return tokenizer, model

    adapter_config_path = model_path / "adapter_config.json" if model_path.is_dir() else None
    if adapter_config_path is not None and adapter_config_path.exists():
        try:
            from peft import PeftModel  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Detected a PEFT adapter directory (adapter_config.json exists) but peft is not installed. "
                "Please install it: pip install peft"
            ) from e

        with adapter_config_path.open("r", encoding="utf-8") as f:
            adapter_cfg = json.load(f)

        base_model_from_cfg = adapter_cfg.get("base_model_name_or_path")
        base_model_override = os.environ.get("PEFT_BASE_MODEL_PATH")
        base_model_name_or_path = base_model_override or base_model_from_cfg
        if not base_model_name_or_path:
            raise ValueError(
                "PEFT adapter config does not specify base_model_name_or_path, and PEFT_BASE_MODEL_PATH is not set."
            )

        base_model_candidate = Path(base_model_name_or_path)
        if not base_model_candidate.is_absolute():
            base_model_candidate = (model_path / base_model_candidate).resolve()

        if not base_model_candidate.exists():
            raise FileNotFoundError(
                "Base model path for PEFT adapter does not exist: "
                f"{base_model_candidate}. "
                "Set PEFT_BASE_MODEL_PATH to a valid local base model directory (e.g. /home/.../meta-llama/Llama-2-7B)."
            )

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            str(base_model_candidate),
            trust_remote_code=True,
            torch_dtype="auto",
        )
        model = PeftModel.from_pretrained(base_model, model_name)
        if hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()
        if hasattr(model, "set_attn_implementation"):
            try:
                model.set_attn_implementation("eager")
            except Exception:
                pass
        model.to(device)
        model.eval()
        return tokenizer, model

    # Standard HuggingFace model directory / model name
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # NOTE:
    # If we pass device_map="auto" and also move inputs to a single cuda:X,
    # HuggingFace/Accelerate may shard the model across different devices,
    # causing embedding to fail with "Expected all tensors to be on the same device".
    # When the caller provides an explicit device (e.g. cuda:6 / cpu), force the
    # entire model onto that device.
    if device is None:
        device_map = "auto"  # Automatically shard across multi-GPU or CPU/GPU
    else:
        device_map = {"": device}

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map=device_map,
        torch_dtype="auto",
    )
    if hasattr(model, "set_attn_implementation"):
        try:
            model.set_attn_implementation("eager")
        except Exception:
            pass
    model.eval()
    return tokenizer, model


@torch.no_grad()
def extract_attention_for_prompt(
    model,
    tokenizer,
    prompt: str,
    device: str,
):
    """Run a forward pass for a single prompt and return:

    - tokens: token list after tokenization
    - attention: 4D attention tensor [L, H, N, N] (converted to a plain Python list)
    """
    # Encode inputs
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )

    # Move to target device
    encoded = {k: v.to(device) for k, v in encoded.items()}

    # Forward pass with attentions enabled
    # Extra safeguard: for models like Qwen2, explicitly set config._attn_implementation
    # to eager to avoid cases where set_attn_implementation does not take effect and
    # SDPA is used, resulting in attentions=None.
    cfg = getattr(model, "config", None)
    if cfg is not None and getattr(cfg, "model_type", None) == "qwen2":
        try:
            setattr(cfg, "_attn_implementation", "eager")
        except Exception:
            pass

    outputs = model(
        **encoded,
        output_attentions=True,
        return_dict=True,
    )

    # outputs.attentions is a tuple of length L,
    # where each element has shape [batch_size, num_heads, seq_len, seq_len]
    attentions = outputs.attentions  # tuple of length L 或 None

    if attentions is None:
        raise RuntimeError(
            "Model did not return attentions (outputs.attentions is None). "
            "Ensure attention implementation is set to 'eager' and that the model "
            "supports output_attentions=True."
        )

    # After pruning, the number of heads may vary across layers (e.g., 32 vs 29).
    # torch.stack would fail due to mismatched dimensions, so we crop all layers to
    # the minimum number of heads, producing a consistent [L, H_min, N, N].
    heads_per_layer = [att.shape[1] for att in attentions]
    H_min = min(heads_per_layer)

    if len(set(heads_per_layer)) > 1:
        print(
            f"[AttnExtract] Detected varying num_heads per layer {heads_per_layer}; "
            f"cropping all layers to H_min={H_min}."
        )

    cropped_layers = [layer_attn[0, :H_min] for layer_attn in attentions]

    # Stack along a new dimension to form [L, H_min, N, N]
    attn_stack = torch.stack(cropped_layers, dim=0)

    print(f"[AttnExtract] Attention matrix shape [L, H, N, N]: {attn_stack.shape}")

    # Recover tokens (useful for downstream inspection)
    input_ids = encoded["input_ids"][0].cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Convert to a plain Python list for JSON serialization
    attn_list = attn_stack.cpu().tolist()

    return tokens, attn_list


def process_dataset(
    data_path: Path,
    model_name: str,
    out_original: Path | None,
    out_corrupted: Path | None,
    device: str | None = None,
):
    if out_original is None and out_corrupted is None:
        raise ValueError(
            "At least one of out_original or out_corrupted must be provided to process_dataset."
        )

    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[AttnExtract] Using device: {device}")
    print(f"[AttnExtract] Loading model: {model_name}")
    tokenizer, model = load_model_and_tokenizer(model_name, device)

    print(f"[AttnExtract] Loading dataset from: {data_path}")
    with data_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    original_results = [] if out_original is not None else None
    corrupted_results = [] if out_corrupted is not None else None

    for idx, item in enumerate(tqdm(dataset, desc="Processing prompts")):
        sample_id = item.get("id", idx)
        original_text = item["original"]
        corrupted_text = item["corrupted"]

        if out_original is not None:
            tokens_o, attn_o = extract_attention_for_prompt(
                model, tokenizer, original_text, device
            )
            original_results.append(
                {
                    "id": sample_id,
                    "prompt": original_text,
                    "tokens": tokens_o,
                    "attention": attn_o,  # Shape: [L][H][N][N]
                }
            )

        if out_corrupted is not None:
            tokens_c, attn_c = extract_attention_for_prompt(
                model, tokenizer, corrupted_text, device
            )
            corrupted_results.append(
                {
                    "id": sample_id,
                    "prompt": corrupted_text,
                    "tokens": tokens_c,
                    "attention": attn_c,  # Shape: [L][H][N][N]
                }
            )

    # Save results
    if out_original is not None:
        print(f"[AttnExtract] Saving original attentions to: {out_original}")
        with out_original.open("w", encoding="utf-8") as f:
            json.dump(original_results, f, ensure_ascii=False)

    if out_corrupted is not None:
        print(f"[AttnExtract] Saving corrupted attentions to: {out_corrupted}")
        with out_corrupted.open("w", encoding="utf-8") as f:
            json.dump(corrupted_results, f, ensure_ascii=False)

    print("[AttnExtract] Done.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract attention matrices for original & corrupted prompts."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="dataset.json",
        help="Path to the input dataset JSON file",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="HuggingFace model name, e.g. gpt2, bert-base-uncased, bert-base-chinese",
    )
    parser.add_argument(
        "--out_original",
        type=str,
        default=None,
        help="Output JSON path for original prompts; if not set, derived from model_name",
    )
    parser.add_argument(
        "--out_corrupted",
        type=str,
        default=None,
        help="Output JSON path for corrupted prompts; if not set, derived from model_name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional: device string such as cuda or cpu; defaults to auto-detect",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data_path)
    model_name = args.model_name

    # Derive default output paths from model_name (can be overridden by explicit args)
    model_base = Path(model_name).name if model_name is not None else "model"

    if args.out_original is None:
        out_original = Path("output/attention") / f"{model_base}_att_origin.json"
    else:
        out_original = Path(args.out_original)

    if args.out_corrupted is None:
        out_corrupted = Path("output/attention") / f"{model_base}_att_perturb.json"
    else:
        out_corrupted = Path(args.out_corrupted)

    # Ensure output directories exist
    out_original.parent.mkdir(parents=True, exist_ok=True)
    out_corrupted.parent.mkdir(parents=True, exist_ok=True)

    process_dataset(
        data_path=data_path,
        model_name=model_name,
        out_original=out_original,
        out_corrupted=out_corrupted,
        device=args.device,
    )


if __name__ == "__main__":
    main()