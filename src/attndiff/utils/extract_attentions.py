#!/usr/bin/env python

import json
import os
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_name: str, device: str):
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
                from auto_gptq import AutoGPTQForCausalLM
                from auto_gptq.modeling._base import BaseQuantizeConfig
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
                    "Detected a GPTQ model directory but could not find a quantization config"
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
            from peft import PeftModel
        except ImportError as e:
            raise ImportError(
                "Detected a PEFT adapter directory but peft is not installed. "
                "Please install it: pip install peft"
            ) from e

        with adapter_config_path.open("r", encoding="utf-8") as f:
            adapter_cfg = json.load(f)

        base_model_from_cfg = adapter_cfg.get("base_model_name_or_path")
        base_model_override = os.environ.get("PEFT_BASE_MODEL_PATH")
        base_model_name_or_path = base_model_override or base_model_from_cfg
        if not base_model_name_or_path:
            raise ValueError("PEFT adapter config does not specify base_model_name_or_path")

        base_model_candidate = Path(base_model_name_or_path)
        if not base_model_candidate.is_absolute():
            base_model_candidate = (model_path / base_model_candidate).resolve()

        if not base_model_candidate.exists():
            raise FileNotFoundError(
                f"Base model path for PEFT adapter does not exist: {base_model_candidate}"
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

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if device is None:
        device_map = "auto"
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
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )

    encoded = {k: v.to(device) for k, v in encoded.items()}

    cfg = getattr(model, "config", None)
    if cfg is not None and getattr(cfg, "model_type", None) == "qwen2":
        try:
            cfg._attn_implementation = "eager"
        except Exception:
            pass

    outputs = model(
        **encoded,
        output_attentions=True,
        return_dict=True,
    )

    attentions = outputs.attentions

    if attentions is None:
        raise RuntimeError(
            "Model did not return attentions. Ensure attention implementation is set to 'eager'."
        )

    heads_per_layer = [att.shape[1] for att in attentions]
    H_min = min(heads_per_layer)

    if len(set(heads_per_layer)) > 1:
        print(
            f"[AttnExtract] Detected varying num_heads per layer {heads_per_layer}; "
            f"cropping all layers to H_min={H_min}."
        )

    cropped_layers = [layer_attn[0, :H_min] for layer_attn in attentions]
    attn_stack = torch.stack(cropped_layers, dim=0)

    print(f"[AttnExtract] Attention matrix shape [L, H, N, N]: {attn_stack.shape}")

    input_ids = encoded["input_ids"][0].cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    attn_list = attn_stack.cpu().tolist()

    return tokens, attn_list


def extract_attentions_for_dataset(
    model_name: str,
    device: str,
    dataset_path: Path,
    original_out: Path,
    corrupted_out: Path,
):
    print(f"[AttnExtract] Using device: {device}")
    print(f"[AttnExtract] Loading model: {model_name}")
    tokenizer, model = load_model_and_tokenizer(model_name, device)

    print(f"[AttnExtract] Loading dataset from: {dataset_path}")
    with dataset_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    original_results = []
    corrupted_results = []

    for idx, item in enumerate(tqdm(dataset, desc="Processing prompts")):
        sample_id = item.get("id", idx)
        original_text = item["original"]
        corrupted_text = item["corrupted"]

        tokens_o, attn_o = extract_attention_for_prompt(model, tokenizer, original_text, device)
        original_results.append({
            "id": sample_id,
            "prompt": original_text,
            "tokens": tokens_o,
            "attention": attn_o,
        })

        tokens_c, attn_c = extract_attention_for_prompt(model, tokenizer, corrupted_text, device)
        corrupted_results.append({
            "id": sample_id,
            "prompt": corrupted_text,
            "tokens": tokens_c,
            "attention": attn_c,
        })

    print(f"[AttnExtract] Saving original attentions to: {original_out}")
    with original_out.open("w", encoding="utf-8") as f:
        json.dump(original_results, f, ensure_ascii=False)

    print(f"[AttnExtract] Saving corrupted attentions to: {corrupted_out}")
    with corrupted_out.open("w", encoding="utf-8") as f:
        json.dump(corrupted_results, f, ensure_ascii=False)

    print("[AttnExtract] Done.")


def process_dataset(
    data_path: Path,
    model_name: str,
    out_original: Path,
    out_corrupted: Path,
    device: str = None,
):
    """
    Wrapper function for backward compatibility with compute.py.
    Maps old parameter names to new function.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    extract_attentions_for_dataset(
        model_name=model_name,
        device=device,
        dataset_path=data_path,
        original_out=out_original,
        corrupted_out=out_corrupted,
    )
