#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import json
from pathlib import Path
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


def load_model_and_tokenizer(model_name: str, device: str):
    """加载 tokenizer 和模型到指定设备。

    默认使用 HuggingFace 的 AutoModel/AutoTokenizer.from_pretrained(model_name)。
    但如果 model_name 指向的是一个二进制 checkpoint 文件（.bin），且该文件是
    LLM-Pruner 保存的 {'model': ..., 'tokenizer': ...} 字典，则直接使用 torch.load
    反序列化该 checkpoint，从而避免与 HuggingFace 原始架构之间的尺寸不匹配问题。
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

    # 分支 1：model_name 指向 LLM-Pruner 生成的 checkpoint 文件
    if model_path.is_file() and model_path.suffix == ".bin":
        # 确保 LLMPruner 包可被导入（供 torch.load 反序列化使用）
        llmpruner_root = Path(
            "/home/kdz/data/xzh/zhb/EverTracer-main/EverTracer-main/Experiments/model-pruning/LLM-Pruner-main"
        )
        if llmpruner_root.exists():
            root_str = str(llmpruner_root.resolve())
            if root_str not in sys.path:
                sys.path.insert(0, root_str)
            try:
                __import__("LLMPruner")  # noqa: F401 - ensure importable
            except ImportError:
                print(
                    f"[AttnExtract] Warning: failed to import LLMPruner from {root_str}. "
                    "If torch.load fails with ModuleNotFoundError('LLMPruner'), "
                    "please check the path."
                )
        else:
            print(
                f"[AttnExtract] Warning: LLM-Pruner root not found at {llmpruner_root}; "
                "if torch.load fails with ModuleNotFoundError('LLMPruner'), please adjust the path in extract_attentions.py."
            )

        print(f"[AttnExtract] Loading LLM-Pruner checkpoint: {model_path}")
        ckpt = torch.load(str(model_path), map_location=device, weights_only=False)
        if not isinstance(ckpt, dict) or "model" not in ckpt or "tokenizer" not in ckpt:
            raise ValueError(
                "LLM-Pruner checkpoint must be a dict containing 'model' and 'tokenizer' keys. "
                f"Found keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}"
            )

        model = ckpt["model"]
        tokenizer = ckpt["tokenizer"]

        if hasattr(model, "to"):
            model.to(device)

        # For models like Qwen2 loaded from LLM-Pruner checkpoints, ensure
        # we use the eager attention backend so that output_attentions=True
        # actually produces attention tensors instead of None (sdpa backend
        # does not support output_attentions for these models).
        if hasattr(model, "set_attn_implementation"):
            try:
                model.set_attn_implementation("eager")
            except Exception:
                pass

        model.eval()

        return tokenizer, model

    # 分支 2：常规 HuggingFace 模型目录 / 名称
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype="auto")
    if hasattr(model, "set_attn_implementation"):
        try:
            model.set_attn_implementation("eager")
        except Exception:
            pass
    model.to(device)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def extract_attention_for_prompt(
    model,
    tokenizer,
    prompt: str,
    device: str,
):
    """
    对单条 prompt 前向计算，并返回：
    - tokens: tokenizer 后的 token 列表
    - attention: 4D 注意力矩阵 [L, H, N, N]（转换成普通 Python list）
    """
    # 编码输入
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )

    # 移到对应设备
    encoded = {k: v.to(device) for k, v in encoded.items()}

    # 前向计算，开启 attentions 输出
    # 额外保险：对 Qwen2 等模型，直接把 config._attn_implementation 设置为 eager，
    # 防止某些场景下 set_attn_implementation 未生效仍走 sdpa 导致 attentions=None。
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

    # outputs.attentions 是长度为 L 的元组，
    # 每个元素形状为 [batch_size, num_heads, seq_len, seq_len]
    attentions = outputs.attentions  # tuple of length L 或 None

    if attentions is None:
        raise RuntimeError(
            "Model did not return attentions (outputs.attentions is None). "
            "Ensure attention implementation is set to 'eager' and that the model "
            "supports output_attentions=True."
        )

    # 剪枝后，不同层的 head 数可能不同（例如某层 32 heads，某层 29 heads），
    # 直接 torch.stack 会因为维度不一致报错。这里统一裁剪到所有层的最小 head 数，
    # 这样得到形状一致的 [L, H_min, N, N]。
    heads_per_layer = [att.shape[1] for att in attentions]
    H_min = min(heads_per_layer)

    if len(set(heads_per_layer)) > 1:
        print(
            f"[AttnExtract] Detected varying num_heads per layer {heads_per_layer}; "
            f"cropping all layers to H_min={H_min}."
        )

    cropped_layers = [layer_attn[0, :H_min] for layer_attn in attentions]

    # 沿新维度堆叠成 [L, H_min, N, N]
    attn_stack = torch.stack(cropped_layers, dim=0)

    print(f"[AttnExtract] Attention matrix shape [L, H, N, N]: {attn_stack.shape}")

    # 还原 tokens（便于后处理查看）
    input_ids = encoded["input_ids"][0].cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # 转成纯 Python list，方便 json 序列化
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

    # 自动检测设备
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
                    "attention": attn_o,  # 形状 [L][H][N][N]
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
                    "attention": attn_c,  # 形状 [L][H][N][N]
                }
            )

    # 保存结果
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
        help="输入数据集 JSON 文件路径",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="HuggingFace 模型名，例如 gpt2、bert-base-uncased、bert-base-chinese 等",
    )
    parser.add_argument(
        "--out_original",
        type=str,
        default=None,
        help="保存原始输入注意力矩阵的 JSON 文件路径；不指定时根据 model_name 自动生成",
    )
    parser.add_argument(
        "--out_corrupted",
        type=str,
        default=None,
        help="保存扰动输入注意力矩阵的 JSON 文件路径；不指定时根据 model_name 自动生成",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="可选：指定设备，如 cuda 或 cpu；默认自动检测",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data_path)
    model_name = args.model_name

    # 根据 model_name 自动生成默认输出路径（可被显式参数覆盖）
    model_base = Path(model_name).name if model_name is not None else "model"

    if args.out_original is None:
        out_original = Path("output/attention") / f"{model_base}_att_origin.json"
    else:
        out_original = Path(args.out_original)

    if args.out_corrupted is None:
        out_corrupted = Path("output/attention") / f"{model_base}_att_perturb.json"
    else:
        out_corrupted = Path(args.out_corrupted)

    # 确保输出目录存在
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

"""
python extract_attentions.py \
  --data_path dataset/dataset.json \
  --model_name /home/kdz/data/OpenSourceModels/meta-llama/Llama-3-8B
"""