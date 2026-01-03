import os
import json
from typing import Optional


# ModelScope (Mota) model download interface (using modelscope's snapshot_download)
def download_from_mota(model_name: str, save_dir: str, **kwargs):
    """
    Download models from ModelScope using snapshot_download.
    Reference: https://www.modelscope.cn/docs/models/download
    """
    try:
        from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download
    except ImportError:
        raise ImportError(
            "Please install modelscope first: pip install modelscope")
    print(f"[ModelScope] Downloading model: {model_name} to {save_dir}")
    ms_snapshot_download(model_id=model_name, local_dir=save_dir, **kwargs)


def download_from_huggingface(model_name: str, save_dir: str, **kwargs):
    try:
        from huggingface_hub import snapshot_download as hf_snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is not installed. Please install it: pip install huggingface_hub"
        )
    # Map deprecated use_auth_token to token for newer huggingface_hub versions
    token = kwargs.pop("use_auth_token", None)
    if token is not None:
        kwargs["token"] = token
    print(f"[HuggingFace] Downloading model: {model_name} to {save_dir}")
    hf_snapshot_download(repo_id=model_name, local_dir=save_dir, **kwargs)


# Future extensible download interfaces
DOWNLOADERS = {
    'mota': download_from_mota,
    'huggingface': download_from_huggingface,
    # Reserved: 'other': download_from_other,
}


def download_model(source: str, model_name: str, save_dir: str, **kwargs):
    """
    General model download entry.
    source: download source (e.g. 'mota', 'huggingface')
    model_name: model name or repo_id
    save_dir: directory to save the model
    kwargs: extra arguments
    """
    if source not in DOWNLOADERS:
        raise ValueError(
            f"Unknown download source: {source}. Supported: {list(DOWNLOADERS.keys())}"
        )
    os.makedirs(save_dir, exist_ok=True)
    DOWNLOADERS[source](model_name, save_dir, **kwargs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Universal Model Downloader")
    parser.add_argument('--source',
                        type=str,
                        required=True,
                        choices=DOWNLOADERS.keys(),
                        help='Download source')
    parser.add_argument('--model',
                        type=str,
                        required=True,
                        help='Model name or repo_id')
    parser.add_argument('--save_dir',
                        type=str,
                        required=True,
                        help='Directory to save the model')
    parser.add_argument(
        '--extra',
        nargs='*',
        help=
        'Extra arguments in key=value format. For list values, use JSON format: key=["value1","value2"]'
    )
    args = parser.parse_args()

    extra_kwargs = {}
    if args.extra:
        for item in args.extra:
            if '=' in item:
                k, v = item.split('=', 1)
                # Try to parse as JSON if it looks like a list or dict
                if v.startswith('[') or v.startswith('{'):
                    try:
                        v = json.loads(v)
                    except json.JSONDecodeError:
                        pass
                extra_kwargs[k] = v

    download_model(args.source, args.model, args.save_dir, **extra_kwargs)
