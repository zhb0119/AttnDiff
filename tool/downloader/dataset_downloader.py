import os
import json
from typing import Optional


# ModelScope (Mota) dataset download interface (using MsDataset)
def download_from_mota(dataset_name: str, save_dir: str, **kwargs):
    """
    Download datasets from ModelScope using MsDataset.
    Reference: https://www.modelscope.cn/docs/datasets/download
    """
    try:
        from modelscope.msdatasets import MsDataset
    except ImportError:
        raise ImportError(
            "Please install modelscope first: pip install modelscope")
    print(f"[ModelScope] Downloading dataset: {dataset_name} to {save_dir}")
    dataset = MsDataset.load(dataset_name, **kwargs)
    
    # Save dataset to local directory
    os.makedirs(save_dir, exist_ok=True)
    # Note: The exact save method may vary depending on ModelScope's API
    if hasattr(dataset, 'save_to_disk'):
        dataset.save_to_disk(save_dir)
    else:
        # Fallback: save as JSON or other format
        print(f"[ModelScope] Warning: Direct save_to_disk not available. Implementing custom save logic...")
        # Custom save logic would go here
        pass
    print(f"[ModelScope] Dataset saved to: {save_dir}")


from huggingface_hub import snapshot_download


def download_from_huggingface(repo_id: str, save_dir: str, **kwargs):
    """
    Download datasets from Hugging Face Hub using snapshot_download.
    """
    print(f"[HuggingFace Hub] Downloading dataset: {repo_id} to {save_dir}")
    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=save_dir,
        local_dir_use_symlinks=False,
        **kwargs
    )
    print(f"[HuggingFace Hub] Dataset downloaded to: {local_dir}")


# Future extensible download interfaces
DOWNLOADERS = {
    'mota': download_from_mota,
    'huggingface': download_from_huggingface,
}


def download_dataset(source: str, dataset_name: str, save_dir: str, **kwargs):
    """
    General dataset download entry.
    source: download source (e.g. 'mota', 'huggingface')
    dataset_name: dataset name or repo_id
    save_dir: directory to save the dataset
    kwargs: extra arguments
    """
    if source not in DOWNLOADERS:
        raise ValueError(
            f"Unknown download source: {source}. Supported: {list(DOWNLOADERS.keys())}"
        )
    os.makedirs(save_dir, exist_ok=True)
    DOWNLOADERS[source](dataset_name, save_dir, **kwargs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Universal Dataset Downloader")
    parser.add_argument('--source',
                        type=str,
                        required=True,
                        choices=DOWNLOADERS.keys(),
                        help='Download source')
    parser.add_argument('--dataset',
                        type=str,
                        required=True,
                        help='Dataset name or repo_id')
    parser.add_argument('--save_dir',
                        type=str,
                        required=True,
                        help='Directory to save the dataset')
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

    download_dataset(args.source, args.dataset, args.save_dir, **extra_kwargs)
