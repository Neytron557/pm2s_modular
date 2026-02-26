"""
Dataset Download Scripts for Audio-to-Score

Downloads the following datasets:
1. PDMX - 250k+ MusicXML files from Zenodo
2. Lakh MIDI - 176k MIDI files from HuggingFace
3. ASAP - Aligned Scores and Performances from GitHub
4. ACPAS - Aligned Classical Piano Audio and Score from GitHub
5. URMP - University of Rochester Multi-modal Music Performance
6. MSMD - Multimodal Sheet Music Dataset from Zenodo
"""

import os
import subprocess
import argparse
import zipfile
import tarfile
from pathlib import Path
from typing import Optional
from tqdm import tqdm

try:
    import requests
except ImportError:
    requests = None

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None


# Dataset configurations
DATASETS = {
    "pdmx": {
        "name": "PDMX (Public Domain MusicXML)",
        "url": "https://zenodo.org/records/13763756/files/PDMX.zip",
        "size": "1.6 GB",
        "type": "zenodo",
        "description": "250k+ public domain MusicXML files for synthetic data generation",
    },
    "lakh_midi": {
        "name": "Lakh MIDI Dataset",
        "hf_repo": "mimbres/lakh_full",
        "size": "~6 GB",
        "type": "huggingface",
        "description": "176k MIDI files for pretraining",
    },
    "asap": {
        "name": "ASAP (Aligned Scores and Performances)",
        "git_url": "https://github.com/fosfrancesco/asap-dataset.git",
        "size": "~2 GB",
        "type": "git",
        "description": "222 scores aligned with 1068 performances (92+ hours)",
    },
    "acpas": {
        "name": "ACPAS (Aligned Classical Piano Audio and Score)",
        "git_url": "https://github.com/cheriell/ACPAS-dataset.git",
        "size": "~1 GB",
        "type": "git",
        "description": "Classical piano audio aligned with scores for PM2S training",
    },
    "urmp": {
        "name": "URMP (University of Rochester Multi-modal Music Performance)",
        "url": "https://labsites.rochester.edu/air/projects/URMP.html",
        "size": "~5 GB",
        "type": "manual",
        "description": "44 multi-instrument pieces with aligned MIDI and audio",
        "download_instructions": """
URMP requires manual download:
1. Visit: https://labsites.rochester.edu/air/projects/URMP.html
2. Fill out the Google Form to request access
3. Download and extract to: {output_dir}/urmp/
""",
    },
    "msmd": {
        "name": "MSMD (Multimodal Sheet Music Dataset)",
        "url": "https://zenodo.org/records/2597505/files/msmd_v1.0.zip",
        "size": "~3 GB",
        "type": "zenodo",
        "description": "497 pieces with aligned audio and sheet music images",
    },
}


def download_file(url: str, output_path: Path, chunk_size: int = 8192) -> bool:
    """Download a file with progress bar."""
    if requests is None:
        print("Error: requests library not installed. Run: pip install requests")
        return False

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with open(output_path, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True, desc=output_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def extract_archive(archive_path: Path, output_dir: Path) -> bool:
    """Extract zip or tar archive."""
    try:
        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(output_dir)
        elif archive_path.suffix in [".tar", ".gz", ".tgz"]:
            with tarfile.open(archive_path, "r:*") as tf:
                tf.extractall(output_dir)
        else:
            print(f"Unknown archive format: {archive_path}")
            return False
        return True
    except Exception as e:
        print(f"Error extracting {archive_path}: {e}")
        return False


def download_zenodo(dataset_key: str, output_dir: Path) -> bool:
    """Download dataset from Zenodo."""
    config = DATASETS[dataset_key]
    url = config["url"]

    # Create output directory
    dataset_dir = output_dir / dataset_key
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Download archive
    archive_name = url.split("/")[-1]
    archive_path = dataset_dir / archive_name

    if archive_path.exists():
        print(f"Archive already exists: {archive_path}")
    else:
        print(f"Downloading {config['name']} from Zenodo...")
        if not download_file(url, archive_path):
            return False

    # Extract
    print(f"Extracting {archive_name}...")
    if not extract_archive(archive_path, dataset_dir):
        return False

    print(f"Successfully downloaded {config['name']} to {dataset_dir}")
    return True


def download_huggingface(dataset_key: str, output_dir: Path) -> bool:
    """Download dataset from HuggingFace."""
    if snapshot_download is None:
        print("Error: huggingface_hub not installed. Run: pip install huggingface_hub")
        return False

    config = DATASETS[dataset_key]
    hf_repo = config["hf_repo"]

    dataset_dir = output_dir / dataset_key
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {config['name']} from HuggingFace...")
    try:
        snapshot_download(
            repo_id=hf_repo,
            repo_type="dataset",
            local_dir=str(dataset_dir),
            local_dir_use_symlinks=False,
        )
        print(f"Successfully downloaded {config['name']} to {dataset_dir}")
        return True
    except Exception as e:
        print(f"Error downloading from HuggingFace: {e}")
        return False


def download_git(dataset_key: str, output_dir: Path) -> bool:
    """Clone dataset from GitHub."""
    config = DATASETS[dataset_key]
    git_url = config["git_url"]

    dataset_dir = output_dir / dataset_key

    if dataset_dir.exists():
        print(f"Dataset directory already exists: {dataset_dir}")
        print("Pulling latest changes...")
        try:
            subprocess.run(["git", "-C", str(dataset_dir), "pull"], check=True)
            return True
        except subprocess.CalledProcessError:
            print("Git pull failed, directory may not be a git repo")
            return False

    print(f"Cloning {config['name']} from GitHub...")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", git_url, str(dataset_dir)],
            check=True,
        )
        print(f"Successfully cloned {config['name']} to {dataset_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")
        return False


def download_dataset(dataset_key: str, output_dir: Path) -> bool:
    """Download a single dataset."""
    if dataset_key not in DATASETS:
        print(f"Unknown dataset: {dataset_key}")
        print(f"Available datasets: {list(DATASETS.keys())}")
        return False

    config = DATASETS[dataset_key]
    dtype = config["type"]

    print(f"\n{'='*60}")
    print(f"Dataset: {config['name']}")
    print(f"Size: {config['size']}")
    print(f"Description: {config['description']}")
    print(f"{'='*60}\n")

    if dtype == "zenodo":
        return download_zenodo(dataset_key, output_dir)
    elif dtype == "huggingface":
        return download_huggingface(dataset_key, output_dir)
    elif dtype == "git":
        return download_git(dataset_key, output_dir)
    elif dtype == "manual":
        print(config["download_instructions"].format(output_dir=output_dir))
        return True
    else:
        print(f"Unknown download type: {dtype}")
        return False


def download_all(output_dir: Path, skip_manual: bool = True) -> dict:
    """Download all datasets."""
    results = {}

    for dataset_key in DATASETS:
        if skip_manual and DATASETS[dataset_key]["type"] == "manual":
            print(f"\nSkipping {dataset_key} (requires manual download)")
            results[dataset_key] = "skipped"
            continue

        success = download_dataset(dataset_key, output_dir)
        results[dataset_key] = "success" if success else "failed"

    return results


def list_datasets():
    """List all available datasets."""
    print("\nAvailable Datasets for Audio-to-Score:")
    print("=" * 70)

    for key, config in DATASETS.items():
        print(f"\n{key}:")
        print(f"  Name: {config['name']}")
        print(f"  Size: {config['size']}")
        print(f"  Type: {config['type']}")
        print(f"  Description: {config['description']}")


def check_existing(output_dir: Path) -> dict:
    """Check which datasets already exist."""
    status = {}
    for dataset_key in DATASETS:
        dataset_path = output_dir / dataset_key
        if dataset_path.exists() and any(dataset_path.iterdir()):
            status[dataset_key] = "exists"
        else:
            status[dataset_key] = "missing"
    return status


def main():
    default_output_dir = Path(__file__).resolve().parents[1] / "datasets"
    parser = argparse.ArgumentParser(
        description="Download datasets for Audio-to-Score training"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(default_output_dir),
        help="Output directory for downloaded datasets",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASETS.keys()) + ["all"],
        default="all",
        help="Which dataset to download (default: all)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check which datasets already exist",
    )
    parser.add_argument(
        "--include-manual",
        action="store_true",
        help="Include datasets that require manual download (shows instructions)",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.list:
        list_datasets()
        return

    if args.check:
        print(f"\nChecking datasets in: {output_dir}")
        status = check_existing(output_dir)
        for key, stat in status.items():
            icon = "✓" if stat == "exists" else "✗"
            print(f"  [{icon}] {key}: {stat}")
        return

    if args.dataset == "all":
        print(f"Downloading all datasets to: {output_dir}")
        results = download_all(output_dir, skip_manual=not args.include_manual)

        print("\n" + "=" * 60)
        print("Download Summary:")
        print("=" * 60)
        for key, result in results.items():
            icon = "✓" if result == "success" else ("⊘" if result == "skipped" else "✗")
            print(f"  [{icon}] {key}: {result}")
    else:
        download_dataset(args.dataset, output_dir)


if __name__ == "__main__":
    main()
