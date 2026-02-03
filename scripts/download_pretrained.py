"""Download pretrained models (HuBERT, RMVPE, etc.) for RVC."""

import logging
import os
from pathlib import Path
import urllib.request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_file(url: str, output_path: str, description: str = "") -> None:
    """Download a file from URL."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists():
        logger.info(f"{description} already exists: {output_path}")
        return
    
    logger.info(f"Downloading {description} from {url}...")
    logger.info(f"Output: {output_path}")
    
    try:
        urllib.request.urlretrieve(url, str(output_path))
        logger.info(f"✅ Downloaded {description}")
    except Exception as e:
        logger.error(f"❌ Failed to download {description}: {e}")
        raise


def download_pretrained_models() -> None:
    """Download all required pretrained models."""
    logger.info("Downloading pretrained models...")
    
    # Create pretrained directory structure
    pretrained_dir = Path("pretrained")
    pretrained_dir.mkdir(exist_ok=True)
    
    # HuBERT model
    # Note: Replace with actual URLs for your pretrained models
    hubert_dir = pretrained_dir / "hubert"
    hubert_dir.mkdir(exist_ok=True)
    
    # Example URLs (replace with actual model URLs)
    # download_file(
    #     url="https://example.com/hubert_base.pt",
    #     output_path=str(hubert_dir / "hubert_base.pt"),
    #     description="HuBERT base model"
    # )
    
    # RMVPE model
    rmvpe_dir = pretrained_dir / "rmvpe"
    rmvpe_dir.mkdir(exist_ok=True)
    
    # Example URLs (replace with actual model URLs)
    # download_file(
    #     url="https://example.com/rmvpe.pt",
    #     output_path=str(rmvpe_dir / "rmvpe.pt"),
    #     description="RMVPE model"
    # )
    
    logger.info("⚠️  Note: Replace placeholder URLs with actual model download links")
    logger.info("Pretrained models directory structure created at: pretrained/")


if __name__ == "__main__":
    download_pretrained_models()
