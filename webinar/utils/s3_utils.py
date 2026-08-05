import os
from functools import lru_cache
from pathlib import Path
from typing import Optional
from PIL import ImageFile
from webinar.config import CONFIG

ImageFile.LOAD_TRUNCATED_IMAGES = True
import boto3

# Dataset paths in this repo predate the S3 move and are rooted at
# 's3_data/From-Algo/' (the GCS-era layout). In the bucket the same tree lives
# under S3_KEY_PREFIX, so that root is swapped when building the object key.
# The local cache layout is unchanged, so caches populated in the GCS era
# still hit.
_LEGACY_ROOT = "s3_data/From-Algo/"


def _download(cloud_file_path: str, local_file_path: Optional[str] = None) -> Path:
    cloud_file_path = str(cloud_file_path)
    # if local_file_path is not specified, cache under $HOME/Tensorleap_data_3/CACHE_DIR
    if local_file_path is None:
        persistent_dir = Path(os.getenv("HOME"))
        local_file_path = persistent_dir / "Tensorleap_data_3" / CONFIG['CACHE_DIR'] / cloud_file_path
    local_file_path = Path(local_file_path)

    # check if the file already exists at the specified local path
    if local_file_path.exists():
        return local_file_path

    local_file_path.parent.mkdir(parents=True, exist_ok=True)

    key = cloud_file_path
    if key.startswith(_LEGACY_ROOT):
        key = key[len(_LEGACY_ROOT):]
    key = CONFIG['S3_KEY_PREFIX'] + key

    _s3_client().download_file(CONFIG['S3_BUCKET'], key, str(local_file_path))
    return local_file_path


@lru_cache()
def _s3_client():
    # Default credential chain: env vars, shared config (AWS_PROFILE), or an
    # attached role. No key file required.
    return boto3.client("s3")
