import os
from functools import lru_cache
from typing import Optional
from PIL import ImageFile
from webinar.config import CONFIG

ImageFile.LOAD_TRUNCATED_IMAGES = True
from google.cloud import storage
from google.cloud.storage import Bucket
import json
from pathlib import Path
from google.oauth2 import service_account


def _download(cloud_file_path: str, local_file_path: Optional[str] = None) -> str:
    # if local_file_path is not specified, save file in the '/nfs/Tensorleap_data/BUCKET_NAME' directory
    if local_file_path is None:
        persistent_dir = Path(os.getenv("HOME"))
        local_file_path = persistent_dir / "Tensorleap_data_3" / CONFIG['BUCKET_NAME'] / cloud_file_path

    # check if the file already exists at the specified local path
    if local_file_path.exists():
        # file exists, return the local file path
        return local_file_path

    # connect to the cloud storage bucket specified by BUCKET_NAME
    bucket = _connect_to_gcs_and_return_bucket(CONFIG['BUCKET_NAME'])

    # create the directory specified by dir_path if it does not exist
    dir_path = local_file_path.parent
    dir_path.mkdir(parents=True, exist_ok=True)

    # download the file from the cloud storage bucket to the specified local file path
    blob = bucket.blob(cloud_file_path)
    blob.download_to_filename(local_file_path)

    # return the local file path
    return local_file_path


@lru_cache()
def _connect_to_gcs_and_return_bucket(bucket_name: str) -> Bucket:
    auth_secret_string = os.environ['AUTH_SECRET']
    auth_secret = json.loads(auth_secret_string)
    # if auth_secret is a dictionary, create a Credentials object from it
    if type(auth_secret) is dict:
        credentials = service_account.Credentials.from_service_account_info(auth_secret)
    # if auth_secret is not a dictionary, assume it is a path and create a Credentials object from it
    else:
        credentials = service_account.Credentials.from_service_account_file(auth_secret)

    # get the project ID from the Credentials object
    project = credentials.project_id

    # create a Client object using the project ID and credentials
    gcs_client = storage.Client(project=project, credentials=credentials)

    # return the bucket with the specified name
    return gcs_client.bucket(bucket_name)
