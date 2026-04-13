"""GCS upload/download helpers for batch inference.

Uses ``google-cloud-storage`` for blob operations. Authentication reuses
the same service-account credentials as the Gemini model wrappers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Union

MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def get_mime_type(path: Union[str, Path]) -> str:
    """Infer MIME type from file extension."""
    ext = Path(path).suffix.lower()
    return MIME_TYPES.get(ext, "application/octet-stream")


def _get_client(credentials_path: Optional[str] = None) -> Any:
    try:
        from google.cloud import storage
    except ImportError:
        raise ImportError(
            "google-cloud-storage is required for GCS operations. "
            "Install with: pip install google-cloud-storage"
        )
    if credentials_path:
        return storage.Client.from_service_account_json(credentials_path)
    return storage.Client()


def upload_file(
    local_path: Union[str, Path],
    bucket_name: str,
    blob_name: str,
    credentials_path: Optional[str] = None,
) -> str:
    """Upload a local file to GCS. Returns ``gs://`` URI."""
    client = _get_client(credentials_path)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(local_path))
    return f"gs://{bucket_name}/{blob_name}"


def upload_file_public(
    local_path: Union[str, Path],
    bucket_name: str,
    blob_name: str,
    credentials_path: Optional[str] = None,
    signed_url_hours: int = 24,
) -> str:
    """Upload a local file to GCS and return an HTTPS URL.

    Returns a signed URL (valid for ``signed_url_hours``) so external
    APIs (OpenAI, Anthropic) can fetch it without the bucket being public.
    """
    import datetime

    client = _get_client(credentials_path)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    mime = get_mime_type(local_path)
    blob.upload_from_filename(str(local_path), content_type=mime)

    url = blob.generate_signed_url(
        version="v4",
        expiration=datetime.timedelta(hours=signed_url_hours),
        method="GET",
    )
    return url


def upload_bytes(
    data: bytes,
    bucket_name: str,
    blob_name: str,
    content_type: str = "application/octet-stream",
    credentials_path: Optional[str] = None,
) -> str:
    """Upload raw bytes to GCS. Returns ``gs://`` URI."""
    client = _get_client(credentials_path)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(data, content_type=content_type)
    return f"gs://{bucket_name}/{blob_name}"


def download_text(
    bucket_name: str,
    blob_name: str,
    credentials_path: Optional[str] = None,
) -> str:
    """Download a blob's contents as UTF-8 text."""
    client = _get_client(credentials_path)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.download_as_text()


def list_blobs(
    bucket_name: str,
    prefix: str,
    credentials_path: Optional[str] = None,
) -> List[str]:
    """List blob names under a GCS prefix."""
    client = _get_client(credentials_path)
    blobs = client.list_blobs(bucket_name, prefix=prefix)
    return [blob.name for blob in blobs]
