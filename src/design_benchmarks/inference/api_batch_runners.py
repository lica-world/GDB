"""Provider-specific batch inference runners (OpenAI, Anthropic, Gemini).

All three runners share the same contract:

- ``run(requests)``        → submit + poll + return (blocking)
- ``submit(requests)``     → fire-and-forget, returns batch_id
- ``collect(batch_id, …)`` → poll + return results

Each uses the provider's native batch API for ~50% cost savings over
individual calls.  For synchronous concurrent inference at full price,
use ``BatchRunner`` from ``batch.py`` instead.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from ..models.base import ModelOutput
from .batch import BatchRequest, BatchResult
from .gcs import upload_file_public


def _require_gcs_bucket(runner_name: str, bucket: Optional[str]) -> str:
    """Ensure a GCS bucket name is configured for batch image uploads."""
    b = (bucket or "").strip()
    if not b:
        raise ValueError(
            f"{runner_name}: GCS bucket is required for batch jobs that upload images. "
            "Pass bucket=... to the runner constructor or set the environment variable "
            "DESIGN_BENCHMARKS_GCS_BUCKET."
        )
    return b


# ---------------------------------------------------------------------------
# Shared image URL helper
# ---------------------------------------------------------------------------


def _to_url(
    image: Union[str, Path],
    bucket: str,
    gcs_prefix: str,
    credentials_path: Optional[str],
) -> str:
    """Convert an image source to an HTTPS URL.

    - Already a URL → pass through
    - gs:// URI → convert to public URL
    - Local file → upload to GCS, return signed URL
    """
    if isinstance(image, str) and image.startswith(("http://", "https://")):
        return image

    if isinstance(image, str) and image.startswith("gs://"):
        parts = image.replace("gs://", "").split("/", 1)
        return f"https://storage.googleapis.com/{parts[0]}/{parts[1]}"

    path = Path(image) if isinstance(image, str) else image
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image}")

    blob_name = f"{gcs_prefix}/{path.name}"
    return upload_file_public(path, bucket, blob_name, credentials_path)


# ═══════════════════════════════════════════════════════════════════════════
# OpenAI Batch API
# ═══════════════════════════════════════════════════════════════════════════


class OpenAIBatchRunner:
    """Native OpenAI Batch API runner (~50% cheaper).

    Images are uploaded to GCS and referenced by public URL in the JSONL,
    keeping the payload small regardless of image count.

    Usage::

        runner = OpenAIBatchRunner(
            model_id="gpt-5",
            bucket="your-gcs-bucket",
        )
        results = runner.run(requests)
    """

    def __init__(
        self,
        model_id: str = "gpt-5",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        poll_interval: int = 30,
        bucket: Optional[str] = None,
        gcs_prefix: str = "lica-bench/batch",
        credentials_path: Optional[str] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ):
        self.model_id = model_id
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.poll_interval = poll_interval
        self.bucket = bucket or os.environ.get("DESIGN_BENCHMARKS_GCS_BUCKET")
        self.gcs_prefix = gcs_prefix
        self.credentials_path = credentials_path
        self.on_status = on_status or (lambda msg: print(msg))

    def _get_client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai is required for OpenAI batch inference. "
                "Install with: pip install openai"
            )
        return OpenAI(api_key=self.api_key)

    def submit(self, requests: List[BatchRequest]) -> str:
        """Upload images to GCS, build JSONL with URLs, submit batch."""
        _require_gcs_bucket("OpenAIBatchRunner", self.bucket)
        client = self._get_client()

        self.on_status(f"Building JSONL for {len(requests)} requests (images → GCS)")
        jsonl_lines = self._build_jsonl(requests)

        self.on_status("Uploading batch input file to OpenAI")
        jsonl_content = "\n".join(jsonl_lines)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            f.write(jsonl_content)
            tmp_path = f.name

        try:
            batch_file = client.files.create(
                file=open(tmp_path, "rb"),
                purpose="batch",
            )
        finally:
            os.unlink(tmp_path)

        self.on_status(f"Uploaded: {batch_file.id}")
        self.on_status(f"Submitting batch for {self.model_id}")

        batch = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": f"design-bench-{self.model_id}"},
        )

        self.on_status(f"Batch submitted: {batch.id}")
        return batch.id

    def collect(
        self, batch_id: str, custom_ids: Optional[List[str]] = None
    ) -> Dict[str, BatchResult]:
        """Poll for completion and return results."""
        client = self._get_client()

        terminal_statuses = {"completed", "failed", "expired", "cancelled"}

        batch = client.batches.retrieve(batch_id)
        self.on_status(f"Status: {batch.status}")

        while batch.status not in terminal_statuses:
            time.sleep(self.poll_interval)
            batch = client.batches.retrieve(batch_id)
            counts = batch.request_counts
            self.on_status(
                f"Status: {batch.status}  "
                f"(completed={counts.completed}/{counts.total}, "
                f"failed={counts.failed})"
            )

        if batch.status != "completed":
            self.on_status(f"Batch FAILED: {batch.status}")
            error_msg = f"Batch {batch.status}"
            if batch.errors and batch.errors.data:
                error_msg += ": " + batch.errors.data[0].message
            fallback = {}
            for cid in custom_ids or []:
                fallback[cid] = BatchResult(
                    custom_id=cid,
                    model_output=ModelOutput(text=""),
                    error=error_msg,
                )
            return fallback

        self.on_status("Batch completed — downloading results")
        result_map: Dict[str, BatchResult] = {}

        if batch.output_file_id:
            result_map.update(self._parse_output(client, batch.output_file_id))

        if batch.error_file_id:
            self.on_status("Downloading error file as well")
            result_map.update(self._parse_output(client, batch.error_file_id))

        if not batch.output_file_id and not batch.error_file_id:
            self.on_status(
                "WARNING: batch completed but neither output_file_id "
                "nor error_file_id is set"
            )

        if custom_ids:
            for cid in custom_ids:
                if cid not in result_map:
                    result_map[cid] = BatchResult(
                        custom_id=cid,
                        model_output=ModelOutput(text=""),
                        error="No output row for this request",
                    )

        self.on_status(f"Parsed {len(result_map)} results")
        return result_map

    def run(self, requests: List[BatchRequest]) -> Dict[str, BatchResult]:
        """Submit a batch job, poll until done, return results."""
        batch_id = self.submit(requests)
        custom_ids = [r.custom_id for r in requests]
        return self.collect(batch_id, custom_ids)

    def _build_jsonl(self, requests: List[BatchRequest]) -> List[str]:
        """Build JSONL lines, uploading local images to GCS as public URLs."""
        job_id = uuid.uuid4().hex[:12]
        job_prefix = f"{self.gcs_prefix}/{job_id}/images"

        jsonl_lines: List[str] = []
        for req in requests:
            content: List[dict] = []
            for img in req.model_input.images:
                url = _to_url(
                    img, self.bucket, job_prefix, self.credentials_path
                )
                content.append(
                    {"type": "image_url", "image_url": {"url": url}}
                )
            if req.model_input.text:
                content.append({"type": "text", "text": req.model_input.text})

            body: dict = {
                "model": self.model_id,
                "messages": [{"role": "user", "content": content}],
                "max_completion_tokens": self.max_tokens,
            }
            if self.temperature is not None:
                body["temperature"] = self.temperature

            line = {
                "custom_id": req.custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            jsonl_lines.append(json.dumps(line, separators=(",", ":")))
        return jsonl_lines

    @staticmethod
    def _parse_output(client: Any, output_file_id: str) -> Dict[str, BatchResult]:
        output_content = client.files.content(output_file_id)
        output_text = output_content.text

        result_map: Dict[str, BatchResult] = {}
        for line in output_text.strip().split("\n"):
            if not line.strip():
                continue
            item = json.loads(line)
            custom_id = item["custom_id"]

            top_error = item.get("error")
            body = item.get("response", {}).get("body", {})
            body_error = body.get("error")

            error = top_error or body_error
            if error:
                msg = error.get("message", json.dumps(error)) if isinstance(error, dict) else str(error)
                result_map[custom_id] = BatchResult(
                    custom_id=custom_id,
                    model_output=ModelOutput(text=""),
                    error=msg,
                )
            else:
                choices = body.get("choices", [])
                text = ""
                if choices:
                    text = choices[0].get("message", {}).get("content", "")
                result_map[custom_id] = BatchResult(
                    custom_id=custom_id,
                    model_output=ModelOutput(text=text),
                )
        return result_map


# ═══════════════════════════════════════════════════════════════════════════
# Anthropic Message Batches API
# ═══════════════════════════════════════════════════════════════════════════


class AnthropicBatchRunner:
    """Native Anthropic Message Batches API runner (~50% cheaper).

    Images are uploaded to GCS and referenced by signed URL.

    Usage::

        runner = AnthropicBatchRunner(model_id="claude-opus-4-6")
        results = runner.run(requests)
    """

    def __init__(
        self,
        model_id: str = "claude-opus-4-6",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        poll_interval: int = 30,
        bucket: Optional[str] = None,
        gcs_prefix: str = "lica-bench/batch",
        credentials_path: Optional[str] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ):
        self.model_id = model_id
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.poll_interval = poll_interval
        self.bucket = bucket or os.environ.get("DESIGN_BENCHMARKS_GCS_BUCKET")
        self.gcs_prefix = gcs_prefix
        self.credentials_path = credentials_path
        self.on_status = on_status or (lambda msg: print(msg))

    def _get_client(self) -> Any:
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic is required for Claude batch inference. "
                "Install with: pip install anthropic"
            )
        return anthropic.Anthropic(api_key=self.api_key)

    def submit(self, requests: List[BatchRequest]) -> str:
        """Submit the batch and return the batch ID immediately."""

        client = self._get_client()

        _require_gcs_bucket("AnthropicBatchRunner", self.bucket)
        self.on_status(f"Building {len(requests)} requests for {self.model_id}")
        batch_requests = self._build_requests(requests)

        self.on_status(f"Submitting batch of {len(batch_requests)} requests")
        batch = client.messages.batches.create(requests=batch_requests)

        self.on_status(f"Batch submitted: {batch.id}")
        return batch.id

    def collect(
        self, batch_id: str, custom_ids: Optional[List[str]] = None
    ) -> Dict[str, BatchResult]:
        """Poll for completion and return results."""
        client = self._get_client()

        batch = client.messages.batches.retrieve(batch_id)
        self.on_status(f"Status: {batch.processing_status}")

        while batch.processing_status != "ended":
            time.sleep(self.poll_interval)
            batch = client.messages.batches.retrieve(batch_id)
            counts = batch.request_counts
            self.on_status(
                f"Status: {batch.processing_status}  "
                f"(succeeded={counts.succeeded}, "
                f"errored={counts.errored}, "
                f"processing={counts.processing})"
            )

        self.on_status("Batch ended — streaming results")
        result_map = self._parse_results(client, batch_id)

        if custom_ids:
            for cid in custom_ids:
                if cid not in result_map:
                    result_map[cid] = BatchResult(
                        custom_id=cid,
                        model_output=ModelOutput(text=""),
                        error="No result returned for this request",
                    )

        self.on_status(f"Parsed {len(result_map)} results")
        return result_map

    def run(self, requests: List[BatchRequest]) -> Dict[str, BatchResult]:
        """Submit a message batch, poll until done, return results."""
        batch_id = self.submit(requests)
        custom_ids = [r.custom_id for r in requests]
        return self.collect(batch_id, custom_ids)

    def _build_requests(self, requests: List[BatchRequest]) -> list:
        try:
            from anthropic.types.message_create_params import (
                MessageCreateParamsNonStreaming,
            )
            from anthropic.types.messages.batch_create_params import Request
        except ImportError:
            raise ImportError(
                'anthropic is required for Claude batch inference. '
                'Install with: pip install -e ".[anthropic]"'
            )

        job_id = uuid.uuid4().hex[:12]
        job_prefix = f"{self.gcs_prefix}/{job_id}/images"

        batch_requests = []
        for req in requests:
            content: List[dict] = []
            for img in req.model_input.images:
                url = _to_url(img, self.bucket, job_prefix, self.credentials_path)
                content.append(
                    {
                        "type": "image",
                        "source": {"type": "url", "url": url},
                    }
                )
            if req.model_input.text:
                content.append({"type": "text", "text": req.model_input.text})

            batch_requests.append(
                Request(
                    custom_id=req.custom_id,
                    params=MessageCreateParamsNonStreaming(
                        model=self.model_id,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        messages=[{"role": "user", "content": content}],
                    ),
                )
            )
        return batch_requests

    @staticmethod
    def _parse_results(client: Any, batch_id: str) -> Dict[str, BatchResult]:
        result_map: Dict[str, BatchResult] = {}
        for entry in client.messages.batches.results(batch_id):
            custom_id = entry.custom_id
            result = entry.result

            if result.type == "succeeded":
                text = ""
                for block in result.message.content:
                    if hasattr(block, "text"):
                        text += block.text
                result_map[custom_id] = BatchResult(
                    custom_id=custom_id,
                    model_output=ModelOutput(text=text),
                )
            elif result.type == "errored":
                error_msg = "Unknown error"
                if result.error:
                    error_msg = getattr(result.error, "message", str(result.error))
                result_map[custom_id] = BatchResult(
                    custom_id=custom_id,
                    model_output=ModelOutput(text=""),
                    error=error_msg,
                )
            else:
                result_map[custom_id] = BatchResult(
                    custom_id=custom_id,
                    model_output=ModelOutput(text=""),
                    error=f"Request {result.type}",
                )
        return result_map


# ═══════════════════════════════════════════════════════════════════════════
# Gemini / Vertex AI Batch Prediction
# ═══════════════════════════════════════════════════════════════════════════


class GeminiBatchRunner:
    """Native Gemini batch prediction via Vertex AI (~50% cheaper).

    Uploads local images to GCS, builds a JSONL input file, submits a
    Vertex AI batch prediction job, polls until completion, then parses
    the output JSONL.

    Usage::

        runner = GeminiBatchRunner(
            model_id="gemini-2.0-flash",
            credentials_path="/path/to/service-account.json",
            bucket="your-gcs-bucket",
        )
        results = runner.run(requests)
    """

    def __init__(
        self,
        model_id: str = "gemini-2.0-flash",
        credentials_path: Optional[str] = None,
        bucket: Optional[str] = None,
        gcs_prefix: str = "lica-bench/batch",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        poll_interval: int = 30,
        on_status: Optional[Callable[[str], None]] = None,
    ):
        self.model_id = model_id
        self.credentials_path = credentials_path
        self.bucket = bucket or os.environ.get("DESIGN_BENCHMARKS_GCS_BUCKET")
        self.gcs_prefix = gcs_prefix
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.poll_interval = poll_interval
        self.on_status = on_status or (lambda msg: print(msg))

        self._project_id = ""
        if credentials_path:
            creds = json.loads(Path(credentials_path).read_text())
            self._project_id = creds.get("project_id", "")

    def _build_client(self) -> Any:
        try:
            from google import genai
            from google.genai.types import HttpOptions
        except ImportError:
            raise ImportError(
                'google-genai is required for Gemini batch inference. '
                'Install with: pip install -e ".[gemini]"'
            )

        if self.credentials_path:
            creds_data = json.loads(Path(self.credentials_path).read_text())
            if creds_data.get("type") == "service_account":
                from google.oauth2 import service_account

                scopes = ["https://www.googleapis.com/auth/cloud-platform"]
                sa_creds = service_account.Credentials.from_service_account_file(
                    self.credentials_path,
                    scopes=scopes,
                )
                return genai.Client(
                    vertexai=True,
                    project=self._project_id,
                    location="global",
                    credentials=sa_creds,
                    http_options=HttpOptions(api_version="v1"),
                )

        return genai.Client(http_options=HttpOptions(api_version="v1"))

    def submit(self, requests: List[BatchRequest]) -> str:
        """Upload images + JSONL, submit batch job, return job name immediately.

        Also stores metadata in ``self._last_submit_meta`` for
        same-process collect.
        """
        try:
            from google.genai.types import CreateBatchJobConfig
        except ImportError:
            raise ImportError(
                'google-genai is required for Gemini batch inference. '
                'Install with: pip install -e ".[gemini]"'
            )

        _require_gcs_bucket("GeminiBatchRunner", self.bucket)
        job_id = uuid.uuid4().hex[:12]
        job_prefix = f"{self.gcs_prefix}/{job_id}"

        self.on_status(
            f"Uploading {len(requests)} images to "
            f"gs://{self.bucket}/{job_prefix}/images/"
        )

        ordered_ids, jsonl_lines = self._build_jsonl(requests, job_prefix)

        jsonl_content = "\n".join(jsonl_lines)
        input_blob = f"{job_prefix}/input.jsonl"

        from .gcs import upload_bytes

        input_uri = upload_bytes(
            jsonl_content.encode(),
            self.bucket,
            input_blob,
            content_type="application/jsonl",
            credentials_path=self.credentials_path,
        )
        output_uri = f"gs://{self.bucket}/{job_prefix}/output/"
        self.on_status(f"Input JSONL: {input_uri}")

        self.on_status(f"Submitting batch job for {self.model_id}")
        client = self._build_client()

        job = client.batches.create(
            model=self.model_id,
            src=input_uri,
            config=CreateBatchJobConfig(dest=output_uri),
        )

        self.on_status(f"Job submitted: {job.name}")

        self._last_submit_meta = {
            "job_prefix": job_prefix,
            "ordered_ids": ordered_ids,
        }
        return job.name

    def collect(
        self,
        batch_id: str,
        custom_ids: Optional[List[str]] = None,
        job_prefix: Optional[str] = None,
    ) -> Dict[str, BatchResult]:
        """Poll for completion and return results."""
        try:
            from google.genai.types import JobState
        except ImportError:
            raise ImportError(
                'google-genai is required for Gemini batch inference. '
                'Install with: pip install -e ".[gemini]"'
            )

        meta = getattr(self, "_last_submit_meta", None) or {}
        if custom_ids is None:
            custom_ids = meta.get("ordered_ids", [])
        if job_prefix is None:
            job_prefix = meta.get("job_prefix")

        _require_gcs_bucket("GeminiBatchRunner", self.bucket)
        client = self._build_client()
        job = client.batches.get(name=batch_id)
        self.on_status(f"State: {job.state}")

        completed_states = {
            JobState.JOB_STATE_SUCCEEDED,
            JobState.JOB_STATE_FAILED,
            JobState.JOB_STATE_CANCELLED,
            JobState.JOB_STATE_PAUSED,
        }

        while job.state not in completed_states:
            time.sleep(self.poll_interval)
            job = client.batches.get(name=batch_id)
            self.on_status(f"State: {job.state}")

        if job.state != JobState.JOB_STATE_SUCCEEDED:
            self.on_status(f"Job FAILED: {job.state}")
            return {
                cid: BatchResult(
                    custom_id=cid,
                    model_output=ModelOutput(text=""),
                    error=f"Batch job {job.state}",
                )
                for cid in custom_ids
            }

        self.on_status("Job succeeded — downloading results")

        if job_prefix is None:
            self.on_status("WARNING: job_prefix unknown, cannot download output")
            return {
                cid: BatchResult(
                    custom_id=cid,
                    model_output=ModelOutput(text=""),
                    error="job_prefix not available for result download",
                )
                for cid in custom_ids
            }

        results = self._download_results(job_prefix, custom_ids)
        self.on_status(f"Parsed {len(results)} results")
        return results

    def run(self, requests: List[BatchRequest]) -> Dict[str, BatchResult]:
        """Submit batch job, poll until done, return results."""
        batch_id = self.submit(requests)
        custom_ids = [r.custom_id for r in requests]
        job_prefix = self._last_submit_meta["job_prefix"]
        return self.collect(batch_id, custom_ids, job_prefix)

    def _build_jsonl(
        self, requests: List[BatchRequest], job_prefix: str
    ) -> tuple:
        """Build JSONL lines and upload images. Returns (ordered_ids, lines)."""
        from .gcs import get_mime_type, upload_file

        ordered_ids: List[str] = []
        jsonl_lines: List[str] = []

        for req in requests:
            ordered_ids.append(req.custom_id)
            parts: List[dict] = []

            for img in req.model_input.images:
                img_str = str(img)
                if img_str.startswith("gs://"):
                    mime = get_mime_type(img_str)
                    parts.append(
                        {"fileData": {"fileUri": img_str, "mimeType": mime}}
                    )
                else:
                    img_path = Path(img_str)
                    if not img_path.exists():
                        raise FileNotFoundError(f"Image not found: {img_str}")
                    mime = get_mime_type(img_path)
                    blob_name = f"{job_prefix}/images/{img_path.name}"
                    gcs_uri = upload_file(
                        img_path, self.bucket, blob_name, self.credentials_path
                    )
                    parts.append(
                        {"fileData": {"fileUri": gcs_uri, "mimeType": mime}}
                    )

            if req.model_input.text:
                parts.append({"text": req.model_input.text})

            line = {
                "request": {
                    "contents": [{"role": "user", "parts": parts}],
                    "generationConfig": {
                        "temperature": self.temperature,
                        "maxOutputTokens": self.max_tokens,
                    },
                }
            }
            jsonl_lines.append(json.dumps(line, separators=(",", ":")))

        return ordered_ids, jsonl_lines

    def _download_results(
        self, job_prefix: str, ordered_ids: List[str]
    ) -> Dict[str, BatchResult]:
        from .gcs import download_text, list_blobs

        output_blob_prefix = f"{job_prefix}/output/"
        blob_names = list_blobs(
            self.bucket, output_blob_prefix, self.credentials_path
        )

        output_lines: List[dict] = []
        for blob_name in sorted(blob_names):
            if not blob_name.endswith(".jsonl"):
                continue
            text = download_text(self.bucket, blob_name, self.credentials_path)
            for line in text.strip().split("\n"):
                if line.strip():
                    output_lines.append(json.loads(line))

        results: Dict[str, BatchResult] = {}
        for i, custom_id in enumerate(ordered_ids):
            if i < len(output_lines):
                out = output_lines[i]
                status = out.get("status", "")
                response = out.get("response", {})
                candidates = response.get("candidates", [])

                if status and not candidates:
                    results[custom_id] = BatchResult(
                        custom_id=custom_id,
                        model_output=ModelOutput(text=""),
                        error=status,
                    )
                else:
                    text = ""
                    if candidates:
                        parts = (
                            candidates[0].get("content", {}).get("parts", [])
                        )
                        text = parts[0].get("text", "") if parts else ""
                    results[custom_id] = BatchResult(
                        custom_id=custom_id,
                        model_output=ModelOutput(text=text),
                    )
            else:
                results[custom_id] = BatchResult(
                    custom_id=custom_id,
                    model_output=ModelOutput(text=""),
                    error="No output row for this request",
                )

        return results
