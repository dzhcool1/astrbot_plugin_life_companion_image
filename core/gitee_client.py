from __future__ import annotations

import asyncio
import base64
import binascii
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx


class GiteeAPIError(RuntimeError):
    """A user-safe error returned by the Gitee image API."""


class GiteeAIClient:
    """Small async client for Gitee AI Images and async image edits."""

    _EDIT_TASK_TYPES = {"id", "style", "subject", "background", "element"}

    def __init__(
        self,
        config: dict[str, Any],
        data_dir: Path,
        *,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.config = config
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "images"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self._client = http_client
        self._owns_client = http_client is None
        self._client_lock = asyncio.Lock()
        self._key_lock = asyncio.Lock()
        self._key_index = 0

    async def close(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    @staticmethod
    def _parse_keys(value: Any) -> list[str]:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    async def _next_key(self) -> str:
        async with self._key_lock:
            keys = self._parse_keys(self.config.get("api_keys", []))
            if not keys:
                raise GiteeAPIError("未配置 Gitee AI API Key")
            self._key_index %= len(keys)
            key = keys[self._key_index]
            self._key_index = (self._key_index + 1) % len(keys)
            return key

    def _base_url(self) -> str:
        value = str(
            self.config.get("api_base_url", "https://ai.gitee.com/v1")
            or "https://ai.gitee.com/v1"
        ).strip()
        value = value.rstrip("/")
        lower = value.lower()
        for suffix in (
            "/v1/images/generations",
            "/images/generations",
            "/v1/images/edits",
            "/images/edits",
        ):
            if lower.endswith(suffix):
                value = value[: -len(suffix)].rstrip("/")
                lower = value.lower()
                break
        try:
            parts = urlsplit(value)
            path = parts.path.rstrip("/").lower()
            if path == "/v1" or path.endswith("/v1") or "/v1/" in path:
                return value
            if parts.scheme and parts.netloc:
                return urlunsplit(
                    (parts.scheme, parts.netloc, parts.path.rstrip("/") + "/v1", "", "")
                ).rstrip("/")
        except ValueError:
            pass
        return f"{value}/v1"

    def _timeout(self) -> float:
        try:
            value = float(self.config.get("timeout", 600))
        except (TypeError, ValueError):
            value = 600.0
        return max(10.0, min(value, 3600.0))

    def _max_image_bytes(self) -> int:
        try:
            value = int(self.config.get("max_image_bytes", 20 * 1024 * 1024))
        except (TypeError, ValueError):
            value = 20 * 1024 * 1024
        return max(256 * 1024, min(value, 200 * 1024 * 1024))

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            async with self._client_lock:
                if self._client is None:
                    self._client = httpx.AsyncClient(
                        timeout=httpx.Timeout(self._timeout()),
                        follow_redirects=True,
                    )
        return self._client

    @staticmethod
    def _error_message(payload: Any, fallback: str) -> str:
        if isinstance(payload, dict):
            for key in ("message", "error", "detail"):
                value = payload.get(key)
                if isinstance(value, dict):
                    value = value.get("message") or value.get("detail")
                if value:
                    return " ".join(str(value).split())[:240]
        if isinstance(payload, str) and payload.strip():
            return " ".join(payload.split())[:240]
        return fallback

    async def _request_json(
        self,
        method: str,
        url: str,
        api_key: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        client = await self._get_client()
        headers = dict(kwargs.pop("headers", {}) or {})
        headers.setdefault("Authorization", f"Bearer {api_key}")
        headers.setdefault("Accept", "application/json")
        try:
            response = await client.request(method, url, headers=headers, **kwargs)
        except httpx.HTTPError as exc:
            raise GiteeAPIError("Gitee AI 网络请求失败，请检查网络或 API 地址") from exc

        try:
            payload = response.json()
        except ValueError:
            payload = {}
        if not 200 <= response.status_code < 300:
            message = self._error_message(
                payload,
                f"HTTP {response.status_code}",
            )
            raise GiteeAPIError(f"Gitee AI 请求失败：{message}")
        if not isinstance(payload, dict):
            raise GiteeAPIError("Gitee AI 返回了无法识别的 JSON")
        return payload

    async def generate(self, prompt: str, *, size: str | None = None) -> Path:
        prompt = str(prompt or "").strip()
        if not prompt:
            raise GiteeAPIError("图片提示词不能为空")

        api_key = await self._next_key()
        body: dict[str, Any] = {
            "model": str(self.config.get("model", "z-image-turbo") or "z-image-turbo"),
            "prompt": prompt,
            "size": str(size or self.config.get("default_size", "1024x1024")),
        }
        try:
            steps = int(self.config.get("num_inference_steps", 9))
        except (TypeError, ValueError):
            steps = 9
        if steps > 0:
            body["num_inference_steps"] = steps
        negative_prompt = str(self.config.get("negative_prompt", "") or "").strip()
        if negative_prompt:
            body["negative_prompt"] = negative_prompt

        payload = await self._request_json(
            "POST",
            f"{self._base_url()}/images/generations",
            api_key,
            json=body,
        )
        entry = self._first_image_entry(payload)
        return await self._materialize_entry(entry)

    async def edit(
        self,
        prompt: str,
        images: list[bytes],
        *,
        task_types: list[str] | None = None,
    ) -> Path:
        prompt = str(prompt or "").strip() or "自然真实地优化这张图片"
        if not images:
            raise GiteeAPIError("改图至少需要一张图片")

        api_key = await self._next_key()
        raw_task_types = task_types or self.config.get("task_types", ["id"])
        if isinstance(raw_task_types, str):
            raw_task_types = [raw_task_types]
        selected_types = [
            str(item)
            for item in (raw_task_types or [])
            if str(item) in self._EDIT_TASK_TYPES
        ]
        selected_types = selected_types or ["id"]
        data = [
            ("prompt", prompt),
            (
                "model",
                str(
                    self.config.get("edit_model", "Qwen-Image-Edit-2511")
                    or "Qwen-Image-Edit-2511"
                ),
            ),
            (
                "num_inference_steps",
                str(self.config.get("edit_num_inference_steps", 4)),
            ),
            ("guidance_scale", str(self.config.get("guidance_scale", 1.0))),
            *[("task_types", item) for item in selected_types],
        ]
        files = [
            ("image", (f"image_{index}.jpg", image, self._mime_type(image)))
            for index, image in enumerate(images)
            if image
        ]
        if not files:
            raise GiteeAPIError("改图图片内容为空")

        payload = await self._request_json(
            "POST",
            f"{self._base_url()}/async/images/edits",
            api_key,
            data=data,
            files=files,
        )
        task_id = payload.get("task_id") or payload.get("id")
        if not task_id:
            raise GiteeAPIError("Gitee AI 未返回改图任务编号")
        return await self._poll_edit(str(task_id), api_key)

    async def _poll_edit(self, task_id: str, api_key: str) -> Path:
        try:
            interval = max(0.0, float(self.config.get("poll_interval", 5)))
        except (TypeError, ValueError):
            interval = 5.0
        try:
            timeout = max(1.0, min(float(self.config.get("poll_timeout", 300)), 3600.0))
        except (TypeError, ValueError):
            timeout = 300.0

        deadline = time.monotonic() + timeout
        while True:
            payload = await self._request_json(
                "GET",
                f"{self._base_url()}/task/{task_id}",
                api_key,
            )
            status = str(payload.get("status", "")).strip().lower()
            if status in {"success", "succeeded", "completed", "complete"}:
                entry = self._first_image_entry(payload)
                return await self._materialize_entry(entry)
            if status in {"failed", "failure", "cancelled", "canceled", "error"}:
                message = self._error_message(payload, status or "unknown")
                raise GiteeAPIError(f"Gitee AI 改图任务失败：{message}")

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise GiteeAPIError(f"Gitee AI 改图任务超时（超过 {int(timeout)} 秒）")
            await asyncio.sleep(min(interval, remaining))

    @classmethod
    def _first_image_entry(cls, payload: dict[str, Any]) -> dict[str, Any]:
        data = payload.get("data")
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return data[0]
        output = payload.get("output")
        if isinstance(output, dict):
            return output
        if isinstance(output, str):
            return {"url": output}
        return payload

    async def _materialize_entry(self, entry: dict[str, Any]) -> Path:
        for key in ("b64_json", "base64", "image_base64"):
            value = entry.get(key)
            if value:
                return self._save_bytes(self._decode_base64(str(value)))

        for key in ("url", "file_url", "image_url"):
            value = str(entry.get(key, "") or "").strip()
            if value:
                return await self._download(value)
        raise GiteeAPIError("Gitee AI 返回成功，但没有图片 URL 或 Base64 数据")

    def _decode_base64(self, value: str) -> bytes:
        if "," in value and value.lower().startswith("data:"):
            value = value.split(",", 1)[1]
        try:
            data = base64.b64decode(value, validate=False)
        except (ValueError, binascii.Error) as exc:
            raise GiteeAPIError("Gitee AI 返回的 Base64 图片无效") from exc
        if not data:
            raise GiteeAPIError("Gitee AI 返回了空图片")
        self._check_size(data)
        return data

    async def _download(self, url: str) -> Path:
        if url.startswith("data:"):
            return self._save_bytes(self._decode_base64(url))
        client = await self._get_client()
        try:
            response = await client.get(url)
        except httpx.HTTPError as exc:
            raise GiteeAPIError("Gitee AI 图片下载失败，请检查网络") from exc
        if response.status_code != 200:
            raise GiteeAPIError(f"Gitee AI 图片下载失败：HTTP {response.status_code}")
        self._check_size(response.content)
        return self._save_bytes(response.content, response.headers.get("content-type"))

    def _check_size(self, data: bytes) -> None:
        if len(data) > self._max_image_bytes():
            raise GiteeAPIError("返回图片超过大小限制")

    def _save_bytes(self, data: bytes, content_type: str | None = None) -> Path:
        self._check_size(data)
        extension = self._extension(data, content_type)
        path = self.image_dir / f"image_{uuid.uuid4().hex}.{extension}"
        path.write_bytes(data)
        return path

    @staticmethod
    def _mime_type(data: bytes) -> str:
        if data.startswith(b"\x89PNG"):
            return "image/png"
        if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
            return "image/webp"
        return "image/jpeg"

    @staticmethod
    def _extension(data: bytes, content_type: str | None = None) -> str:
        if data.startswith(b"\x89PNG"):
            return "png"
        if data.startswith(b"\xff\xd8\xff"):
            return "jpg"
        if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
            return "webp"
        content_type = str(content_type or "").lower()
        if "png" in content_type:
            return "png"
        if "webp" in content_type:
            return "webp"
        return "jpg"
