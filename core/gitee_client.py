from __future__ import annotations

import asyncio
import base64
import binascii
import re
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

    def _is_provider_config(self) -> bool:
        return isinstance(self.config.get("features"), dict) and isinstance(
            self.config.get("providers"), list
        )

    def _feature_config(self, operation: str) -> dict[str, Any]:
        features = self.config.get("features")
        if not isinstance(features, dict):
            return {}
        value = features.get(operation)
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _chain_provider_id(item: Any) -> tuple[str, str] | None:
        if isinstance(item, str):
            provider_id = item.strip()
            return (provider_id, "") if provider_id else None
        if not isinstance(item, dict):
            return None
        nested = item.get("provider")
        source = nested if isinstance(nested, dict) else item
        provider_id = str(
            source.get("provider_id")
            or source.get("id")
            or source.get("provider")
            or source.get("backend")
            or ""
        ).strip()
        if not provider_id:
            return None
        output = str(
            item.get("output")
            or source.get("output")
            or item.get("default_output")
            or source.get("default_output")
            or ""
        ).strip()
        return provider_id, output

    @staticmethod
    def _provider_template(provider: dict[str, Any]) -> str:
        for key in ("__template_key", "template_key", "type", "provider_type"):
            value = str(provider.get(key) or "").strip().lower()
            if value:
                if value == "gitee":
                    return "gitee_images"
                return value
        if "default_resolution" in provider and "api_url" in provider:
            return "gemini_native"
        if "num_inference_steps" in provider and "base_url" in provider:
            return "gitee_images"
        if "base_url" in provider:
            return "openai_images"
        return ""

    def _provider_candidates(self, operation: str) -> list[dict[str, Any]]:
        if not self._is_provider_config():
            return [
                {
                    "id": "legacy",
                    "template": "legacy",
                    "config": self.config,
                    "output": "",
                }
            ]

        providers = {
            str(item.get("id") or "").strip(): item
            for item in self.config.get("providers", [])
            if isinstance(item, dict) and str(item.get("id") or "").strip()
        }
        feature = self._feature_config(operation)
        raw_chain = feature.get("chain")
        chain = raw_chain if isinstance(raw_chain, list) else []
        candidates: list[dict[str, Any]] = []
        for item in chain:
            parsed = self._chain_provider_id(item)
            if not parsed:
                continue
            provider_id, output = parsed
            provider = providers.get(provider_id)
            if provider is None:
                candidates.append(
                    {
                        "id": provider_id,
                        "template": "missing",
                        "config": {},
                        "output": output,
                    }
                )
                continue
            candidates.append(
                {
                    "id": provider_id,
                    "template": self._provider_template(provider),
                    "config": provider,
                    "output": output,
                }
            )
        if not candidates:
            raise GiteeAPIError(
                f"未配置 {operation} 服务商链路，请在 features.{operation}.chain 中添加 provider"
            )
        return candidates

    def _provider_keys(self, provider: dict[str, Any]) -> list[str]:
        return self._parse_keys(provider.get("api_keys") or provider.get("api_key"))

    async def _next_provider_key(self, provider: dict[str, Any]) -> str:
        keys = self._provider_keys(provider)
        if not keys:
            raise GiteeAPIError("未配置图片服务商 API Key")
        async with self._key_lock:
            self._key_index %= len(keys)
            key = keys[self._key_index]
            self._key_index = (self._key_index + 1) % len(keys)
            return key

    async def _next_key(self) -> str:
        async with self._key_lock:
            keys = self._parse_keys(self.config.get("api_keys", []))
            if not keys:
                raise GiteeAPIError("未配置 Gitee AI API Key")
            self._key_index %= len(keys)
            key = keys[self._key_index]
            self._key_index = (self._key_index + 1) % len(keys)
            return key

    @staticmethod
    def _normalize_openai_base_url(raw_value: Any) -> str:
        value = str(raw_value or "https://ai.gitee.com/v1").strip().rstrip("/")
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

    def _base_url(self) -> str:
        return self._normalize_openai_base_url(
            self.config.get("api_base_url", "https://ai.gitee.com/v1")
        )

    @staticmethod
    def _provider_timeout(provider: dict[str, Any], default: float = 600.0) -> float:
        try:
            value = float(provider.get("timeout", default) or default)
        except (TypeError, ValueError):
            value = default
        return max(10.0, min(value, 3600.0))

    def _timeout(self) -> float:
        try:
            value = float(self.config.get("timeout", 600))
        except (TypeError, ValueError):
            value = 600.0
        return max(10.0, min(value, 3600.0))

    def _max_image_bytes(self) -> int:
        value_source: Any = self.config.get("max_image_bytes", 50 * 1024 * 1024)
        network = self.config.get("network")
        if isinstance(network, dict):
            value_source = network.get("max_image_bytes", value_source)
        try:
            value = int(value_source)
        except (TypeError, ValueError):
            value = 50 * 1024 * 1024
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
        request_timeout = kwargs.pop("request_timeout", None)
        headers.setdefault("Authorization", f"Bearer {api_key}")
        headers.setdefault("Accept", "application/json")
        try:
            response = await client.request(
                method,
                url,
                headers=headers,
                timeout=request_timeout,
                **kwargs,
            )
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

    def _operation_output(self, operation: str, size: str | None) -> str:
        if size:
            return str(size).strip()
        return str(self._feature_config(operation).get("default_output") or "").strip()

    @staticmethod
    def _gemini_output_spec(size: str | None, default_resolution: Any) -> tuple[str, str]:
        value = str(size or "").strip().lower().replace("×", "x")
        sizes = {
            "256x256": ("1:1", "1K"),
            "512x512": ("1:1", "1K"),
            "1024x1024": ("1:1", "1K"),
            "2048x2048": ("1:1", "2K"),
            "4096x4096": ("1:1", "4K"),
            "1024x576": ("16:9", "1K"),
            "2048x1152": ("16:9", "2K"),
            "576x1024": ("9:16", "1K"),
            "1152x2048": ("9:16", "2K"),
            "1152x896": ("4:3", "1K"),
            "2048x1536": ("4:3", "2K"),
            "768x1024": ("3:4", "1K"),
            "1536x2048": ("3:4", "2K"),
            "2048x1360": ("3:2", "2K"),
            "1360x2048": ("2:3", "2K"),
        }
        aspect_ratio = sizes.get(value, ("", ""))[0]
        resolution = sizes.get(value, ("", ""))[1]
        for token in re.split(r"[ ,/]+", value):
            if re.fullmatch(r"(?:1|2|4)k", token):
                resolution = token.upper()
            elif re.fullmatch(r"\d+:\d+", token):
                aspect_ratio = token
        resolution = resolution or str(default_resolution or "4K").strip() or "4K"
        return aspect_ratio, resolution

    @staticmethod
    def _gemini_models_base_url(raw_value: Any) -> str:
        value = str(raw_value or "").strip().rstrip("/")
        lower = value.lower()
        for suffix in ("/v1beta/models", "/v1beta", "/v1"):
            if lower.endswith(suffix):
                value = value[: -len(suffix)].rstrip("/")
                break
        return f"{value}/v1beta/models"

    @staticmethod
    def _extract_gemini_images(payload: dict[str, Any]) -> list[bytes]:
        images: list[bytes] = []
        candidates = payload.get("candidates")
        if not isinstance(candidates, list):
            return images
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content")
            if not isinstance(content, dict):
                continue
            parts = content.get("parts")
            if not isinstance(parts, list):
                continue
            for part in parts:
                if not isinstance(part, dict):
                    continue
                inline = part.get("inlineData") or part.get("inline_data")
                if not isinstance(inline, dict):
                    continue
                value = inline.get("data")
                if not isinstance(value, str) or not value.strip():
                    continue
                try:
                    images.append(base64.b64decode(value, validate=False))
                except (ValueError, binascii.Error):
                    continue
        return [image for image in images if image]

    async def _gemini_request(
        self,
        prompt: str,
        provider: dict[str, Any],
        *,
        images: list[bytes] | None = None,
        size: str | None = None,
    ) -> Path:
        api_key = await self._next_provider_key(provider)
        model = str(provider.get("model") or "gemini-3-pro-image-preview").strip()
        url = f"{self._gemini_models_base_url(provider.get('api_url'))}/{model}:generateContent"
        aspect_ratio, resolution = self._gemini_output_spec(
            size, provider.get("default_resolution", "4K")
        )
        instruction = (
            f"Generate a high quality {resolution} resolution image. "
            f"Follow this instruction: {prompt}. Output the image directly."
        )
        if images:
            instruction = (
                f"Re-imagine the attached image based on this instruction: {prompt}. "
                f"Generate a high quality {resolution} resolution image. "
                "Output the transformed image directly."
            )
        parts: list[dict[str, Any]] = [{"text": instruction}]
        for image in images or []:
            parts.append(
                {
                    "inlineData": {
                        "mimeType": self._mime_type(image),
                        "data": base64.b64encode(image).decode("ascii"),
                    }
                }
            )
        image_config: dict[str, str] = {"imageSize": resolution}
        if aspect_ratio:
            image_config["aspectRatio"] = aspect_ratio
        body = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "maxOutputTokens": 8192,
                "responseModalities": ["TEXT", "IMAGE"],
                "imageConfig": image_config,
            },
        }
        client = await self._get_client()
        retries = max(0, min(int(provider.get("max_retries", 2) or 0), 10))
        timeout = self._provider_timeout(provider)
        last_error: GiteeAPIError | None = None
        for attempt in range(retries + 1):
            try:
                response = await client.post(
                    url,
                    headers={
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                        "x-goog-api-key": api_key,
                    },
                    json=body,
                    timeout=timeout,
                )
                try:
                    payload = response.json()
                except ValueError:
                    payload = {}
                if 200 <= response.status_code < 300 and isinstance(payload, dict):
                    images_found = self._extract_gemini_images(payload)
                    if images_found:
                        return self._save_bytes(images_found[-1])
                    message = self._error_message(payload, "未返回图片")
                    raise GiteeAPIError(f"Gemini 未返回图片：{message}")
                message = self._error_message(payload, f"HTTP {response.status_code}")
                error = GiteeAPIError(f"Gemini 请求失败：{message}")
                retryable = response.status_code in {408, 409, 425, 429} or response.status_code >= 500
                if not retryable or attempt >= retries:
                    raise error
                last_error = error
            except httpx.HTTPError as exc:
                error = GiteeAPIError("Gemini 网络请求失败，请检查网络或 API 地址")
                if attempt >= retries:
                    raise error from exc
                last_error = error
            if last_error is not None:
                await asyncio.sleep(min(2**attempt, 8))
        raise last_error or GiteeAPIError("Gemini 请求失败")

    async def _generate_provider(
        self, prompt: str, size: str | None, candidate: dict[str, Any]
    ) -> Path:
        template = candidate["template"]
        provider = candidate["config"]
        if template == "gemini_native":
            return await self._gemini_request(prompt, provider, size=size)
        if template in {"legacy", "gitee_images", "openai_images", "gemini_openai_images"}:
            api_key = await self._next_provider_key(provider)
            base_url = (
                self._base_url()
                if template == "legacy"
                else self._normalize_openai_base_url(provider.get("base_url"))
            )
            default_size = provider.get("default_size", "1024x1024")
            body: dict[str, Any] = {
                "model": str(provider.get("model", "z-image-turbo") or "z-image-turbo"),
                "prompt": prompt,
                "size": str(size or default_size),
            }
            steps = provider.get("num_inference_steps", 9)
            try:
                steps = int(steps)
            except (TypeError, ValueError):
                steps = 9
            if steps > 0:
                body["num_inference_steps"] = steps
            negative_prompt = str(provider.get("negative_prompt", "") or "").strip()
            if negative_prompt:
                body["negative_prompt"] = negative_prompt
            extra_body = provider.get("extra_body")
            if isinstance(extra_body, dict):
                body.update(extra_body)
            payload = await self._request_json(
                "POST",
                f"{base_url}/images/generations",
                api_key,
                request_timeout=self._provider_timeout(provider),
                json=body,
            )
            return await self._materialize_entry(self._first_image_entry(payload))
        raise GiteeAPIError(f"图片服务商 {candidate['id']} 暂不支持文生图")

    async def generate(self, prompt: str, *, size: str | None = None) -> Path:
        prompt = str(prompt or "").strip()
        if not prompt:
            raise GiteeAPIError("图片提示词不能为空")
        last_error: Exception | None = None
        for candidate in self._provider_candidates("draw"):
            try:
                requested_size = candidate["output"] or self._operation_output("draw", size)
                return await self._generate_provider(prompt, requested_size, candidate)
            except Exception as exc:
                last_error = exc
        raise GiteeAPIError(f"图片生成失败：{last_error}") from last_error

    async def _edit_provider(
        self,
        prompt: str,
        images: list[bytes],
        task_types: list[str] | None,
        size: str | None,
        candidate: dict[str, Any],
    ) -> Path:
        template = candidate["template"]
        provider = candidate["config"]
        if template == "gemini_native":
            return await self._gemini_request(prompt, provider, images=images, size=size)
        if template == "gitee_async":
            api_key = await self._next_provider_key(provider)
            raw_task_types = task_types or provider.get("task_types") or ["id"]
            selected_types = [
                str(item) for item in (raw_task_types if isinstance(raw_task_types, list) else [raw_task_types])
                if str(item) in self._EDIT_TASK_TYPES
            ] or ["id"]
            data = [
                ("prompt", prompt),
                ("model", str(provider.get("model") or "Qwen-Image-Edit-2511")),
                ("num_inference_steps", str(provider.get("num_inference_steps", 4))),
                ("guidance_scale", str(provider.get("guidance_scale", 1.0))),
                *[("task_types", item) for item in selected_types],
            ]
            files = [
                ("image", (f"image_{index}.jpg", image, self._mime_type(image)))
                for index, image in enumerate(images)
                if image
            ]
            if not files:
                raise GiteeAPIError("改图图片内容为空")
            base_url = self._normalize_openai_base_url(provider.get("base_url"))
            payload = await self._request_json(
                "POST",
                f"{base_url}/async/images/edits",
                api_key,
                request_timeout=self._provider_timeout(provider),
                data=data,
                files=files,
            )
            task_id = payload.get("task_id") or payload.get("id")
            if not task_id:
                raise GiteeAPIError("Gitee AI 未返回改图任务编号")
            return await self._poll_provider_edit(str(task_id), api_key, provider)
        if template in {"openai_images", "gemini_openai_images"}:
            api_key = await self._next_provider_key(provider)
            files = [
                ("image", (f"image_{index}.jpg", image, self._mime_type(image)))
                for index, image in enumerate(images)
                if image
            ]
            if not files:
                raise GiteeAPIError("改图图片内容为空")
            data = {
                "model": str(provider.get("model") or ""),
                "prompt": prompt,
            }
            if size:
                data["size"] = size
            payload = await self._request_json(
                "POST",
                f"{self._normalize_openai_base_url(provider.get('base_url'))}/images/edits",
                api_key,
                request_timeout=self._provider_timeout(provider),
                data=data,
                files=files,
            )
            return await self._materialize_entry(self._first_image_entry(payload))
        raise GiteeAPIError(f"图片服务商 {candidate['id']} 暂不支持改图")

    async def _poll_provider_edit(
        self, task_id: str, api_key: str, provider: dict[str, Any]
    ) -> Path:
        try:
            interval = max(0.0, float(provider.get("poll_interval", 5)))
        except (TypeError, ValueError):
            interval = 5.0
        try:
            timeout = max(1.0, min(float(provider.get("poll_timeout", 300)), 3600.0))
        except (TypeError, ValueError):
            timeout = 300.0
        deadline = time.monotonic() + timeout
        base_url = self._normalize_openai_base_url(provider.get("base_url"))
        while True:
            payload = await self._request_json(
                "GET",
                f"{base_url}/task/{task_id}",
                api_key,
                request_timeout=self._provider_timeout(provider),
            )
            status = str(payload.get("status", "")).strip().lower()
            if status in {"success", "succeeded", "completed", "complete"}:
                return await self._materialize_entry(self._first_image_entry(payload))
            if status in {"failed", "failure", "cancelled", "canceled", "error"}:
                raise GiteeAPIError(
                    f"Gitee AI 改图任务失败：{self._error_message(payload, status or 'unknown')}"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise GiteeAPIError(f"Gitee AI 改图任务超时（超过 {int(timeout)} 秒）")
            await asyncio.sleep(min(interval, remaining))

    async def edit(
        self,
        prompt: str,
        images: list[bytes],
        *,
        task_types: list[str] | None = None,
        size: str | None = None,
        operation: str = "edit",
    ) -> Path:
        prompt = str(prompt or "").strip() or "自然真实地优化这张图片"
        if not images:
            raise GiteeAPIError("改图至少需要一张图片")

        if self._is_provider_config():
            last_error: Exception | None = None
            try:
                candidates = self._provider_candidates(operation)
            except GiteeAPIError:
                selfie = self._feature_config("selfie")
                if operation != "selfie" or not bool(
                    selfie.get("use_edit_chain_when_empty", True)
                ):
                    raise
                candidates = self._provider_candidates("edit")
            for candidate in candidates:
                try:
                    requested_size = candidate["output"] or self._operation_output(
                        operation, size
                    )
                    return await self._edit_provider(
                        prompt, images, task_types, requested_size, candidate
                    )
                except Exception as exc:
                    last_error = exc
            raise GiteeAPIError(f"图片修改失败：{last_error}") from last_error

        api_key = await self._next_key()
        raw_task_types = task_types or self.config.get("task_types", ["id"])
        if isinstance(raw_task_types, str):
            raw_task_types = [raw_task_types]
        selected_types = [
            str(item)
            for item in (raw_task_types or [])
            if str(item) in self._EDIT_TASK_TYPES
        ] or ["id"]
        data = [
            ("prompt", prompt),
            ("model", str(self.config.get("edit_model", "Qwen-Image-Edit-2511"))),
            ("num_inference_steps", str(self.config.get("edit_num_inference_steps", 4))),
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
            "POST", f"{self._base_url()}/async/images/edits", api_key,
            data=data, files=files,
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
