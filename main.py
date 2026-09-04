from __future__ import annotations

import asyncio
import base64
import binascii
import inspect
import re
from contextlib import suppress
from pathlib import Path
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import File, Image, Reply
from astrbot.api.star import Context, Star, StarTools

from .core.gitee_client import GiteeAIClient, GiteeAPIError
from .core.prompting import build_selfie_prompt, split_size_suffix


class LifeCompanionImagePlugin(Star):
    """Gitee AI 图片生成 with optional Life Companion context."""

    _WAIT_MESSAGE_TIMEOUT = 2.5
    _WAIT_MESSAGE_SYSTEM_PROMPT = (
        "你现在只负责替主人先接住用户的话。请结合已有的人设、对话上下文和用户语气，"
        "写一句简短、自然、像真人聊天一样的中文承接话。图片工作已经开始，但结果尚未完成，"
        "所以不要说已经好了、完成了、发给你了或图片来了。不要提生成、插件、模型、API、接口、"
        "提示词、任务、请稍候等技术内容，也不要解释。只输出这一句承接话。"
    )

    _LIFE_PLUGIN_NAMES = (
        "astrbot_plugin_life_companion",
        "astrbot_plugin_life_scheduler",
    )
    _IMAGE_OPERATION_LABELS = {
        "draw": "文生图",
        "selfie": "自拍",
        "edit": "改图",
    }
    _IMAGE_OPERATION_ALIASES = {
        "文生图": "draw",
        "生活照": "draw",
        "draw": "draw",
        "自拍": "selfie",
        "生活自拍": "selfie",
        "selfie": "selfie",
        "改图": "edit",
        "图生图": "edit",
        "生活改图": "edit",
        "edit": "edit",
        "全部": "all",
        "所有": "all",
        "all": "all",
    }

    def __init__(self, context: Context, config: dict[str, Any]):
        super().__init__(context)
        self.context = context
        self.config = config
        self.data_dir = StarTools.get_data_dir("astrbot_plugin_life_companion_image")
        self.reference_dir = self.data_dir / "references"
        self.reference_dir.mkdir(parents=True, exist_ok=True)
        self.client: GiteeAIClient | None = None

    async def initialize(self) -> None:
        self.client = GiteeAIClient(self.config, self.data_dir)
        logger.info("[LifeCompanionImage] 插件初始化完成")

    async def terminate(self) -> None:
        if self.client:
            await self.client.close()
            self.client = None

    def _save_config(self) -> None:
        save_config = getattr(self.config, "save_config", None)
        if callable(save_config):
            save_config()

    def _bool_config(self, key: str, default: bool) -> bool:
        value = self.config.get(key, default)
        if isinstance(value, str):
            return value.strip().lower() not in {"", "0", "false", "no", "off"}
        return bool(value)

    async def _life_context(self, *, allow_generate: bool) -> dict[str, Any]:
        if not self._bool_config("use_life_companion", True):
            return {}
        get_registered_star = getattr(self.context, "get_registered_star", None)
        if not callable(get_registered_star):
            return {}

        for plugin_name in self._LIFE_PLUGIN_NAMES:
            try:
                metadata = get_registered_star(plugin_name)
                plugin = getattr(metadata, "star_cls", None)
                get_context = getattr(plugin, "get_life_context", None)
                if not callable(get_context):
                    continue
                result = get_context(allow_generate=allow_generate)
                if inspect.isawaitable(result):
                    result = await result
                if isinstance(result, dict):
                    return result
            except TypeError:
                # Do not retry without the keyword: old APIs may generate an LLM
                # schedule when their cache is empty.
                logger.debug("[LifeCompanionImage] %s 不支持只读生活状态接口", plugin_name)
            except Exception as exc:
                logger.debug("[LifeCompanionImage] 读取 %s 失败：%s", plugin_name, exc)
        return {}

    def _default_size(self) -> str:
        features = self.config.get("features")
        draw = features.get("draw") if isinstance(features, dict) else {}
        if isinstance(draw, dict) and draw.get("default_output"):
            return str(draw["default_output"]).strip()
        return str(self.config.get("default_size", "1024x1024") or "1024x1024").strip()

    def _selfie_prefix(self) -> str:
        features = self.config.get("features")
        selfie = features.get("selfie") if isinstance(features, dict) else {}
        if isinstance(selfie, dict) and selfie.get("prompt_prefix"):
            return str(selfie["prompt_prefix"]).strip()
        return str(
            self.config.get(
                "selfie_prompt_prefix",
                "请根据参考图生成一张自然真实的人像照片。保持第1张参考图的人脸身份和气质一致；"
                "其它参考图只用于服装、姿势、构图或场景参考；不要拼图、不要水印、不要文字，"
                "画面采用由他人拍摄的自然生活照；人物不做自拍动作，拍摄设备位于画面外。",
            )
            or ""
        ).strip()

    async def _prepare_prompt(
        self,
        prompt: str,
        *,
        selfie: bool = False,
        operation: str = "draw",
        allow_generate: bool = False,
    ) -> tuple[str, str, dict[str, Any]]:
        prompt, size = split_size_suffix(
            prompt, self._operation_size(operation) or self._default_size()
        )
        life_context = await self._life_context(allow_generate=allow_generate)
        if selfie:
            prompt = build_selfie_prompt(prompt, life_context, self._selfie_prefix())
        elif not prompt:
            prompt = str(life_context.get("image_prompt") or "").strip()
            if not prompt:
                outfit = str(life_context.get("outfit") or "").strip()
                schedule = str(life_context.get("schedule") or "").strip()
                prompt = (
                    f"自然真实的今日生活照，穿着：{outfit or '日常穿搭'}；"
                    f"场景和活动：{schedule or '轻松日常'}"
                )
        return prompt, size, life_context

    @staticmethod
    def _raw_arg(event: AstrMessageEvent, command_names: tuple[str, ...]) -> str:
        raw = str(getattr(event, "message_str", "") or "").strip()
        for name in command_names:
            match = re.search(
                rf"(?:^|\s)[/!！.。．]?{re.escape(name)}(?:\s|$)(.*)",
                raw,
                flags=re.IGNORECASE,
            )
            if match:
                return match.group(1).strip()
        return raw.split(maxsplit=1)[1].strip() if " " in raw else ""

    @staticmethod
    def _decode_image_data(value: str | bytes) -> bytes:
        if isinstance(value, bytes):
            return value
        text = str(value or "").strip()
        if "," in text and text.lower().startswith("data:"):
            text = text.split(",", 1)[1]
        try:
            return base64.b64decode(text, validate=False)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("图片 Base64 无效") from exc

    async def _read_image_segments(self, segments: list[Any]) -> list[bytes]:
        images: list[bytes] = []
        for segment in segments:
            if isinstance(segment, Reply):
                images.extend(
                    await self._read_image_segments(
                        getattr(segment, "chain", []) or []
                    )
                )
                continue
            if not isinstance(segment, Image):
                continue
            try:
                convert = getattr(segment, "convert_to_base64", None)
                if callable(convert):
                    encoded = convert()
                    if inspect.isawaitable(encoded):
                        encoded = await encoded
                    data = self._decode_image_data(encoded)
                    if data:
                        images.append(data)
                        continue
                file_path = str(getattr(segment, "file", "") or "")
                if file_path and Path(file_path).is_file():
                    data = Path(file_path).read_bytes()
                    if data:
                        images.append(data)
            except (OSError, ValueError) as exc:
                logger.warning("[LifeCompanionImage] 读取消息图片失败：%s", exc)
        return images

    async def _event_images(self, event: AstrMessageEvent) -> list[bytes]:
        message_obj = getattr(event, "message_obj", None)
        segments = getattr(message_obj, "message", []) or []
        return await self._read_image_segments(segments)

    def _configured_reference_paths(self) -> list[Path]:
        features = self.config.get("features")
        selfie = features.get("selfie") if isinstance(features, dict) else {}
        raw_paths = (
            selfie.get("reference_images", [])
            if isinstance(selfie, dict) and "reference_images" in selfie
            else self.config.get("reference_images", [])
        )
        if isinstance(raw_paths, str):
            raw_paths = [raw_paths]
        if not isinstance(raw_paths, list):
            return []

        paths: list[Path] = []
        base = self.data_dir.resolve()
        for value in raw_paths:
            path = Path(str(value or "").strip())
            if not str(path):
                continue
            if not path.is_absolute():
                path = base / path
            path = path.resolve(strict=False)
            try:
                path.relative_to(base)
            except ValueError:
                logger.warning("[LifeCompanionImage] 忽略数据目录外的参考图：%s", path)
                continue
            if path.is_file():
                paths.append(path)
        return paths[:8]

    def _operation_size(self, operation: str) -> str:
        features = self.config.get("features")
        section = features.get(operation) if isinstance(features, dict) else {}
        if isinstance(section, dict):
            output = str(section.get("default_output") or "").strip()
            aspect_ratio = str(section.get("default_aspect_ratio") or "").strip()
            if operation == "selfie" and aspect_ratio and ":" not in output:
                return f"{aspect_ratio} {output}".strip()
            return output
        return ""

    def _feature_enabled(self, operation: str, default: bool = True) -> bool:
        features = self.config.get("features")
        section = features.get(operation) if isinstance(features, dict) else {}
        if isinstance(section, dict) and "enabled" in section:
            return self._bool_config_value(section.get("enabled"), default)
        return default

    @staticmethod
    def _bool_config_value(value: Any, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, str):
            return value.strip().lower() not in {"", "0", "false", "no", "off"}
        return bool(value)

    def _edit_task_types(self, operation: str) -> list[str] | None:
        features = self.config.get("features")
        section = features.get(operation) if isinstance(features, dict) else {}
        if isinstance(section, dict):
            value = section.get("gitee_task_types")
            if isinstance(value, list):
                return [str(item) for item in value]
        value = self.config.get("task_types")
        if isinstance(value, list):
            return [str(item) for item in value]
        return None

    def _configured_providers(self) -> list[dict[str, Any]]:
        providers = self.config.get("providers")
        if not isinstance(providers, list):
            return []
        return [
            provider
            for provider in providers
            if isinstance(provider, dict) and str(provider.get("id") or "").strip()
        ]

    @staticmethod
    def _provider_display_name(provider: dict[str, Any]) -> str:
        provider_id = str(provider.get("id") or "").strip()
        label = str(provider.get("label") or "").strip()
        if label and label.casefold() != provider_id.casefold():
            return f"{label}（{provider_id}）"
        return label or provider_id

    @staticmethod
    def _provider_match_names(provider: dict[str, Any]) -> set[str]:
        return {
            str(provider.get(key) or "").strip().casefold()
            for key in ("id", "label")
            if str(provider.get(key) or "").strip()
        }

    @staticmethod
    def _provider_capabilities(provider: dict[str, Any]) -> set[str]:
        template = GiteeAIClient._provider_template(provider)
        capabilities = {
            "gemini_native": {"draw", "selfie", "edit"},
            "gitee_images": {"draw"},
            "gitee_async": {"selfie", "edit"},
            "openai_images": {"draw", "selfie", "edit"},
            "gemini_openai_images": {"draw", "selfie", "edit"},
        }.get(template, set())
        if "supports_edit" in provider and not LifeCompanionImagePlugin._bool_config_value(
            provider.get("supports_edit"), True
        ):
            capabilities = capabilities - {"selfie", "edit"}
        return capabilities

    def _provider_for_id(self, provider_id: str) -> dict[str, Any] | None:
        for provider in self._configured_providers():
            if str(provider.get("id") or "").strip() == provider_id:
                return provider
        return None

    def _find_provider(
        self, query: str
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        normalized = " ".join(str(query or "").split()).casefold()
        if not normalized:
            return None, []
        matches = [
            provider
            for provider in self._configured_providers()
            if normalized in self._provider_match_names(provider)
        ]
        return (matches[0] if len(matches) == 1 else None), matches

    def _chain_provider_ids(self, operation: str) -> list[str]:
        features = self.config.get("features")
        section = features.get(operation) if isinstance(features, dict) else {}
        chain = section.get("chain") if isinstance(section, dict) else []
        if not isinstance(chain, list):
            return []
        result = []
        for item in chain:
            parsed = GiteeAIClient._chain_provider_id(item)
            if parsed:
                result.append(parsed[0])
        return result

    def _current_provider_id(self, operation: str) -> str:
        provider_ids = self._chain_provider_ids(operation)
        if provider_ids:
            return provider_ids[0]
        if operation == "selfie":
            features = self.config.get("features")
            selfie = features.get("selfie", {}) if isinstance(features, dict) else {}
            if isinstance(selfie, dict) and self._bool_config_value(
                selfie.get("use_edit_chain_when_empty"), True
            ):
                return (self._chain_provider_ids("edit") or [""])[0]
        return ""

    def _format_provider_ref(self, provider_id: str) -> str:
        provider = self._provider_for_id(provider_id)
        return self._provider_display_name(provider) if provider else provider_id or "未配置"

    def _provider_list_text(self) -> str:
        providers = self._configured_providers()
        if not providers:
            return "当前没有配置 providers 服务商，请先在插件配置中添加服务商。"

        lines = ["当前已配置的生图服务商和模型："]
        for provider in providers:
            model = str(provider.get("model") or "").strip() or "未填写模型"
            supported = [
                self._IMAGE_OPERATION_LABELS[operation]
                for operation in ("draw", "selfie", "edit")
                if operation in self._provider_capabilities(provider)
            ]
            support_text = "、".join(supported) or "当前客户端不支持"
            lines.append(
                f"- {self._provider_display_name(provider)}：{model}（支持：{support_text}）"
            )

        lines.append("")
        lines.append("当前首选服务商：")
        for operation, label in self._IMAGE_OPERATION_LABELS.items():
            lines.append(f"- {label}：{self._format_provider_ref(self._current_provider_id(operation))}")
        lines.extend(
            [
                "",
                "切换用法：",
                "/切换生图 服务商名称（切换文生图、自拍、改图）",
                "/切换生图 自拍 服务商名称",
                "/切换生图 文生图 服务商名称",
                "/切换生图 改图 服务商名称",
            ]
        )
        return "\n".join(lines)

    @classmethod
    def _parse_switch_request(cls, raw: str) -> tuple[str, str]:
        value = " ".join(str(raw or "").split())
        if not value:
            return "", ""
        first, separator, remainder = value.partition(" ")
        operation = cls._IMAGE_OPERATION_ALIASES.get(first.casefold(), "all")
        if not separator:
            return operation, "" if operation != "all" else value
        return operation, remainder.strip()

    def _move_provider_to_front(self, operation: str, provider_id: str) -> None:
        features = self.config.get("features")
        if not isinstance(features, dict):
            raise GiteeAPIError("插件配置缺少 features，无法切换服务商")
        section = features.get(operation)
        if not isinstance(section, dict):
            section = {}
            features[operation] = section
        raw_chain = section.get("chain")
        chain = list(raw_chain) if isinstance(raw_chain, list) else []
        selected = None
        fallback = []
        for item in chain:
            parsed = GiteeAIClient._chain_provider_id(item)
            if parsed and parsed[0] == provider_id:
                if selected is None:
                    selected = item
                continue
            fallback.append(item)
        section["chain"] = [selected or {"provider_id": provider_id}, *fallback]

    async def _reference_images(self, event: AstrMessageEvent) -> list[bytes]:
        configured = self._configured_reference_paths()
        result = []
        for path in configured:
            try:
                data = path.read_bytes()
            except OSError:
                continue
            if data:
                result.append(data)
        if result:
            result.extend((await self._event_images(event))[:4])
            return result
        return (await self._event_images(event))[:8]

    @staticmethod
    def _image_extension(data: bytes) -> str:
        if data.startswith(b"\x89PNG"):
            return "png"
        if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
            return "webp"
        return "jpg"

    async def _send_text(self, event: AstrMessageEvent, text: str) -> None:
        await event.send(event.plain_result(text))

    @staticmethod
    def _event_extra(event: AstrMessageEvent, key: str, default: Any = None) -> Any:
        getter = getattr(event, "get_extra", None)
        if not callable(getter):
            return default
        return getter(key, default)

    @staticmethod
    def _request_value(request: Any, key: str, default: Any = None) -> Any:
        if isinstance(request, dict):
            return request.get(key, default)
        return getattr(request, key, default)

    @staticmethod
    def _clean_wait_message(value: Any) -> str:
        text = str(value or "").strip()
        text = re.sub(r"<think>.*?</think>", " ", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"```(?:\w+)?|```", "", text)
        text = re.sub(r"^(?:assistant|回复|答复)\s*[:：]\s*", "", text, flags=re.IGNORECASE)
        text = " ".join(text.split()).strip(" `\"'“”‘’")
        if not text:
            return ""
        blocked = re.compile(
            r"正在生成|生成中|生图|请稍候|请稍等|稍候|插件|模型|api|接口|提示词|"
            r"任务|已经好了|已完成|完成了|发给你|发送给你|图片来了|图片已",
            re.IGNORECASE,
        )
        if blocked.search(text):
            return ""
        if len(text) > 72:
            first_sentence = re.split(r"[。！？!?]", text, maxsplit=1)[0].strip()
            text = first_sentence if 0 < len(first_sentence) <= 72 else ""
        return text

    @staticmethod
    def _wait_message_fallback(operation: str) -> str:
        return {
            "draw": "好呀，我这就按你说的来一张。",
            "selfie": "好呀，我这就给你拍拍看。",
            "edit": "嗯，我照你说的改改看。",
        }.get(operation, "好呀，我先按你说的来看看。")

    async def _contextual_wait_message(
        self,
        event: AstrMessageEvent,
        operation: str,
        request_prompt: str,
    ) -> str:
        fallback = self._wait_message_fallback(operation)
        try:
            provider = None
            selected_provider = str(
                self._event_extra(event, "selected_provider", "") or ""
            ).strip()
            if selected_provider:
                get_provider = getattr(self.context, "get_provider_by_id", None)
                if callable(get_provider):
                    provider = get_provider(selected_provider)
                    if inspect.isawaitable(provider):
                        provider = await provider
            if provider is None:
                get_using_provider = getattr(self.context, "get_using_provider", None)
                if not callable(get_using_provider):
                    return fallback
                try:
                    provider = get_using_provider(
                        umo=getattr(event, "unified_msg_origin", None)
                    )
                except TypeError:
                    provider = get_using_provider()
                if inspect.isawaitable(provider):
                    provider = await provider
            if provider is None:
                return fallback

            provider_config = getattr(provider, "provider_config", {})
            provider_id = (
                provider_config.get("id")
                if isinstance(provider_config, dict)
                else None
            )
            if not provider_id:
                meta = getattr(provider, "meta", None)
                if callable(meta):
                    provider_meta = meta()
                    if inspect.isawaitable(provider_meta):
                        provider_meta = await provider_meta
                    provider_id = getattr(provider_meta, "id", None)
            if not provider_id:
                return fallback

            request = self._event_extra(event, "provider_request")
            contexts = self._request_value(request, "contexts")
            if isinstance(contexts, list):
                contexts = list(contexts)
            else:
                contexts = None
            system_prompt = str(
                self._request_value(request, "system_prompt", "") or ""
            ).strip()
            system_prompt = (
                f"{system_prompt}\n\n{self._WAIT_MESSAGE_SYSTEM_PROMPT}"
                if system_prompt
                else self._WAIT_MESSAGE_SYSTEM_PROMPT
            )
            prompt = (
                f"当前要处理的事情：{operation}。\n"
                f"用户这次提出的图片要求：{str(request_prompt or '').strip() or '结合当前对话自然处理'}\n"
                "请先给用户一句符合当前语气的承接话，只要一句。"
            )
            llm_generate = getattr(self.context, "llm_generate", None)
            if not callable(llm_generate):
                return fallback
            response = await llm_generate(
                chat_provider_id=str(provider_id),
                prompt=prompt,
                contexts=contexts,
                system_prompt=system_prompt,
                tools=None,
            )
            text = getattr(response, "completion_text", "")
            if not text:
                result_chain = getattr(response, "result_chain", None)
                get_plain_text = getattr(result_chain, "get_plain_text", None)
                if callable(get_plain_text):
                    text = get_plain_text()
            return self._clean_wait_message(text) or fallback
        except Exception as exc:
            logger.debug("[LifeCompanionImage] 动态承接话生成失败：%s", exc)
            return fallback

    async def _generate_with_notice(
        self,
        event: AstrMessageEvent,
        operation: str,
        request_prompt: str,
        image_operation: Any,
    ) -> None:
        image_task = asyncio.create_task(image_operation)
        try:
            await asyncio.sleep(0)
            try:
                notice = await asyncio.wait_for(
                    self._contextual_wait_message(event, operation, request_prompt),
                    timeout=self._WAIT_MESSAGE_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.debug("[LifeCompanionImage] 动态承接话超时，使用自然兜底")
                notice = self._wait_message_fallback(operation)
            except Exception as exc:
                logger.debug("[LifeCompanionImage] 动态承接话异常：%s", exc)
                notice = self._wait_message_fallback(operation)
            await self._send_text(event, notice)
            path = await image_task
            await self._send_image(event, path)
        finally:
            if not image_task.done():
                image_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await image_task

    async def _send_image(self, event: AstrMessageEvent, path: Path) -> None:
        try:
            await event.send(event.chain_result([Image.fromFileSystem(str(path))]))
        except Exception as first_exc:
            logger.warning("[LifeCompanionImage] 图片消息发送失败，尝试文件发送：%s", first_exc)
            await event.send(
                event.chain_result([File(name=path.name, file=str(path))])
            )

    async def _ensure_client(self) -> GiteeAIClient:
        if not self.client:
            raise GiteeAPIError("图片插件尚未初始化")
        return self.client

    async def _draw(self, event: AstrMessageEvent, raw_prompt: str) -> None:
        if not self._feature_enabled("draw"):
            await self._send_text(event, "文生图功能已关闭。")
            return
        prompt, size, _ = await self._prepare_prompt(
            raw_prompt, operation="draw", allow_generate=False
        )
        if not prompt:
            await self._send_text(event, "请提供图片提示词，例如：/生活照 阳台上的下午茶")
            return
        client = await self._ensure_client()
        await self._generate_with_notice(
            event,
            "draw",
            raw_prompt or prompt,
            client.generate(prompt, size=size),
        )

    async def _selfie(
        self,
        event: AstrMessageEvent,
        raw_prompt: str,
        *,
        announce: bool = True,
    ) -> None:
        if not self._feature_enabled("selfie"):
            await self._send_text(event, "生活自拍功能已关闭。")
            return
        prompt, size, _ = await self._prepare_prompt(
            raw_prompt,
            selfie=True,
            operation="selfie",
            allow_generate=False,
        )
        images = await self._reference_images(event)
        if not images:
            await self._send_text(
                event,
                "还没有自拍参考照。请发送一张图片并使用：/生活参考照 设置；"
                "也可以在同一条消息附图后直接使用 /生活自拍。",
            )
            return
        if announce:
            client = await self._ensure_client()
            await self._generate_with_notice(
                event,
                "selfie",
                raw_prompt or prompt,
                client.edit(
                    prompt,
                    images,
                    task_types=self._edit_task_types("selfie"),
                    size=size,
                    operation="selfie",
                ),
            )
            return
        client = await self._ensure_client()
        path = await client.edit(
            prompt,
            images,
            task_types=self._edit_task_types("selfie"),
            size=size,
            operation="selfie",
        )
        await self._send_image(event, path)

    async def generate_life_photo(
        self, event: AstrMessageEvent, prompt: str = ""
    ) -> None:
        """Public bridge used by ``astrbot_plugin_life_companion``."""
        await self._selfie(event, prompt, announce=True)

    async def _edit(self, event: AstrMessageEvent, raw_prompt: str) -> None:
        if not self._feature_enabled("edit"):
            await self._send_text(event, "改图功能已关闭。")
            return
        images = await self._event_images(event)
        if not images:
            await self._send_text(event, "请在消息中附带需要修改的图片。")
            return
        prompt, size, _ = await self._prepare_prompt(
            raw_prompt,
            selfie=True,
            operation="edit",
            allow_generate=False,
        )
        client = await self._ensure_client()
        await self._generate_with_notice(
            event,
            "edit",
            raw_prompt or prompt,
            client.edit(
                prompt,
                images,
                task_types=self._edit_task_types("edit"),
                size=size,
                operation="edit",
            ),
        )

    @filter.command("生图模型", alias={"image models"})
    async def image_models(self, event: AstrMessageEvent):
        await self._send_text(event, self._provider_list_text())

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("切换生图", alias={"switch image"})
    async def switch_image_provider(self, event: AstrMessageEvent):
        operation, provider_query = self._parse_switch_request(
            self._raw_arg(event, ("切换生图", "switch image"))
        )
        if not operation or not provider_query:
            await self._send_text(
                event,
                "用法：/切换生图 服务商名称；或 /切换生图 自拍/文生图/改图 服务商名称。"
                "先使用 /生图模型 查看服务商名称和模型。",
            )
            return

        provider, matches = self._find_provider(provider_query)
        if not provider:
            if len(matches) > 1:
                names = "、".join(self._provider_display_name(item) for item in matches)
                await self._send_text(
                    event,
                    f"服务商名称“{provider_query}”对应多个配置：{names}，请使用唯一的服务商 ID。",
                )
            else:
                available = "、".join(
                    self._provider_display_name(item)
                    for item in self._configured_providers()
                )
                await self._send_text(
                    event,
                    f"没有找到服务商“{provider_query}”。当前可用服务商：{available or '无'}。",
                )
            return

        operations = (
            tuple(self._IMAGE_OPERATION_LABELS)
            if operation == "all"
            else (operation,)
        )
        unsupported = [
            self._IMAGE_OPERATION_LABELS[item]
            for item in operations
            if item not in self._provider_capabilities(provider)
        ]
        if unsupported:
            await self._send_text(
                event,
                f"服务商 {self._provider_display_name(provider)} 不支持：{'、'.join(unsupported)}，本次未切换。",
            )
            return

        for item in operations:
            self._move_provider_to_front(item, str(provider["id"]).strip())
        self._save_config()
        target = self._provider_display_name(provider)
        model = str(provider.get("model") or "").strip() or "未填写模型"
        if operation == "all":
            message = f"已将文生图、自拍、改图的首选服务商切换为 {target}（模型：{model}）。原有服务商已保留为兜底。"
        else:
            message = f"已将{self._IMAGE_OPERATION_LABELS[operation]}的首选服务商切换为 {target}（模型：{model}）。原有服务商已保留为兜底。"
        await self._send_text(event, message)

    @filter.command("生活照", alias={"life image", "生活生图"})
    async def life_image(self, event: AstrMessageEvent):
        try:
            await self._draw(event, self._raw_arg(event, ("生活照", "life image", "生活生图")))
        except Exception as exc:
            logger.error("[LifeCompanionImage] 文生图失败：%s", exc, exc_info=True)
            await self._send_text(event, f"图片生成失败：{self._safe_error(exc)}")

    @filter.command("生活自拍", alias={"life selfie"})
    async def life_selfie(self, event: AstrMessageEvent):
        try:
            await self._selfie(event, self._raw_arg(event, ("生活自拍", "life selfie")))
        except Exception as exc:
            logger.error("[LifeCompanionImage] 自拍失败：%s", exc, exc_info=True)
            await self._send_text(event, f"生活自拍失败：{self._safe_error(exc)}")

    @filter.command("生活改图", alias={"life edit"})
    async def life_edit(self, event: AstrMessageEvent):
        try:
            await self._edit(event, self._raw_arg(event, ("生活改图", "life edit")))
        except Exception as exc:
            logger.error("[LifeCompanionImage] 改图失败：%s", exc, exc_info=True)
            await self._send_text(event, f"图片修改失败：{self._safe_error(exc)}")

    @filter.command("生活参考照", alias={"life reference"})
    async def life_reference(self, event: AstrMessageEvent):
        arg = self._raw_arg(event, ("生活参考照", "life reference")).lower()
        if arg in {"查看", "show", "看"}:
            count = len(self._configured_reference_paths())
            await self._send_text(event, f"当前已配置 {count} 张生活自拍参考照。")
            return
        if arg in {"删除", "delete", "del", "clear"}:
            features = self.config.get("features")
            selfie = features.get("selfie") if isinstance(features, dict) else None
            if isinstance(selfie, dict) and "reference_images" in selfie:
                selfie["reference_images"] = []
            else:
                self.config["reference_images"] = []
            self._save_config()
            await self._send_text(event, "已清除生活自拍参考照配置。")
            return
        if arg not in {"设置", "set"}:
            await self._send_text(event, "用法：发送图片 + /生活参考照 设置；查看：/生活参考照 查看；删除：/生活参考照 删除")
            return

        images = await self._event_images(event)
        if not images:
            await self._send_text(event, "没有读取到图片，请把图片和“生活参考照 设置”放在同一条消息中。")
            return
        saved_paths = []
        for index, data in enumerate(images[:8], 1):
            path = self.reference_dir / f"reference_{index}.{self._image_extension(data)}"
            path.write_bytes(data)
            saved_paths.append(str(path.relative_to(self.data_dir)))
        features = self.config.get("features")
        selfie = features.get("selfie") if isinstance(features, dict) else None
        if isinstance(selfie, dict) and "reference_images" in selfie:
            selfie["reference_images"] = saved_paths
        else:
            self.config["reference_images"] = saved_paths
        self._save_config()
        await self._send_text(event, f"已保存 {len(saved_paths)} 张生活自拍参考照。")

    @filter.llm_tool(name="life_companion_image")
    async def life_companion_image(
        self,
        event: AstrMessageEvent,
        prompt: str = "",
        mode: str = "life_photo",
        size: str = "",
    ) -> str:
        """Generate a life-context image and send it to the current conversation."""
        raw_prompt = f"{prompt} {size}".strip()
        if mode.strip().lower() in {"selfie", "life_selfie", "selfie_ref"}:
            await self._selfie(event, raw_prompt)
        else:
            await self._draw(event, raw_prompt)
        return "图片生成任务已执行；如果图片没有出现，请检查插件配置和日志。"

    @filter.llm_tool(name="aiimg_generate")
    async def aiimg_generate(
        self,
        event: AstrMessageEvent,
        prompt: str = "",
        mode: str = "auto",
        backend: str = "auto",
        output: str = "",
        aspect_ratio: str = "auto",
        resolution: str = "auto",
        reason: str = "",
    ) -> str:
        """Keep the former Gitee tool name available after plugin replacement."""
        del backend, resolution
        raw_prompt = str(prompt or "").strip() or str(reason or "").strip()
        size = next(
            (
                str(value).strip()
                for value in (aspect_ratio, output)
                if str(value or "").strip().lower() not in {"", "auto"}
            ),
            "",
        )
        if size:
            raw_prompt = f"{raw_prompt} {size}".strip()

        normalized_mode = str(mode or "auto").strip().lower()
        if normalized_mode in {"selfie_ref", "selfie", "ref"}:
            await self._selfie(event, raw_prompt)
        elif normalized_mode in {"edit", "img2img", "aiedit"}:
            await self._edit(event, raw_prompt)
        elif normalized_mode == "auto":
            if re.search(r"自拍|selfie|看看你|发张你|你的照片", raw_prompt, re.IGNORECASE):
                await self._selfie(event, raw_prompt)
            elif await self._event_images(event):
                await self._edit(event, raw_prompt)
            else:
                await self._draw(event, raw_prompt)
        else:
            await self._draw(event, raw_prompt)
        return "图片生成任务已执行；如果图片没有出现，请检查插件配置和日志。"

    @staticmethod
    def _safe_error(exc: Exception) -> str:
        message = " ".join(str(exc).split())
        return message[:180] or "未知错误"
