from __future__ import annotations

import asyncio
import base64
import binascii
import inspect
import re
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import File, Image, Plain, Reply
from astrbot.api.star import Context, Star, StarTools

from .core.gitee_client import GiteeAIClient, GiteeAPIError
from .core.prompting import build_selfie_prompt, split_size_suffix


class ImageCommandWakePrefixFilter(filter.CustomFilter):
    """Keep image commands aligned with AstrBot's configured wake behavior."""

    @staticmethod
    def _wake_prefixes(cfg: object) -> tuple[str, ...]:
        try:
            raw = cfg.get("wake_prefix", ["/"])
        except Exception:
            raw = ["/"]
        if isinstance(raw, str):
            return (raw,) if raw else ("/",)
        if isinstance(raw, (list, tuple, set)):
            return tuple(str(item) for item in raw if str(item)) or ("/",)
        return ("/",)

    @staticmethod
    def _is_private_chat(event: AstrMessageEvent) -> bool:
        try:
            return bool(event.is_private_chat())
        except Exception:
            message_obj = getattr(event, "message_obj", None)
            return not bool(getattr(message_obj, "group", None))

    @staticmethod
    def _plain_has_configured_prefix(
        text: str, prefixes: tuple[str, ...]
    ) -> bool:
        plain = str(text or "").lstrip()
        for prefix in prefixes:
            if not plain.startswith(prefix):
                continue
            end = len(prefix)
            if end < len(plain) and not plain[end].isspace():
                return True
        return False

    def filter(self, event: AstrMessageEvent, cfg: object) -> bool:
        if self._is_private_chat(event):
            return bool(getattr(event, "is_at_or_wake_command", False))
        prefixes = self._wake_prefixes(cfg)
        try:
            chain = event.get_messages()
        except Exception:
            chain = []
        return any(
            isinstance(segment, Plain)
            and self._plain_has_configured_prefix(
                str(getattr(segment, "text", "") or ""), prefixes
            )
            for segment in chain or []
        )


class LifeCompanionImagePlugin(Star):
    """Gitee AI 图片生成 with optional Life Companion context."""

    _WAIT_MESSAGE_TIMEOUT = 2.5
    _REFERENCE_REQUEST_BUDGET = 40 * 1024 * 1024
    _WAIT_MESSAGE_SYSTEM_PROMPT = (
        "你现在只负责替主人先接住用户的话。请结合已有的人设、对话上下文和用户语气，"
        "写一句简短、自然、像真人聊天一样的中文承接话。图片请求已经开始，但结果尚未完成，"
        "所以不要说已经好了、完成了、发给你了或图片来了。不要提生成、插件、模型、API、接口、"
        "提示词、任务、请稍候等技术内容，也不要解释。不要套用固定句式或重复上一轮的说法，"
        "要根据最近对话中的称呼、情绪和具体内容自然发挥。只输出这一句承接话。"
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
    _FOLLOW_UP_IMAGE_RE = re.compile(
        r"^(?:看看?|看(?:一下|下)?|再看(?:一下|下)?|再来一张|来一张)"
        r"(?:嘛|吧|呗|呀|啊|拜托)*$",
        re.IGNORECASE,
    )

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
                    timeline = result.get("timeline")
                    timeline_count = (
                        sum(1 for item in timeline if isinstance(item, dict))
                        if isinstance(timeline, list)
                        else 0
                    )
                    logger.info(
                        "[LifeCompanionImage] 已读取生活状态：来源=%s，日期=%s，日程=%s，时间线=%d条，穿搭=%s",
                        plugin_name,
                        str(result.get("date") or "未知").strip(),
                        "有" if str(result.get("schedule") or "").strip() else "无",
                        timeline_count,
                        "有" if str(result.get("outfit") or "").strip() else "无",
                    )
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
            prompt = self._append_life_context(prompt, life_context)
        schedule = str(life_context.get("schedule") or "").strip()
        timeline = life_context.get("timeline")
        timeline_activities = [
            str(item.get("activity") or item.get("title") or item.get("text") or "").strip()
            for item in timeline
            if isinstance(item, dict)
        ] if isinstance(timeline, list) else []
        logger.info(
            "[LifeCompanionImage] 生活状态已响应给图片提示词：操作=%s，日程=%s，时间线=%d条，日程已写入=%s，时间线已写入=%s",
            operation,
            "有" if schedule else "无",
            len(timeline_activities),
            "是" if schedule and schedule in prompt else "否",
            "是" if any(activity and activity in prompt for activity in timeline_activities) else "否",
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
    def _is_selfie_request(value: Any) -> bool:
        return bool(
            re.search(
                r"自拍|selfie|看看你|发张你|你的照片|拍照|"
                r"拍(?:一张|张)(?:照片|照)|"
                r"(?:就|再|给我|先)?拍(?:一张|张|一下)"
                r"(?:嘛|吧|呗|呀|啊|求你了|求求你了|拜托)*"
                r"(?=$|[，,。.!！？?\s])|"
                r"(?:发|来)(?:一张|张)(?:你的?)?(?:照片|照)",
                str(value or ""),
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _context_text(value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if not isinstance(value, list):
            return ""
        parts = []
        for item in value:
            if isinstance(item, dict) and item.get("type") == "text":
                text = str(item.get("text") or "").strip()
                if text:
                    parts.append(text)
        return " ".join(parts).strip()

    def _conversation_messages(self, event: AstrMessageEvent) -> list[tuple[str, str]]:
        request = self._event_extra(event, "provider_request")
        contexts = self._request_value(request, "contexts")
        if not isinstance(contexts, list):
            return []
        messages = []
        for item in contexts:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").strip().lower()
            if role not in {"user", "assistant"}:
                continue
            text = self._context_text(item.get("content"))
            if text:
                messages.append((role, text))
        return messages

    @classmethod
    def _is_follow_up_image_request(cls, value: Any) -> bool:
        compact = re.sub(r"[\s，,。.!！？?]+", "", str(value or "").strip())
        return bool(cls._FOLLOW_UP_IMAGE_RE.fullmatch(compact))

    def _auto_selfie_request(
        self, event: AstrMessageEvent, raw_prompt: str
    ) -> bool:
        direct_text = str(raw_prompt or "").strip()
        if direct_text and self._is_selfie_request(direct_text):
            return True

        event_text = str(getattr(event, "message_str", "") or "").strip()
        if event_text and self._is_selfie_request(event_text):
            return True

        request = self._event_extra(event, "provider_request")
        provider_prompt = str(
            self._request_value(request, "prompt", "") or ""
        ).strip()
        if provider_prompt and self._is_selfie_request(provider_prompt):
            return True

        messages = self._conversation_messages(event)
        latest_user_index = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if messages[index][0] == "user"
            ),
            None,
        )
        latest_user = (
            messages[latest_user_index][1]
            if latest_user_index is not None
            else ""
        )
        current_text = event_text or provider_prompt or latest_user or direct_text
        if self._is_selfie_request(current_text):
            return True
        if not self._is_follow_up_image_request(current_text):
            return False

        previous_assistant = next(
            (
                messages[index][1]
                for index in range(len(messages) - 1, -1, -1)
                if messages[index][0] == "assistant"
            ),
            "",
        )
        return self._is_selfie_request(previous_assistant)

    @staticmethod
    def _append_life_context(prompt: str, life_context: dict[str, Any]) -> str:
        lines = []
        outfit = str(life_context.get("outfit") or "").strip()
        schedule = str(life_context.get("schedule") or "").strip()
        if outfit:
            lines.append(f"- 今日穿搭：{outfit}")
        if schedule:
            lines.append(f"- 今日日程：{schedule}")
        timeline = life_context.get("timeline")
        if isinstance(timeline, list):
            entries = []
            for item in timeline[:8]:
                if not isinstance(item, dict):
                    continue
                time_value = str(item.get("time") or "").strip()
                activity = str(
                    item.get("activity") or item.get("title") or item.get("text") or ""
                ).strip()
                if activity:
                    entries.append(f"{time_value} {activity[:120]}".strip())
            if entries:
                lines.extend(["- 今日时间线：", *entries])
        if not lines:
            return prompt
        return (
            f"{prompt}\n\n今日生活状态（用户没有指定其它场景时请遵循）：\n"
            + "\n".join(lines)
        ).strip()

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
        current_provider_id = ""
        for item in chain:
            parsed = GiteeAIClient._chain_provider_id(item)
            if parsed:
                current_provider_id = parsed[0]
                break
        selected = None
        fallback = []
        for item in chain:
            parsed = GiteeAIClient._chain_provider_id(item)
            if parsed and parsed[0] in {provider_id, current_provider_id}:
                if parsed[0] == provider_id and selected is None:
                    selected = item
                continue
            fallback.append(item)
        section["chain"] = [
            selected or {"__template_key": "provider", "provider_id": provider_id},
            *fallback,
        ]

    async def _reference_images(self, event: AstrMessageEvent) -> list[bytes]:
        configured = self._configured_reference_paths()
        result: list[bytes] = []
        for path in configured:
            try:
                data = path.read_bytes()
            except OSError:
                continue
            if data:
                result.append(data)
        if result:
            result.extend((await self._event_images(event))[:4])
        else:
            result = (await self._event_images(event))[:8]

        return self._select_reference_images(result)

    def _select_reference_images(self, candidates: list[bytes]) -> list[bytes]:
        selected: list[bytes] = []
        selected_bytes = 0
        skipped = 0
        for image in candidates:
            if not image:
                continue
            if selected and selected_bytes + len(image) > self._REFERENCE_REQUEST_BUDGET:
                skipped += 1
                continue
            selected.append(image)
            selected_bytes += len(image)
        if skipped:
            logger.warning(
                "[LifeCompanionImage] 参考图请求超过单次原图预算：候选=%d张/%d bytes，"
                "保留=%d张/%d bytes，跳过=%d张；未压缩或重编码原图",
                len(candidates),
                sum(len(image) for image in candidates),
                len(selected),
                selected_bytes,
                skipped,
            )
        return selected

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
    def _llm_tool_failure_message(operation: str) -> str:
        return {
            "draw": "图片请求未执行。请自然地告诉用户需要先提供图片描述。",
            "selfie": "图片请求未执行。请自然地告诉用户需要先提供一张人物参考图。",
            "edit": "图片请求未执行。请自然地告诉用户需要在消息中附带要修改的图片。",
        }.get(operation, "图片请求未执行。请自然地告诉用户当前请求缺少必要信息。")

    async def _contextual_wait_message(
        self,
        event: AstrMessageEvent,
        operation: str,
        request_prompt: str,
    ) -> str:
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
                    return ""
                try:
                    provider = get_using_provider(
                        umo=getattr(event, "unified_msg_origin", None)
                    )
                except TypeError:
                    provider = get_using_provider()
                if inspect.isawaitable(provider):
                    provider = await provider
            if provider is None:
                return ""

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
                return ""

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
                return ""
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
            return self._clean_wait_message(text)
        except Exception as exc:
            logger.debug("[LifeCompanionImage] 动态承接话生成失败：%s", exc)
            return ""

    async def _generate_with_notice(
        self,
        event: AstrMessageEvent,
        operation: str,
        request_prompt: str,
        image_operation: Any,
    ) -> None:
        started_at = time.perf_counter()
        image_started_at: float | None = None

        async def run_image_operation() -> Any:
            nonlocal image_started_at
            image_started_at = time.perf_counter()
            return await image_operation

        image_task = asyncio.ensure_future(run_image_operation())
        try:
            try:
                notice = await asyncio.wait_for(
                    self._contextual_wait_message(event, operation, request_prompt),
                    timeout=self._WAIT_MESSAGE_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.debug("[LifeCompanionImage] 动态承接话超时，不发送承接话")
                notice = ""
            except Exception as exc:
                logger.debug("[LifeCompanionImage] 动态承接话异常：%s", exc)
                notice = ""
            if notice:
                await self._send_text(event, notice)
            else:
                logger.info("[LifeCompanionImage] 未生成合适的动态承接话，继续发送图片")
            notice_sent_at = time.perf_counter()
            path = await image_task
            image_finished_at = time.perf_counter()
            send_started_at = image_finished_at
            await self._send_image(event, path)
            finished_at = time.perf_counter()
            logger.info(
                "[LifeCompanionImage] 图片链路完成：操作=%s，进入插件后承接话耗时=%.2fs，图片请求耗时=%.2fs，图片发送耗时=%.2fs，总耗时=%.2fs",
                operation,
                notice_sent_at - started_at,
                image_finished_at - (image_started_at or started_at),
                finished_at - send_started_at,
                finished_at - started_at,
            )
        finally:
            if not image_task.done():
                image_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await image_task

    async def _send_image(self, event: AstrMessageEvent, path: Path) -> None:
        started_at = time.perf_counter()
        try:
            image_size = path.stat().st_size
        except OSError:
            image_size = "unknown"
        logger.info(
            "[LifeCompanionImage] 图片发送开始：文件=%s，大小=%s bytes",
            path.name,
            image_size,
        )
        try:
            await event.send(event.chain_result([Image.fromFileSystem(str(path))]))
            logger.info(
                "[LifeCompanionImage] 图片发送结束：文件=%s，耗时=%.2fs",
                path.name,
                time.perf_counter() - started_at,
            )
        except Exception as first_exc:
            logger.warning("[LifeCompanionImage] 图片消息发送失败，尝试文件发送：%s", first_exc)
            await event.send(event.chain_result([File(name=path.name, file=str(path))]))
            logger.info(
                "[LifeCompanionImage] 图片文件发送结束：文件=%s，耗时=%.2fs",
                path.name,
                time.perf_counter() - started_at,
            )

    async def _ensure_client(self) -> GiteeAIClient:
        if not self.client:
            raise GiteeAPIError("图片插件尚未初始化")
        return self.client

    async def _draw(
        self, event: AstrMessageEvent, raw_prompt: str, *, notify: bool = True
    ) -> str | None:
        if not self._feature_enabled("draw"):
            message = "文生图功能已关闭。"
            if notify:
                await self._send_text(event, message)
            return message
        prompt, size, _ = await self._prepare_prompt(
            raw_prompt, operation="draw", allow_generate=False
        )
        if not prompt:
            message = "请提供图片提示词，例如：/生活照 阳台上的下午茶"
            if notify:
                await self._send_text(event, message)
            return message
        client = await self._ensure_client()
        await self._generate_with_notice(
            event,
            "draw",
            raw_prompt or str(getattr(event, "message_str", "") or "").strip() or prompt,
            client.generate(prompt, size=size),
        )
        return None

    async def _selfie(
        self,
        event: AstrMessageEvent,
        raw_prompt: str,
        *,
        announce: bool = True,
        notify: bool = True,
    ) -> str | None:
        started_at = time.perf_counter()
        if not self._feature_enabled("selfie"):
            message = "生活自拍功能已关闭。"
            if notify:
                await self._send_text(event, message)
            return message
        prompt, size, _ = await self._prepare_prompt(
            raw_prompt,
            selfie=True,
            operation="selfie",
            allow_generate=False,
        )
        images = await self._reference_images(event)
        if not images:
            message = (
                "还没有自拍参考照。请发送一张图片并使用：/生活参考照 设置；"
                "也可以在同一条消息附图后直接使用 /生活自拍。"
            )
            if notify:
                await self._send_text(event, message)
            return message
        client = await self._ensure_client()
        if announce:
            await self._generate_with_notice(
                event,
                "selfie",
                raw_prompt or str(getattr(event, "message_str", "") or "").strip() or prompt,
                client.edit(
                    prompt,
                    images,
                    task_types=self._edit_task_types("selfie"),
                    size=size,
                    operation="selfie",
                ),
            )
            return None
        request_started_at = time.perf_counter()
        logger.info(
            "[LifeCompanionImage] 自拍直达图片请求开始：前置处理耗时=%.2fs，参考图=%d张，尺寸=%s",
            request_started_at - started_at,
            len(images),
            size or "auto",
        )
        path = await client.edit(
            prompt,
            images,
            task_types=self._edit_task_types("selfie"),
            size=size,
            operation="selfie",
        )
        request_finished_at = time.perf_counter()
        await self._send_image(event, path)
        finished_at = time.perf_counter()
        logger.info(
            "[LifeCompanionImage] 自拍直达链路完成：图片请求耗时=%.2fs，图片发送耗时=%.2fs，总耗时=%.2fs",
            request_finished_at - request_started_at,
            finished_at - request_finished_at,
            finished_at - started_at,
        )
        return None

    async def generate_life_photo(
        self, event: AstrMessageEvent, prompt: str = ""
    ) -> None:
        """Public bridge used by ``astrbot_plugin_life_companion``."""
        await self._selfie(event, prompt, announce=True)

    async def _edit(
        self, event: AstrMessageEvent, raw_prompt: str, *, notify: bool = True
    ) -> str | None:
        if not self._feature_enabled("edit"):
            message = "改图功能已关闭。"
            if notify:
                await self._send_text(event, message)
            return message
        images = await self._event_images(event)
        if not images:
            message = "请在消息中附带需要修改的图片。"
            if notify:
                await self._send_text(event, message)
            return message
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
            raw_prompt or str(getattr(event, "message_str", "") or "").strip() or prompt,
            client.edit(
                prompt,
                images,
                task_types=self._edit_task_types("edit"),
                size=size,
                operation="edit",
            ),
        )
        return None

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
            message = f"已将文生图、自拍、改图的首选服务商替换为 {target}（模型：{model}）。原首选服务商已从链路移除。"
        else:
            message = f"已将{self._IMAGE_OPERATION_LABELS[operation]}的首选服务商替换为 {target}（模型：{model}）。原首选服务商已从链路移除。"
        await self._send_text(event, message)

    @filter.command("生活照", alias={"life image", "生活生图"})
    @filter.custom_filter(ImageCommandWakePrefixFilter)
    async def life_image(self, event: AstrMessageEvent):
        event.should_call_llm(True)
        try:
            await self._draw(event, self._raw_arg(event, ("生活照", "life image", "生活生图")))
        except Exception as exc:
            logger.error("[LifeCompanionImage] 文生图失败：%s", exc, exc_info=True)
            await self._send_text(event, f"图片生成失败：{self._safe_error(exc)}")
        finally:
            event.stop_event()

    @filter.command("自拍")
    @filter.custom_filter(ImageCommandWakePrefixFilter)
    async def selfie_command(self, event: AstrMessageEvent):
        """Directly execute the explicit /自拍 command without an LLM tool round-trip."""
        event.should_call_llm(True)
        started_at = time.perf_counter()
        prompt = self._raw_arg(event, ("自拍",))
        logger.info(
            "[LifeCompanionImage] /自拍 已进入图片插件：提示词=%s",
            "有" if prompt else "无",
        )
        try:
            await self._selfie(event, prompt, announce=False)
        except Exception as exc:
            logger.error("[LifeCompanionImage] 自拍失败：%s", exc, exc_info=True)
            await self._send_text(event, f"生活自拍失败：{self._safe_error(exc)}")
        finally:
            event.stop_event()
            logger.info(
                "[LifeCompanionImage] /自拍 命令处理结束：插件处理耗时=%.2fs",
                time.perf_counter() - started_at,
            )

    @filter.command("生活自拍", alias={"life selfie"})
    @filter.custom_filter(ImageCommandWakePrefixFilter)
    async def life_selfie(self, event: AstrMessageEvent):
        event.should_call_llm(True)
        try:
            await self._selfie(
                event,
                self._raw_arg(event, ("生活自拍", "life selfie")),
                announce=False,
            )
        except Exception as exc:
            logger.error("[LifeCompanionImage] 自拍失败：%s", exc, exc_info=True)
            await self._send_text(event, f"生活自拍失败：{self._safe_error(exc)}")
        finally:
            event.stop_event()

    @filter.command("生活改图", alias={"life edit"})
    @filter.custom_filter(ImageCommandWakePrefixFilter)
    async def life_edit(self, event: AstrMessageEvent):
        event.should_call_llm(True)
        try:
            await self._edit(event, self._raw_arg(event, ("生活改图", "life edit")))
        except Exception as exc:
            logger.error("[LifeCompanionImage] 改图失败：%s", exc, exc_info=True)
            await self._send_text(event, f"图片修改失败：{self._safe_error(exc)}")
        finally:
            event.stop_event()

    @filter.command("生活参考照", alias={"life reference"})
    @filter.custom_filter(ImageCommandWakePrefixFilter)
    async def life_reference(self, event: AstrMessageEvent):
        event.should_call_llm(True)
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
        mode: str = "auto",
        size: str = "",
    ) -> str:
        """Generate a life-context image and send it to the current conversation."""
        raw_prompt = f"{prompt} {size}".strip()
        normalized_mode = str(mode or "auto").strip().lower()
        operation = "draw"
        result = None
        logger.info(
            "[LifeCompanionImage] LLM工具 life_companion_image 被调用：模式=%s，提示词=%s，尺寸=%s",
            normalized_mode,
            "有" if raw_prompt else "无",
            size or "auto",
        )
        if normalized_mode in {"selfie", "life_selfie", "selfie_ref"}:
            operation = "selfie"
            result = await self._selfie(event, raw_prompt, notify=False)
        elif normalized_mode in {"edit", "img2img", "aiedit"}:
            operation = "edit"
            result = await self._edit(event, raw_prompt, notify=False)
        elif normalized_mode == "auto" and self._auto_selfie_request(event, raw_prompt):
            operation = "selfie"
            result = await self._selfie(event, raw_prompt, notify=False)
        elif normalized_mode == "auto" and await self._event_images(event):
            operation = "edit"
            result = await self._edit(event, raw_prompt, notify=False)
        else:
            result = await self._draw(event, raw_prompt, notify=False)
        if result:
            logger.info(
                "[LifeCompanionImage] LLM工具未提交图片请求：操作=%s，原因=%s",
                operation,
                result,
            )
            return self._llm_tool_failure_message(operation)
        logger.info("[LifeCompanionImage] LLM工具已提交图片请求：操作=%s", operation)
        return "图片请求已提交，图片插件会直接发送承接话和图片；不要向用户复述内部状态。"

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
        del backend
        raw_prompt = str(prompt or "").strip() or str(reason or "").strip()
        size_tokens: list[str] = []
        seen_size_tokens: set[str] = set()
        for value in (aspect_ratio, resolution, output):
            for token in re.split(r"[\s,/]+", str(value or "").strip()):
                normalized_token = token.casefold()
                if normalized_token in {"", "auto"} or normalized_token in seen_size_tokens:
                    continue
                seen_size_tokens.add(normalized_token)
                size_tokens.append(token)
        size = " ".join(size_tokens)
        if size:
            raw_prompt = f"{raw_prompt} {size}".strip()

        normalized_mode = str(mode or "auto").strip().lower()
        operation = "draw"
        result = None
        logger.info(
            "[LifeCompanionImage] LLM工具 aiimg_generate 被调用：模式=%s，提示词=%s，尺寸=%s",
            normalized_mode,
            "有" if raw_prompt else "无",
            size or "auto",
        )
        if normalized_mode in {"selfie_ref", "selfie", "ref"}:
            operation = "selfie"
            result = await self._selfie(event, raw_prompt, notify=False)
        elif normalized_mode in {"edit", "img2img", "aiedit"}:
            operation = "edit"
            result = await self._edit(event, raw_prompt, notify=False)
        elif normalized_mode == "auto":
            if self._auto_selfie_request(event, raw_prompt):
                operation = "selfie"
                result = await self._selfie(event, raw_prompt, notify=False)
            elif await self._event_images(event):
                operation = "edit"
                result = await self._edit(event, raw_prompt, notify=False)
            else:
                result = await self._draw(event, raw_prompt, notify=False)
        else:
            result = await self._draw(event, raw_prompt, notify=False)
        if result:
            logger.info(
                "[LifeCompanionImage] LLM工具未提交图片请求：工具=aiimg_generate，操作=%s，原因=%s",
                operation,
                result,
            )
            return self._llm_tool_failure_message(operation)
        logger.info(
            "[LifeCompanionImage] LLM工具已提交图片请求：工具=aiimg_generate，操作=%s",
            operation,
        )
        return "图片请求已提交，图片插件会直接发送承接话和图片；不要向用户复述内部状态。"

    @staticmethod
    def _safe_error(exc: Exception) -> str:
        message = " ".join(str(exc).split())
        return message[:180] or "未知错误"
