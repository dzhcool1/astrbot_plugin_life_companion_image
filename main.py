from __future__ import annotations

import base64
import binascii
import inspect
import re
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

    _LIFE_PLUGIN_NAMES = (
        "astrbot_plugin_life_companion",
        "astrbot_plugin_life_scheduler",
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
                    return result
            except TypeError:
                # Do not retry without the keyword: old APIs may generate an LLM
                # schedule when their cache is empty.
                logger.debug("[LifeCompanionImage] %s 不支持只读生活状态接口", plugin_name)
            except Exception as exc:
                logger.debug("[LifeCompanionImage] 读取 %s 失败：%s", plugin_name, exc)
        return {}

    def _default_size(self) -> str:
        return str(self.config.get("default_size", "1024x1024") or "1024x1024").strip()

    def _selfie_prefix(self) -> str:
        return str(
            self.config.get(
                "selfie_prompt_prefix",
                "请根据参考图生成一张自然真实的人像照片。保持第1张参考图的人脸身份和气质一致；"
                "其它参考图只用于服装、姿势、构图或场景参考；不要拼图、不要水印、不要文字，"
                "不要无故增加手机、相机或拍摄界面。",
            )
            or ""
        ).strip()

    async def _prepare_prompt(
        self,
        prompt: str,
        *,
        selfie: bool = False,
        allow_generate: bool = False,
    ) -> tuple[str, str, dict[str, Any]]:
        prompt, size = split_size_suffix(prompt, self._default_size())
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
        raw_paths = self.config.get("reference_images", [])
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
        prompt, size, _ = await self._prepare_prompt(raw_prompt, allow_generate=False)
        if not prompt:
            await self._send_text(event, "请提供图片提示词，例如：/生活照 阳台上的下午茶")
            return
        await self._send_text(event, "正在根据生活状态生成图片，请稍候...")
        path = await (await self._ensure_client()).generate(prompt, size=size)
        await self._send_image(event, path)

    async def _selfie(
        self,
        event: AstrMessageEvent,
        raw_prompt: str,
        *,
        announce: bool = True,
    ) -> None:
        prompt, _, _ = await self._prepare_prompt(
            raw_prompt,
            selfie=True,
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
            await self._send_text(event, "正在结合今日日程生成生活自拍，请稍候...")
        path = await (await self._ensure_client()).edit(prompt, images)
        await self._send_image(event, path)

    async def generate_life_photo(
        self, event: AstrMessageEvent, prompt: str = ""
    ) -> None:
        """Public bridge used by ``astrbot_plugin_life_companion``."""
        await self._selfie(event, prompt, announce=False)

    async def _edit(self, event: AstrMessageEvent, raw_prompt: str) -> None:
        images = await self._event_images(event)
        if not images:
            await self._send_text(event, "请在消息中附带需要修改的图片。")
            return
        prompt, _, _ = await self._prepare_prompt(
            raw_prompt,
            selfie=True,
            allow_generate=False,
        )
        await self._send_text(event, "正在结合今日日程修改图片，请稍候...")
        path = await (await self._ensure_client()).edit(prompt, images)
        await self._send_image(event, path)

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

    @staticmethod
    def _safe_error(exc: Exception) -> str:
        message = " ".join(str(exc).split())
        return message[:180] or "未知错误"
