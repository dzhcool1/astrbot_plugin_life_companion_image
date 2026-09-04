import sys
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock


_PLUGIN_PARENT = str(Path(__file__).resolve().parents[2])
if _PLUGIN_PARENT not in sys.path:
    sys.path.insert(0, _PLUGIN_PARENT)


def _install_astrbot_stubs():
    if "astrbot" in sys.modules:
        return
    modules = {
        "astrbot": types.ModuleType("astrbot"),
        "astrbot.api": types.ModuleType("astrbot.api"),
        "astrbot.api.event": types.ModuleType("astrbot.api.event"),
        "astrbot.api.message_components": types.ModuleType(
            "astrbot.api.message_components"
        ),
        "astrbot.api.star": types.ModuleType("astrbot.api.star"),
    }

    class _Logger:
        def debug(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

        def info(self, *args, **kwargs):
            pass

    class _Filter:
        @staticmethod
        def _identity(*args, **kwargs):
            def decorate(func):
                return func

            return decorate

        command = _identity
        llm_tool = _identity

    class _Image:
        def __init__(self, encoded=""):
            self.encoded = encoded

        async def convert_to_base64(self):
            return self.encoded

    class _Reply:
        def __init__(self, chain):
            self.chain = chain

    class _File:
        pass

    class _Star:
        def __init__(self, *args, **kwargs):
            pass

    modules["astrbot.api"].logger = _Logger()
    modules["astrbot.api.event"].AstrMessageEvent = object
    modules["astrbot.api.event"].filter = _Filter
    modules["astrbot.api.message_components"].Image = _Image
    modules["astrbot.api.message_components"].File = _File
    modules["astrbot.api.message_components"].Reply = _Reply
    modules["astrbot.api.star"].Context = object
    modules["astrbot.api.star"].Star = _Star

    class _StarTools:
        pass

    modules["astrbot.api.star"].StarTools = _StarTools
    sys.modules.update(modules)


_install_astrbot_stubs()

from astrbot_plugin_life_companion_image.main import (  # noqa: E402
    LifeCompanionImagePlugin,
)


class _LifePlugin:
    def __init__(self):
        self.calls = []

    async def get_life_context(self, *, allow_generate):
        self.calls.append(allow_generate)
        return {"outfit": "测试穿搭"}


class _Context:
    def __init__(self, life_plugin):
        self.life_plugin = life_plugin

    def get_registered_star(self, name):
        if name == "astrbot_plugin_life_companion":
            return types.SimpleNamespace(star_cls=self.life_plugin)
        raise KeyError(name)


class _Provider:
    provider_config = {"id": "chat-provider"}


class _LLMContext:
    def __init__(self, response_text):
        self.response_text = response_text
        self.calls = []

    def get_using_provider(self, *, umo):
        return _Provider()

    async def llm_generate(self, **kwargs):
        self.calls.append(kwargs)
        return types.SimpleNamespace(completion_text=self.response_text)


class _SendEvent:
    unified_msg_origin = "test:conversation"

    def __init__(self):
        self.sent = []

    def get_extra(self, key, default=None):
        return {"provider_request": types.SimpleNamespace(
            contexts=[{"role": "user", "content": "用户说想看一张照片"}],
            system_prompt="她说话自然、亲近。",
        )}.get(key, default)

    def plain_result(self, text):
        return text

    async def send(self, message):
        self.sent.append(message)


class PluginContextTest(unittest.IsolatedAsyncioTestCase):
    async def test_legacy_aiimg_tool_alias_preserves_old_modes(self):
        plugin = LifeCompanionImagePlugin.__new__(LifeCompanionImagePlugin)
        plugin._draw = AsyncMock()
        plugin._edit = AsyncMock()
        plugin._selfie = AsyncMock()
        event = types.SimpleNamespace()

        result = await plugin.aiimg_generate(
            event,
            prompt="咖啡馆里的生活照",
            mode="text",
            aspect_ratio="16:9",
        )

        plugin._draw.assert_awaited_once_with(event, "咖啡馆里的生活照 16:9")
        self.assertIn("图片生成任务已执行", result)

        await plugin.aiimg_generate(event, prompt="换成晴天", mode="edit")
        plugin._edit.assert_awaited_once_with(event, "换成晴天")

        await plugin.aiimg_generate(event, prompt="发张自拍", mode="selfie_ref")
        plugin._selfie.assert_awaited_once_with(event, "发张自拍")

    async def test_life_context_uses_explicit_read_only_flag(self):
        life_plugin = _LifePlugin()
        plugin = LifeCompanionImagePlugin.__new__(LifeCompanionImagePlugin)
        plugin.context = _Context(life_plugin)
        plugin.config = {"use_life_companion": True}

        result = await plugin._life_context(allow_generate=False)

        self.assertEqual(result["outfit"], "测试穿搭")
        self.assertEqual(life_plugin.calls, [False])

    async def test_event_images_includes_images_from_reply_chain(self):
        image_type = sys.modules["astrbot.api.message_components"].Image
        reply_type = sys.modules["astrbot.api.message_components"].Reply
        plugin = LifeCompanionImagePlugin.__new__(LifeCompanionImagePlugin)
        event = types.SimpleNamespace(
            message_obj=types.SimpleNamespace(
                message=[reply_type([image_type("aW1hZ2U=")])]
            )
        )

        result = await plugin._event_images(event)

        self.assertEqual(result, [b"image"])

    async def test_wait_message_uses_current_provider_context_and_filters_technical_text(self):
        plugin = LifeCompanionImagePlugin.__new__(LifeCompanionImagePlugin)
        plugin.context = _LLMContext("正在生成图片，请稍候，已经好了")
        event = _SendEvent()

        result = await plugin._contextual_wait_message(event, "selfie", "在窗边喝咖啡")

        self.assertEqual(result, "好呀，我这就给你拍拍看。")
        self.assertEqual(plugin.context.calls[0]["chat_provider_id"], "chat-provider")
        self.assertEqual(plugin.context.calls[0]["contexts"][0]["role"], "user")
        self.assertIsNone(plugin.context.calls[0]["tools"])
        self.assertIn("她说话自然、亲近。", plugin.context.calls[0]["system_prompt"])

    async def test_image_starts_before_dynamic_wait_message_and_sends_image_after_notice(self):
        plugin = LifeCompanionImagePlugin.__new__(LifeCompanionImagePlugin)
        event = _SendEvent()
        image_started = False

        async def image_operation():
            nonlocal image_started
            image_started = True
            return Path("/tmp/generated-test.jpg")

        async def wait_message(*args):
            self.assertTrue(image_started)
            return "你等等呀，我给你找个好看的角度。"

        plugin._contextual_wait_message = wait_message
        plugin._send_text = AsyncMock()
        plugin._send_image = AsyncMock()

        await plugin._generate_with_notice(
            event, "selfie", "窗边喝咖啡", image_operation()
        )

        plugin._send_text.assert_awaited_once_with(
            event, "你等等呀，我给你找个好看的角度。"
        )
        plugin._send_image.assert_awaited_once_with(
            event, Path("/tmp/generated-test.jpg")
        )
