import asyncio
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import patch

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
        class PermissionType:
            ADMIN = "admin"

        class CustomFilter:
            def __init__(self, raise_error=True, **kwargs):
                self.raise_error = raise_error

        @staticmethod
        def _identity(*args, **kwargs):
            def decorate(func):
                return func

            return decorate

        command = _identity
        custom_filter = _identity
        llm_tool = _identity
        permission_type = _identity

    class _Image:
        def __init__(self, encoded=""):
            self.encoded = encoded

        async def convert_to_base64(self):
            return self.encoded

    class _Plain:
        def __init__(self, text=""):
            self.text = text

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
    modules["astrbot.api.message_components"].Plain = _Plain
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
    ImageCommandWakePrefixFilter,
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


class _CommandEvent:
    def __init__(self, message_str):
        self.message_str = message_str
        self.call_llm = False
        self.stopped = False

    def should_call_llm(self, value):
        self.call_llm = value

    def stop_event(self):
        self.stopped = True


class _GroupCommandEvent:
    def __init__(self, segments):
        self.segments = segments

    def is_private_chat(self):
        return False

    def get_messages(self):
        return self.segments


class PluginContextTest(unittest.IsolatedAsyncioTestCase):
    def test_image_command_filter_requires_configured_group_prefix(self):
        command_filter = ImageCommandWakePrefixFilter()
        event = _GroupCommandEvent([types.SimpleNamespace(text="/自拍")])

        self.assertFalse(command_filter.filter(event, {"wake_prefix": ["/"]}))

        plain = sys.modules["astrbot.api.message_components"].Plain("/自拍")
        event = _GroupCommandEvent([plain])
        self.assertTrue(command_filter.filter(event, {"wake_prefix": ["/"]}))

    async def test_direct_selfie_skips_contextual_llm_and_sends_image(self):
        plugin = LifeCompanionImagePlugin.__new__(LifeCompanionImagePlugin)
        plugin.config = {
            "features": {
                "selfie": {
                    "enabled": True,
                    "default_output": "4K",
                    "default_aspect_ratio": "16:9",
                    "gitee_task_types": ["id"],
                }
            }
        }
        plugin._prepare_prompt = AsyncMock(
            return_value=("包含今日日程的提示词", "16:9 4K", {})
        )
        plugin._reference_images = AsyncMock(return_value=[b"reference"])
        plugin._contextual_wait_message = AsyncMock()
        plugin._send_image = AsyncMock()
        output_path = Path("/tmp/life-companion-test.png")
        plugin.client = types.SimpleNamespace(
            edit=AsyncMock(return_value=output_path)
        )

        event = _CommandEvent("/自拍")
        await plugin._selfie(event, "", announce=False)

        plugin._contextual_wait_message.assert_not_awaited()
        plugin.client.edit.assert_awaited_once()
        self.assertEqual(plugin.client.edit.await_args.kwargs["size"], "16:9 4K")
        plugin._send_image.assert_awaited_once_with(event, output_path)
    @staticmethod
    def _provider_plugin():
        plugin = LifeCompanionImagePlugin.__new__(LifeCompanionImagePlugin)
        plugin.config = {
            "features": {
                "draw": {
                    "chain": [
                        {"__template_key": "provider", "provider_id": "old"},
                        {"__template_key": "provider", "provider_id": "backup"},
                    ]
                },
                "selfie": {
                    "chain": [{"__template_key": "provider", "provider_id": "old"}]
                },
                "edit": {
                    "chain": [{"__template_key": "provider", "provider_id": "old"}]
                },
            },
            "providers": [
                {
                    "id": "old",
                    "label": "旧服务",
                    "__template_key": "gemini_native",
                    "model": "same-model",
                },
                {
                    "id": "new",
                    "label": "新服务",
                    "__template_key": "gemini_native",
                    "model": "same-model",
                },
            ],
        }
        plugin._send_text = AsyncMock()
        return plugin

    async def test_legacy_aiimg_tool_alias_preserves_old_modes(self):
        plugin = LifeCompanionImagePlugin.__new__(LifeCompanionImagePlugin)
        plugin._draw = AsyncMock(return_value=None)
        plugin._edit = AsyncMock(return_value=None)
        plugin._selfie = AsyncMock(return_value=None)
        event = types.SimpleNamespace()

        result = await plugin.aiimg_generate(
            event,
            prompt="咖啡馆里的生活照",
            mode="text",
            aspect_ratio="16:9",
            resolution="4K",
        )

        plugin._draw.assert_awaited_once_with(
            event, "咖啡馆里的生活照 16:9 4K", notify=False
        )
        self.assertIn("图片请求已提交", result)

        await plugin.aiimg_generate(event, prompt="换成晴天", mode="edit")
        plugin._edit.assert_awaited_once_with(event, "换成晴天", notify=False)

        await plugin.aiimg_generate(event, prompt="发张自拍", mode="selfie_ref")
        plugin._selfie.assert_awaited_once_with(event, "发张自拍", notify=False)

    async def test_explicit_selfie_command_bypasses_llm_tool_round_trip(self):
        plugin = LifeCompanionImagePlugin.__new__(LifeCompanionImagePlugin)
        plugin._selfie = AsyncMock(return_value=None)
        event = _CommandEvent("/自拍 来个自然生活照")

        await plugin.selfie_command(event)

        plugin._selfie.assert_awaited_once_with(
            event, "来个自然生活照", announce=False
        )
        self.assertTrue(event.call_llm)
        self.assertTrue(event.stopped)

    async def test_life_tool_auto_detects_selfie_from_original_user_message(self):
        plugin = LifeCompanionImagePlugin.__new__(LifeCompanionImagePlugin)
        plugin._draw = AsyncMock(return_value=None)
        plugin._edit = AsyncMock(return_value=None)
        plugin._selfie = AsyncMock(return_value=None)
        event = _SendEvent()
        event.message_str = "看看你"

        result = await plugin.life_companion_image(event)

        plugin._selfie.assert_awaited_once_with(event, "", notify=False)
        plugin._draw.assert_not_awaited()
        plugin._edit.assert_not_awaited()
        self.assertIn("图片请求已提交", result)
        self.assertEqual(event.sent, [])

    async def test_life_tool_auto_detects_follow_up_selfie_from_provider_context(self):
        plugin = LifeCompanionImagePlugin.__new__(LifeCompanionImagePlugin)
        plugin._draw = AsyncMock(return_value=None)
        plugin._edit = AsyncMock(return_value=None)
        plugin._selfie = AsyncMock(return_value=None)

        class _ContextEvent(_SendEvent):
            def __init__(self):
                super().__init__()
                self.message_str = ""

            def get_extra(self, key, default=None):
                if key == "provider_request":
                    return types.SimpleNamespace(
                        prompt="看一下嘛，拜托拜托",
                        contexts=[
                            {"role": "assistant", "content": "刚才不是才给你看过了？没空陪你自拍连击。"},
                        ]
                    )
                return default

        event = _ContextEvent()

        result = await plugin.life_companion_image(event)

        plugin._selfie.assert_awaited_once_with(event, "", notify=False)
        plugin._draw.assert_not_awaited()
        plugin._edit.assert_not_awaited()
        self.assertIn("图片请求已提交", result)

    async def test_life_tool_auto_detects_colloquial_single_photo_request(self):
        plugin = LifeCompanionImagePlugin.__new__(LifeCompanionImagePlugin)
        plugin._draw = AsyncMock(return_value=None)
        plugin._edit = AsyncMock(return_value=None)
        plugin._selfie = AsyncMock(return_value=None)
        event = _SendEvent()
        event.message_str = "就拍一张，求你了"

        result = await plugin.life_companion_image(event)

        plugin._selfie.assert_awaited_once_with(event, "", notify=False)
        plugin._draw.assert_not_awaited()
        plugin._edit.assert_not_awaited()
        self.assertIn("图片请求已提交", result)

    def test_selfie_request_accepts_colloquial_phrases_without_matching_subjects(self):
        plugin = LifeCompanionImagePlugin.__new__(LifeCompanionImagePlugin)

        for request in ("拍一张", "给我拍一张", "再拍一张", "拍一下", "就拍一张，求你了"):
            with self.subTest(request=request):
                self.assertTrue(plugin._is_selfie_request(request))

        self.assertFalse(plugin._is_selfie_request("拍一张猫"))

    def test_auto_selfie_does_not_reuse_old_selfie_context_for_new_draw_prompt(self):
        plugin = LifeCompanionImagePlugin.__new__(LifeCompanionImagePlugin)

        class _ContextEvent:
            message_str = ""

            def get_extra(self, key, default=None):
                if key == "provider_request":
                    return types.SimpleNamespace(
                        prompt="画一只猫",
                        contexts=[{"role": "assistant", "content": "刚才陪你自拍过了。"}],
                    )
                return default

        self.assertFalse(plugin._auto_selfie_request(_ContextEvent(), ""))

    async def test_empty_draw_prompt_includes_schedule_and_timeline(self):
        plugin = LifeCompanionImagePlugin.__new__(LifeCompanionImagePlugin)
        plugin.config = {"features": {"draw": {"default_output": "16:9 4K"}}}
        plugin._life_context = AsyncMock(
            return_value={
                "image_prompt": "窗边的自然生活照",
                "outfit": "浅色针织衫",
                "schedule": "下午在咖啡馆阅读",
                "timeline": [{"time": "15:00", "activity": "在咖啡馆阅读"}],
            }
        )

        prompt, _, _ = await plugin._prepare_prompt("", operation="draw")

        self.assertIn("窗边的自然生活照", prompt)
        self.assertIn("今日日程：下午在咖啡馆阅读", prompt)
        self.assertIn("15:00 在咖啡馆阅读", prompt)

    async def test_life_tool_returns_missing_reference_without_sending_internal_message(self):
        plugin = LifeCompanionImagePlugin.__new__(LifeCompanionImagePlugin)
        missing_reference = (
            "还没有自拍参考照。请发送一张图片并使用：/生活参考照 设置；"
            "也可以在同一条消息附图后直接使用 /生活自拍。"
        )
        plugin._draw = AsyncMock(return_value=None)
        plugin._edit = AsyncMock(return_value=None)
        plugin._selfie = AsyncMock(return_value=missing_reference)
        event = _SendEvent()
        event.message_str = "自拍"

        result = await plugin.life_companion_image(event)

        self.assertIn("图片请求未执行", result)
        self.assertIn("人物参考图", result)
        self.assertNotIn("/生活参考照", result)
        self.assertEqual(event.sent, [])

    async def test_legacy_aiimg_auto_detects_selfie_from_original_user_message(self):
        plugin = LifeCompanionImagePlugin.__new__(LifeCompanionImagePlugin)
        plugin._draw = AsyncMock(return_value=None)
        plugin._edit = AsyncMock(return_value=None)
        plugin._selfie = AsyncMock(return_value=None)
        event = _SendEvent()
        event.message_str = "/自拍"

        with patch("astrbot_plugin_life_companion_image.main.logger") as log:
            result = await plugin.aiimg_generate(event)

        plugin._selfie.assert_awaited_once_with(event, "", notify=False)
        plugin._draw.assert_not_awaited()
        plugin._edit.assert_not_awaited()
        self.assertIn("图片请求已提交", result)
        self.assertTrue(
            any(
                "LLM工具 aiimg_generate 被调用" in call.args[0]
                for call in log.info.call_args_list
            )
        )
        self.assertTrue(
            any(
                "LLM工具已提交图片请求" in call.args[0]
                for call in log.info.call_args_list
            )
        )

    async def test_selfie_preflight_does_not_send_missing_reference_to_user(self):
        plugin = LifeCompanionImagePlugin.__new__(LifeCompanionImagePlugin)
        plugin.config = {"features": {"selfie": {"enabled": True}}}
        plugin._life_context = AsyncMock(return_value={})
        plugin._reference_images = AsyncMock(return_value=[])
        event = _SendEvent()
        event.message_str = "/自拍"

        result = await plugin.life_companion_image(event, mode="selfie")

        self.assertIn("图片请求未执行", result)
        self.assertIn("人物参考图", result)
        self.assertNotIn("/生活参考照", result)
        self.assertEqual(event.sent, [])

    async def test_explicit_selfie_command_keeps_missing_reference_message_user_visible(self):
        plugin = LifeCompanionImagePlugin.__new__(LifeCompanionImagePlugin)
        plugin._selfie = AsyncMock()
        event = _CommandEvent("/生活自拍")

        await plugin.life_selfie(event)

        plugin._selfie.assert_awaited_once_with(event, "", announce=False)
        self.assertTrue(event.call_llm)
        self.assertTrue(event.stopped)

    async def test_life_context_uses_explicit_read_only_flag(self):
        life_plugin = _LifePlugin()
        plugin = LifeCompanionImagePlugin.__new__(LifeCompanionImagePlugin)
        plugin.context = _Context(life_plugin)
        plugin.config = {"use_life_companion": True}

        result = await plugin._life_context(allow_generate=False)

        self.assertEqual(result["outfit"], "测试穿搭")
        self.assertEqual(life_plugin.calls, [False])

    async def test_prompt_log_marks_schedule_and_timeline_as_injected(self):
        plugin = LifeCompanionImagePlugin.__new__(LifeCompanionImagePlugin)
        plugin.config = {
            "features": {
                "draw": {"default_output": "1024x1024"},
                "selfie": {"default_output": "1K", "default_aspect_ratio": "3:4"},
            }
        }
        plugin._life_context = AsyncMock(
            return_value={
                "date": "2026-09-05",
                "outfit": "测试穿搭",
                "schedule": "下午在咖啡馆阅读",
                "timeline": [{"time": "15:00", "activity": "在咖啡馆阅读"}],
            }
        )

        with patch("astrbot_plugin_life_companion_image.main.logger") as log:
            await plugin._prepare_prompt(
                "自然生活照", selfie=True, operation="selfie"
            )

        self.assertTrue(
            any(
                "生活状态已响应给图片提示词" in call.args[0]
                and call.args[1:] == ("selfie", "有", 1, "是", "是")
                for call in log.info.call_args_list
            )
        )

    async def test_event_images_includes_images_from_reply_chain(self):
        image_type = sys.modules["astrbot.api.message_components"].Image
        reply_type = sys.modules["astrbot.api.message_components"].Reply
        plugin = LifeCompanionImagePlugin.__new__(LifeCompanionImagePlugin)
        image = image_type("aW1hZ2U=")
        try:
            reply = reply_type(id="test", chain=[image])
        except TypeError:
            reply = reply_type([image])
        event = types.SimpleNamespace(
            message_obj=types.SimpleNamespace(
                message=[reply]
            )
        )

        result = await plugin._event_images(event)

        self.assertEqual(result, [b"image"])

    async def test_wait_message_uses_current_provider_context_and_filters_technical_text(self):
        plugin = LifeCompanionImagePlugin.__new__(LifeCompanionImagePlugin)
        plugin.context = _LLMContext("正在生成图片，请稍候，已经好了")
        event = _SendEvent()

        result = await plugin._contextual_wait_message(event, "selfie", "在窗边喝咖啡")

        self.assertEqual(result, "")
        self.assertEqual(plugin.context.calls[0]["chat_provider_id"], "chat-provider")
        self.assertEqual(plugin.context.calls[0]["contexts"][0]["role"], "user")
        self.assertIsNone(plugin.context.calls[0]["tools"])
        self.assertIn("她说话自然、亲近。", plugin.context.calls[0]["system_prompt"])
        self.assertIn("不要套用固定句式", plugin.context.calls[0]["system_prompt"])

    async def test_image_starts_during_wait_message_and_sends_result_after_notice(self):
        plugin = LifeCompanionImagePlugin.__new__(LifeCompanionImagePlugin)
        event = _SendEvent()
        image_started = False
        notice_sent = False

        async def image_operation():
            nonlocal image_started
            image_started = True
            return Path("/tmp/generated-test.jpg")

        async def wait_message(*args):
            await asyncio.sleep(0.01)
            self.assertTrue(image_started)
            return "你等等呀，我给你找个好看的角度。"

        async def send_text(*args):
            nonlocal notice_sent
            notice_sent = True

        plugin._contextual_wait_message = wait_message
        plugin._send_text = send_text
        plugin._send_image = AsyncMock()

        await plugin._generate_with_notice(
            event, "selfie", "窗边喝咖啡", image_operation()
        )

        self.assertTrue(notice_sent)
        plugin._send_image.assert_awaited_once_with(
            event, Path("/tmp/generated-test.jpg")
        )

    async def test_image_does_not_send_text_when_dynamic_notice_is_unavailable(self):
        plugin = LifeCompanionImagePlugin.__new__(LifeCompanionImagePlugin)
        event = _SendEvent()
        plugin._contextual_wait_message = AsyncMock(return_value="")
        plugin._send_text = AsyncMock()
        plugin._send_image = AsyncMock()

        await plugin._generate_with_notice(
            event,
            "selfie",
            "窗边喝咖啡",
            asyncio.sleep(0, result=Path("/tmp/generated-test.jpg")),
        )

        plugin._send_text.assert_not_awaited()
        plugin._send_image.assert_awaited_once_with(
            event, Path("/tmp/generated-test.jpg")
        )

    def test_reference_selection_keeps_original_bytes_and_limits_request_budget(self):
        plugin = LifeCompanionImagePlugin.__new__(LifeCompanionImagePlugin)
        first = b"1234567"
        skipped = b"skip!"
        later = b"abc"

        with patch.object(LifeCompanionImagePlugin, "_REFERENCE_REQUEST_BUDGET", 10):
            selected = plugin._select_reference_images([first, skipped, later])

        self.assertEqual(selected, [first, later])

    async def test_image_models_lists_provider_names_and_models(self):
        plugin = self._provider_plugin()

        await plugin.image_models(types.SimpleNamespace(message_str="/生图模型"))

        text = plugin._send_text.await_args.args[1]
        self.assertIn("旧服务（old）：same-model", text)
        self.assertIn("新服务（new）：same-model", text)
        self.assertIn("当前首选服务商", text)

    async def test_switch_all_replaces_current_provider_and_keeps_other_fallbacks(self):
        plugin = self._provider_plugin()

        await plugin.switch_image_provider(
            types.SimpleNamespace(message_str="/切换生图 新服务")
        )

        self.assertEqual(
            plugin.config["features"]["draw"]["chain"],
            [
                {"__template_key": "provider", "provider_id": "new"},
                {"__template_key": "provider", "provider_id": "backup"},
            ],
        )
        self.assertEqual(
            plugin.config["features"]["selfie"]["chain"],
            [
                {"__template_key": "provider", "provider_id": "new"},
            ],
        )
        self.assertEqual(
            plugin.config["features"]["edit"]["chain"],
            [
                {"__template_key": "provider", "provider_id": "new"},
            ],
        )
        message = plugin._send_text.await_args.args[1]
        self.assertIn("文生图、自拍、改图", message)
        self.assertIn("原首选服务商已从链路移除", message)

    async def test_switch_single_operation_changes_only_requested_chain(self):
        plugin = self._provider_plugin()

        await plugin.switch_image_provider(
            types.SimpleNamespace(message_str="/切换生图 文生图 新服务")
        )

        self.assertEqual(
            plugin.config["features"]["draw"]["chain"][0],
            {"__template_key": "provider", "provider_id": "new"},
        )
        self.assertEqual(
            plugin.config["features"]["selfie"]["chain"][0],
            {"__template_key": "provider", "provider_id": "old"},
        )
        self.assertEqual(
            plugin.config["features"]["edit"]["chain"][0],
            {"__template_key": "provider", "provider_id": "old"},
        )

    async def test_switch_rejects_unsupported_provider_without_partial_change(self):
        plugin = self._provider_plugin()
        plugin.config["providers"].append(
            {
                "id": "draw-only",
                "label": "仅文生图",
                "__template_key": "gitee_images",
                "model": "draw-model",
            }
        )

        await plugin.switch_image_provider(
            types.SimpleNamespace(message_str="/切换生图 仅文生图")
        )

        self.assertEqual(
            plugin.config["features"]["draw"]["chain"][0],
            {"__template_key": "provider", "provider_id": "old"},
        )
        self.assertIn("不支持", plugin._send_text.await_args.args[1])

    async def test_switch_does_not_treat_model_name_as_provider_name(self):
        plugin = self._provider_plugin()

        await plugin.switch_image_provider(
            types.SimpleNamespace(message_str="/切换生图 same-model")
        )

        self.assertEqual(
            plugin.config["features"]["draw"]["chain"][0],
            {"__template_key": "provider", "provider_id": "old"},
        )
        self.assertIn("没有找到服务商", plugin._send_text.await_args.args[1])
