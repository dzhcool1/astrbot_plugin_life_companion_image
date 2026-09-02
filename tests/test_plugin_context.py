import sys
import types
import unittest
from pathlib import Path


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


class PluginContextTest(unittest.IsolatedAsyncioTestCase):
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
