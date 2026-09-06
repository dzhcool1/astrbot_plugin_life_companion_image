import base64
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from core.gitee_client import GiteeAIClient, GiteeAPIError
from core.prompting import build_selfie_prompt, normalize_output_size, split_size_suffix


class FakeResponse:
    def __init__(self, status_code=200, payload=None, content=b"", headers=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.content = content
        self.headers = headers or {}

    def json(self):
        return self._payload


class FakeHTTPClient:
    def __init__(self, responses, download=None):
        self.responses = list(responses)
        self.download = download
        self.requests = []

    async def request(self, method, url, **kwargs):
        self.requests.append((method, url, kwargs))
        return self.responses.pop(0)

    async def get(self, url, **kwargs):
        self.requests.append(("GET", url, kwargs))
        if self.download is not None:
            return self.download
        return self.responses.pop(0)

    async def post(self, url, **kwargs):
        return await self.request("POST", url, **kwargs)


class PromptingTest(unittest.TestCase):
    def test_ratio_suffix_uses_gitee_whitelist_size(self):
        prompt, size = split_size_suffix("阳台下午茶 16:9", "1024x1024")
        self.assertEqual(prompt, "阳台下午茶")
        self.assertEqual(size, "1024x576")

    def test_ratio_suffix_maps_all_gitee_landscape_sizes(self):
        self.assertEqual(split_size_suffix("测试 4:3", "1024x1024")[1], "1152x896")
        self.assertEqual(split_size_suffix("测试 3:2", "1024x1024")[1], "2048x1360")

    def test_combined_ratio_and_resolution_is_an_exact_size(self):
        self.assertEqual(normalize_output_size("3:4 4K"), "3072x4096")
        self.assertEqual(normalize_output_size("16:9 4K"), "4096x2304")
        self.assertEqual(split_size_suffix("测试 3:4 4K", "1024x1024")[1], "3072x4096")
        self.assertEqual(normalize_output_size("auto"), "auto")
        self.assertEqual(normalize_output_size("1024x1024"), "1024x1024")

    def test_selfie_prompt_keeps_user_request_above_life_defaults(self):
        result = build_selfie_prompt(
            "用户要求红色雨衣，不要裙子",
            {
                "outfit": "蓝色学院风裙装",
                "schedule": "下午在咖啡店阅读",
                "timeline": [{"time": "15:00", "activity": "阅读"}],
            },
            "保持参考图人物身份一致",
        )
        self.assertIn("红色雨衣，不要裙子", result)
        self.assertIn("蓝色学院风裙装", result)
        self.assertIn("15:00 阅读", result)
        self.assertIn("用户要求（最高优先级）", result)

    def test_selfie_prompt_excludes_capture_devices_from_all_prompt_sources(self):
        result = build_selfie_prompt(
            "",
            {
                "outfit": "白衬衫，手持智能手机",
                "schedule": "用相机记录街景",
                "timeline": [
                    {"time": "12:00", "activity": "看着手机屏幕"},
                ],
                "image_prompt": "自然自拍",
            },
            "固定手机入镜",
        )

        prompt_without_policy = result.split("\n\n拍摄方式规则", 1)[0]
        for device in ("手机", "相机", "摄像头", "自拍杆", "屏幕"):
            self.assertNotIn(device, prompt_without_policy)
        self.assertIn("用户要求（最高优先级）：由他人拍摄的生活照", prompt_without_policy)

        explicit = build_selfie_prompt("对镜自拍，手机自然入镜", {}, "")
        self.assertIn("用户要求（最高优先级）：由他人拍摄的生活照，自然入镜", explicit)
        self.assertNotIn("手机自然入镜", explicit)
        self.assertIn("由他人从画面外拍摄的自然生活照，而不是自拍", explicit)
        self.assertIn("不做伸手举手机、对镜看屏幕或持拍摄设备的动作", explicit)


class GiteeClientTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.config = {
            "api_base_url": "https://example.test/v1",
            "api_keys": ["key-a", "key-b"],
            "model": "z-image-turbo",
            "default_size": "1024x1024",
            "num_inference_steps": 9,
            "poll_interval": 0,
            "poll_timeout": 2,
        }

    def tearDown(self):
        self.tmp.cleanup()

    async def test_generate_decodes_base64_and_rotates_key(self):
        image = b"\x89PNG\r\nimage"
        fake = FakeHTTPClient(
            [FakeResponse(payload={"data": [{"b64_json": base64.b64encode(image).decode()}]})]
        )
        client = GiteeAIClient(self.config, Path(self.tmp.name), http_client=fake)

        result = await client.generate("一只猫", size="1024x1024")

        self.assertEqual(result.read_bytes(), image)
        self.assertEqual(result.suffix, ".png")
        self.assertEqual(fake.requests[0][0:2], ("POST", "https://example.test/v1/images/generations"))
        self.assertEqual(fake.requests[0][2]["headers"]["Authorization"], "Bearer key-a")
        self.assertEqual(fake.requests[0][2]["json"]["prompt"], "一只猫")

    async def test_base_url_without_v1_is_normalized(self):
        client = GiteeAIClient(
            {**self.config, "api_base_url": "https://example.test"},
            Path(self.tmp.name),
            http_client=FakeHTTPClient(
                [FakeResponse(payload={"data": [{"b64_json": base64.b64encode(b"jpg").decode()}]})]
            ),
        )
        await client.generate("测试")
        self.assertEqual(client._base_url(), "https://example.test/v1")

    async def test_generate_logs_model_prompt_and_never_logs_api_key(self):
        image = b"generated-image"
        config = {
            "features": {
                "draw": {
                    "default_output": "16:9 4K",
                    "chain": [{"provider_id": "mengyu"}],
                }
            },
            "providers": [
                {
                    "id": "mengyu",
                    "__template_key": "openai_images",
                    "base_url": "https://example.test",
                    "api_keys": ["secret-key"],
                    "model": "gpt-image-test",
                }
            ],
        }
        fake = FakeHTTPClient(
            [
                FakeResponse(
                    payload={
                        "data": [{"b64_json": base64.b64encode(image).decode()}]
                    }
                )
            ]
        )
        client = GiteeAIClient(config, Path(self.tmp.name), http_client=fake)

        with patch("core.gitee_client.logger") as logger_mock:
            await client.generate("夜晚街角的咖啡馆", size="16:9 4K")

        messages = "\n".join(repr(call) for call in logger_mock.info.call_args_list)
        self.assertIn("gpt-image-test", messages)
        self.assertIn("夜晚街角的咖啡馆", messages)
        self.assertIn("16:9 4K", messages)
        self.assertNotIn("secret-key", messages)

    async def test_edit_logs_model_prompt_and_reference_summary(self):
        image = b"\x89PNG\r\nreference"
        generated = b"\x89PNG\r\ngenerated"
        config = {
            "features": {
                "selfie": {
                    "default_output": "4K",
                    "chain": [{"provider_id": "meinianda"}],
                }
            },
            "providers": [
                {
                    "id": "meinianda",
                    "__template_key": "gemini_native",
                    "api_url": "https://example.test",
                    "api_keys": ["secret-key"],
                    "model": "gemini-image-test",
                    "max_retries": 0,
                }
            ],
        }
        fake = FakeHTTPClient(
            [
                FakeResponse(
                    payload={
                        "candidates": [
                            {
                                "content": {
                                    "parts": [
                                        {
                                            "inlineData": {
                                                "mimeType": "image/png",
                                                "data": base64.b64encode(generated).decode(),
                                            }
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                )
            ]
        )
        client = GiteeAIClient(config, Path(self.tmp.name), http_client=fake)

        with patch("core.gitee_client.logger") as logger_mock:
            await client.edit("窗边自然入镜", [image], operation="selfie")

        start_call = next(
            call
            for call in logger_mock.info.call_args_list
            if "%s开始" in str(call.args[0])
        )
        self.assertIn("gemini-image-test", start_call.args)
        self.assertIn("窗边自然入镜", start_call.args)
        self.assertEqual(start_call.args[5], 1)
        messages = "\n".join(repr(call) for call in logger_mock.info.call_args_list)
        self.assertNotIn("secret-key", messages)

    async def test_edit_polls_until_success_and_downloads_url(self):
        image = b"\xff\xd8\xffjpeg"
        fake = FakeHTTPClient(
            [
                FakeResponse(payload={"task_id": "task-1"}),
                FakeResponse(payload={"status": "processing"}),
                FakeResponse(payload={"status": "success", "output": {"file_url": "https://cdn.test/a.jpg"}}),
            ],
            download=FakeResponse(content=image, headers={"content-type": "image/jpeg"}),
        )
        client = GiteeAIClient(self.config, Path(self.tmp.name), http_client=fake)

        result = await client.edit("换成晴天", [image], task_types=["id"])

        self.assertEqual(result.read_bytes(), image)
        self.assertEqual(result.suffix, ".jpg")
        data = fake.requests[0][2]["data"]
        self.assertIn(("task_types", "id"), data)
        self.assertNotIn(("task_types", "invalid"), data)
        self.assertEqual(len(fake.requests), 4)
        self.assertNotIn("Authorization", fake.requests[-1][2].get("headers", {}))

    async def test_http_error_is_user_safe(self):
        fake = FakeHTTPClient([FakeResponse(401, {"message": "invalid token"})])
        client = GiteeAIClient(self.config, Path(self.tmp.name), http_client=fake)

        with self.assertRaisesRegex(GiteeAPIError, "invalid token"):
            await client.generate("测试")

    async def test_missing_key_fails_before_network_request(self):
        fake = FakeHTTPClient([])
        config = {**self.config, "api_keys": []}
        client = GiteeAIClient(config, Path(self.tmp.name), http_client=fake)

        with self.assertRaisesRegex(GiteeAPIError, "API Key"):
            await client.generate("测试")
        self.assertEqual(fake.requests, [])

    async def test_v5_provider_config_uses_draw_chain_and_network_limit(self):
        config = {
            "features": {
                "draw": {
                    "default_output": "16:9 4K",
                    "chain": [{"provider_id": "meinianda"}],
                }
            },
            "network": {"max_image_bytes": 50 * 1024 * 1024},
            "providers": [
                {
                    "id": "meinianda",
                    "__template_key": "gemini_native",
                    "api_url": "https://example.test",
                    "api_keys": ["key-a"],
                    "model": "gemini-test",
                }
            ],
        }
        client = GiteeAIClient(config, Path(self.tmp.name))

        candidates = client._provider_candidates("draw")

        self.assertEqual(candidates[0]["id"], "meinianda")
        self.assertEqual(candidates[0]["template"], "gemini_native")
        self.assertEqual(client._max_image_bytes(), 50 * 1024 * 1024)

    async def test_generate_uses_first_provider_in_configured_chain(self):
        image = b"generated-by-mengyu"
        fake = FakeHTTPClient(
            [
                FakeResponse(
                    payload={
                        "data": [
                            {"b64_json": base64.b64encode(image).decode()}
                        ]
                    }
                )
            ]
        )
        config = {
            "features": {
                "draw": {
                    "chain": [
                        {"provider_id": "mengyu"},
                        {"provider_id": "meinianda"},
                    ]
                }
            },
            "providers": [
                {
                    "id": "mengyu",
                    "__template_key": "openai_images",
                    "base_url": "https://ai.zhicloud.top",
                    "api_keys": ["mengyu-key"],
                    "model": "gpt-image-2",
                },
                {
                    "id": "meinianda",
                    "__template_key": "gemini_native",
                    "api_url": "https://meinianda.top",
                    "api_keys": ["meinianda-key"],
                    "model": "gemini-test",
                },
            ],
        }
        client = GiteeAIClient(config, Path(self.tmp.name), http_client=fake)

        result = await client.generate("测试场景", size="3:4 4K")

        self.assertEqual(result.read_bytes(), image)
        self.assertEqual(
            fake.requests[0][1],
            "https://ai.zhicloud.top/v1/images/generations",
        )
        self.assertEqual(fake.requests[0][2]["json"]["model"], "gpt-image-2")
        self.assertEqual(fake.requests[0][2]["json"]["size"], "3072x4096")

    async def test_gemini_provider_uses_copied_chain_output_settings(self):
        image = b"\x89PNG\r\nimage"
        fake = FakeHTTPClient(
            [
                FakeResponse(
                    payload={
                        "candidates": [
                            {
                                "content": {
                                    "parts": [
                                        {
                                            "inlineData": {
                                                "mimeType": "image/png",
                                                "data": base64.b64encode(image).decode(),
                                            }
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                )
            ]
        )
        config = {
            "features": {
                "draw": {
                    "default_output": "16:9 4K",
                    "chain": [{"provider_id": "meinianda"}],
                }
            },
            "providers": [
                {
                    "id": "meinianda",
                    "__template_key": "gemini_native",
                    "api_url": "https://example.test",
                    "api_keys": ["key-a"],
                    "model": "gemini-test",
                    "max_retries": 0,
                }
            ],
        }
        client = GiteeAIClient(config, Path(self.tmp.name), http_client=fake)

        result = await client.generate("测试场景")

        self.assertEqual(result.read_bytes(), image)
        method, url, request = fake.requests[0]
        self.assertEqual(method, "POST")
        self.assertEqual(
            url,
            "https://example.test/v1beta/models/gemini-test:generateContent",
        )
        self.assertEqual(request["headers"]["x-goog-api-key"], "key-a")
        self.assertEqual(
            request["json"]["generationConfig"]["imageConfig"],
            {"imageSize": "4K", "aspectRatio": "16:9"},
        )

    async def test_gemini_edit_preserves_large_reference_bytes_and_output_controls(self):
        reference = b"\xff\xd8\xff" + bytes(range(256)) * 5000
        generated = b"\x89PNG\r\ngenerated-image"
        config = {
            "features": {
                "selfie": {
                    "default_output": "4K",
                    "chain": [{"provider_id": "meinianda"}],
                }
            },
            "providers": [
                {
                    "id": "meinianda",
                    "__template_key": "gemini_native",
                    "api_url": "https://example.test",
                    "api_keys": ["key-a"],
                    "model": "gemini-test",
                    "max_retries": 0,
                }
            ],
        }
        fake = FakeHTTPClient(
            [
                FakeResponse(
                    payload={
                        "candidates": [
                            {
                                "content": {
                                    "parts": [
                                        {
                                            "inlineData": {
                                                "mimeType": "image/png",
                                                "data": base64.b64encode(generated).decode(),
                                            }
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                )
            ]
        )
        client = GiteeAIClient(config, Path(self.tmp.name), http_client=fake)

        await client.edit(
            "窗边自然生活照",
            [reference],
            task_types=["id", "background", "style"],
            size="16:9 4K",
            operation="selfie",
        )

        request = fake.requests[0][2]["json"]
        self.assertEqual(
            request["generationConfig"]["imageConfig"],
            {"imageSize": "4K", "aspectRatio": "16:9"},
        )
        inline = request["contents"][0]["parts"][1]["inlineData"]
        self.assertEqual(inline["mimeType"], "image/jpeg")
        self.assertEqual(base64.b64decode(inline["data"]), reference)

    async def test_gemini_read_error_is_not_retried_after_request_body_was_sent(self):
        class ReadErrorClient:
            def __init__(self):
                self.calls = 0

            async def post(self, *args, **kwargs):
                self.calls += 1
                raise httpx.ReadError("upstream closed")

        fake = ReadErrorClient()
        config = {
            "features": {
                "selfie": {
                    "default_output": "4K",
                    "chain": [{"provider_id": "meinianda"}],
                }
            },
            "providers": [
                {
                    "id": "meinianda",
                    "__template_key": "gemini_native",
                    "api_url": "https://example.test",
                    "api_keys": ["key-a"],
                    "model": "gemini-test",
                    "max_retries": 2,
                }
            ],
        }
        client = GiteeAIClient(config, Path(self.tmp.name), http_client=fake)

        with self.assertRaisesRegex(GiteeAPIError, "上游在返回响应前关闭连接"):
            await client.edit(
                "窗边自然生活照",
                [b"reference"],
                operation="selfie",
            )

        self.assertEqual(fake.calls, 1)
