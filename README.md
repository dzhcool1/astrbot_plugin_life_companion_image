# Life Companion Image

一个与 `astrbot_plugin_gitee_aiimg` 使用方式相近、但可以独立安装的 AstrBot 生图插件。它直接调用 Gitee AI Images API，并和 `astrbot_plugin_life_companion` 共享每日穿搭、日程、时间线及生活照提示词。

## 功能

- Gitee AI 文生图：`/生活照 <提示词>`，支持命令末尾追加 `1:1`、`16:9`、`9:16` 等比例。
- 生活自拍：`/生活自拍 [补充要求]`，使用参考人像和 Life Companion 状态调用 Gitee 异步改图接口。
- 图片修改：发送图片并使用 `/生活改图 <修改要求>`。
- 参考照管理：发送图片并使用 `/生活参考照 设置`；查看或删除使用 `/生活参考照 查看`、`/生活参考照 删除`。
- LLM 工具：`life_companion_image`，`mode` 可选 `life_photo` 或 `selfie`。
- API Key 池轮询、Base64/URL 图片结果、结果本地缓存，以及图片消息失败时的文件发送回退。

## 安装

将本目录放到 AstrBot 的 `data/plugins/astrbot_plugin_life_companion_image`，安装依赖并重启：

```bash
pip install -r requirements.txt
```

如果同时安装 `astrbot_plugin_life_companion`，请确保两个插件目录并列。Life Companion 的 `今日生活照` 会优先调用本插件；没有本插件时仍会尝试兼容旧的 `astrbot_plugin_gitee_aiimg`。图片插件读取的是 Life Companion 已缓存的状态，不会因为准备图片提示词而额外触发 LLM 日程生成；需要生成当天状态时可先使用 `查看日程` 或 `今日生活照`。

## 配置

在 WebUI 配置以下项目：

| 配置 | 默认值 | 说明 |
| --- | --- | --- |
| `api_base_url` | `https://ai.gitee.com/v1` | Gitee AI Base URL |
| `api_keys` | `[]` | 一个或多个 Gitee AI API Key |
| `model` | `z-image-turbo` | 文生图模型 |
| `default_size` | `1024x1024` | Gitee 白名单尺寸 |
| `edit_model` | `Qwen-Image-Edit-2511` | 自拍/改图模型 |
| `use_life_companion` | `true` | 是否读取 Life Companion 的缓存状态 |

参考照建议使用命令保存。配置里的 `reference_images` 只接受插件数据目录内的路径，插件会拒绝数据目录外的路径。

## API 说明

文生图请求：`POST {api_base_url}/images/generations`。

自拍/改图请求：`POST {api_base_url}/async/images/edits`，随后轮询 `GET {api_base_url}/task/{task_id}`。

API Key、模型和接口可用性由 Gitee AI 账户及服务端决定；插件不会内置或共享任何 Key。

## 开发检查

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
python -m compileall -q .
```
