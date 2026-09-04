# Life Companion Image

一个与 `astrbot_plugin_gitee_aiimg` 使用方式相近、但可以独立安装的 AstrBot 生图插件。它直接调用 Gitee AI Images API，并和 `astrbot_plugin_life_companion` 共享每日穿搭、日程、时间线及生活照提示词。

更新记录见 [CHANGELOG.md](CHANGELOG.md)。

## 功能

- Gitee AI 文生图：`/生活照 <提示词>`，支持命令末尾追加 `1:1`、`16:9`、`9:16` 等比例。
- 生活自拍：`/生活自拍 [补充要求]`，使用参考人像和 Life Companion 状态调用 Gitee 异步改图接口。
- 图片修改：发送图片并使用 `/生活改图 <修改要求>`。
- 参考照管理：发送图片并使用 `/生活参考照 设置`；查看或删除使用 `/生活参考照 查看`、`/生活参考照 删除`。
- 服务商查看与切换：`/生图模型` 查看已配置的服务商和模型；管理员使用 `/切换生图 服务商名称` 切换全部类型，或使用 `/切换生图 自拍/文生图/改图 服务商名称` 切换单一类型。
- LLM 工具：`life_companion_image`，`mode` 支持 `auto`、`life_photo`、`selfie` 和 `edit`；未传模式时会结合用户原话和附图自动选择；兼容旧 Gitee 工具名 `aiimg_generate`。
- API Key 池轮询、Base64/URL 图片结果、结果本地缓存，以及图片消息失败时的文件发送回退。

## 安装

将本目录放到 AstrBot 的 `data/plugins/astrbot_plugin_life_companion_image`，安装依赖并重启：

```bash
pip install -r requirements.txt
```

如果同时安装 `astrbot_plugin_life_companion`，请确保两个插件目录并列。Life Companion 的 `今日生活照` 会调用本插件；图片插件只读取 Life Companion 已缓存的状态，不会读取旧生图插件配置，也不会因为准备图片提示词而额外触发 LLM 日程生成；需要生成当天状态时可先使用 `查看日程` 或 `今日生活照`。

## 配置

WebUI 配置结构与 `astrbot_plugin_gitee_aiimg v5.1.30` 对齐，包含 `features`、`storage`、`image_encoding`、`send`、`network`、`providers`、并发和防抖等全部配置项。服务商链路按 `features.draw.chain`、`features.edit.chain` 和 `features.selfie.chain` 的顺序尝试，服务商参数集中放在 `providers` 中。

使用 `/生图模型` 可查看每个已配置服务商的显示名称和模型。切换命令只接受服务商的 `label` 或 `id`，不会把模型名称当作切换参数；不带类型时同时替换文生图、自拍和改图的首选服务商，并从对应链路移除原首选服务商，其它已存在的链路项继续作为备用。自拍和改图只允许切换到支持改图的服务商，全量切换还要求服务商同时支持文生图。

生活插件实际使用文生图、改图和自拍链路；当前支持 Gitee Images、Gitee Async、Gemini 原生和 OpenAI Images 兼容服务商。复制其它服务商配置不会丢失设置，但生活插件不会为未支持的服务商伪造兼容性。

Life Companion 专用兼容项为 `use_life_companion`、`reference_images` 和 `selfie_prompt_prefix`。推荐使用 `features.selfie.reference_images` 管理参考照，或通过 `/生活参考照 设置`、`查看`、`删除` 管理。参考照路径只接受插件数据目录内的路径，插件会拒绝数据目录外的路径。

`network.max_image_bytes` 默认是 52428800（50 MiB），用于限制下载或返回的单张图片大小。比例与分辨率组合（例如 `3:4 4K`）会在请求前转换成接口接受的精确尺寸（`3072x4096`）。

## API 说明

文生图请求：`POST {api_base_url}/images/generations`。

自拍/改图请求：`POST {api_base_url}/async/images/edits`，随后轮询 `GET {api_base_url}/task/{task_id}`。

API Key、模型和接口可用性由 Gitee AI 账户及服务端决定；插件不会内置或共享任何 Key。

## 开发检查

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
python -m compileall -q .
```
