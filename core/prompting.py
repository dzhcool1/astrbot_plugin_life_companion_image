from __future__ import annotations

import re
from typing import Any


RATIO_SIZES = {
    "1:1": "1024x1024",
    "16:9": "1024x576",
    "9:16": "576x1024",
    "4:3": "1152x896",
    "3:4": "768x1024",
    "3:2": "2048x1360",
    "2:3": "1360x2048",
}
_SIZE_SUFFIX_RE = re.compile(
    r"(?:^|\s)(?P<size>(?:\d{3,5}x\d{3,5}|(?:16|9|4|3|2|1):(?:16|9|4|3|2|1)))$",
    re.IGNORECASE,
)


def split_size_suffix(text: str, default_size: str) -> tuple[str, str]:
    """Allow the familiar ``/aiimg prompt 16:9`` command form."""
    value = str(text or "").strip()
    match = _SIZE_SUFFIX_RE.search(value)
    if not match:
        return value, str(default_size or "1024x1024")
    size = match.group("size").lower()
    normalized = RATIO_SIZES.get(size, size)
    prompt = value[: match.start()].strip()
    return prompt, normalized


def build_selfie_prompt(
    user_prompt: str,
    life_context: dict[str, Any] | None,
    prefix: str,
) -> str:
    """Merge life companion data as defaults while preserving user intent."""
    context = life_context if isinstance(life_context, dict) else {}
    outfit = str(context.get("outfit") or "").strip()
    schedule = str(context.get("schedule") or "").strip()
    image_prompt = str(context.get("image_prompt") or "").strip()
    requested = str(user_prompt or "").strip() or image_prompt or "自然真实的生活照"

    lines = [str(prefix or "").strip()]
    if outfit or schedule:
        lines.extend(
            [
                "",
                "今日生活状态（仅作为默认背景；用户本次要求优先）：",
                f"- 今日穿搭：{outfit or '未记录'}",
                f"- 今日日程：{schedule or '未记录'}",
            ]
        )
    timeline = context.get("timeline")
    if isinstance(timeline, list):
        entries = []
        for item in timeline[:8]:
            if not isinstance(item, dict):
                continue
            time_value = str(item.get("time") or "").strip()
            activity = str(item.get("activity") or "").strip()
            if time_value and activity:
                entries.append(f"{time_value} {activity[:120]}")
        if entries:
            lines.extend(["- 今日时间线：", *entries])
    if outfit or schedule or timeline:
        lines.append("用户没有指定其它穿搭或场景时，优先保持上述生活状态一致。")
    lines.extend(["", f"用户要求（最高优先级）：{requested}"])
    return "\n".join(line for line in lines if line is not None).strip()
