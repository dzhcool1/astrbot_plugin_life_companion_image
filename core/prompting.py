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
_RESOLUTION_SQUARE_SIZES = {
    "1k": "1024x1024",
    "2k": "2048x2048",
    "4k": "4096x4096",
}
_RATIO_RESOLUTION_SIZES = {
    ("16:9", "2k"): "2048x1152",
    ("16:9", "4k"): "4096x2304",
    ("9:16", "2k"): "1152x2048",
    ("9:16", "4k"): "2304x4096",
    ("4:3", "2k"): "2048x1536",
    ("4:3", "4k"): "4096x3072",
    ("3:4", "2k"): "1536x2048",
    ("3:4", "4k"): "3072x4096",
    ("3:2", "2k"): "2048x1360",
    ("3:2", "4k"): "4096x2720",
    ("2:3", "2k"): "1360x2048",
    ("2:3", "4k"): "2720x4096",
}
_SIZE_SUFFIX_RE = re.compile(
    r"(?:^|\s)(?P<size>(?:\d{3,5}[x×]\d{3,5}|\d{1,2}:\d{1,2}(?:\s+(?:1|2|4)k)?|(?:1|2|4)k))$",
    re.IGNORECASE,
)
_CAPTURE_DEVICE_RE = re.compile(
    r"(?:智能手机|手机|移动电话|前置摄像头|后置摄像头|摄像头|相机|摄影机|摄像机|"
    r"自拍杆|屏幕|取景器|拍摄界面|相机界面|smartphone|phone\s*screen|"
    r"selfie\s*stick|camcorder|camera|screen|phone)",
    re.IGNORECASE,
)
_SELFIE_VIEW_RE = re.compile(
    r"(?:对镜|镜面|手持|自然)?\s*(?:自拍(?:照|视角|角度)?|selfie)",
    re.IGNORECASE,
)


def _sanitize_selfie_text(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = _CAPTURE_DEVICE_RE.sub("", text)
    text = _SELFIE_VIEW_RE.sub("由他人拍摄的生活照", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*([，,、；;])\s*", r"\1", text)
    text = re.sub(r"([，,、；;])(?:[，,、；;])+", r"\1", text)
    return text.strip(" \t,，、;；")


def normalize_output_size(value: str, fallback: str = "1024x1024") -> str:
    """Convert UI size values into the WIDTHxHEIGHT/auto form accepted by image APIs."""
    normalized = " ".join(str(value or "").strip().lower().split()).replace("×", "x")
    if normalized == "auto":
        return "auto"

    dimension = re.fullmatch(r"(\d{3,5})x(\d{3,5})", normalized)
    if dimension:
        return f"{dimension.group(1)}x{dimension.group(2)}"

    tokens = re.split(r"[\s,/]+", normalized)
    ratio = next((token for token in tokens if re.fullmatch(r"\d{1,2}:\d{1,2}", token)), "")
    resolution = next((token for token in tokens if token in _RESOLUTION_SQUARE_SIZES), "")
    if ratio:
        if not resolution:
            return RATIO_SIZES.get(ratio, fallback or "1024x1024")
        if resolution == "1k":
            return RATIO_SIZES.get(ratio, fallback or "1024x1024")
        mapped = _RATIO_RESOLUTION_SIZES.get((ratio, resolution))
        if mapped:
            return mapped
        try:
            ratio_width, ratio_height = (int(part) for part in ratio.split(":"))
            long_side = int(resolution[:-1]) * 1024
            if ratio_width >= ratio_height:
                return f"{long_side}x{round(long_side * ratio_height / ratio_width)}"
            return f"{round(long_side * ratio_width / ratio_height)}x{long_side}"
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    if resolution:
        return _RESOLUTION_SQUARE_SIZES[resolution]

    fallback_value = str(fallback or "").strip()
    if fallback_value and fallback_value != value:
        return normalize_output_size(fallback_value, "1024x1024")
    return "1024x1024"


def split_size_suffix(text: str, default_size: str) -> tuple[str, str]:
    """Allow the familiar ``/aiimg prompt 16:9`` command form."""
    value = str(text or "").strip()
    match = _SIZE_SUFFIX_RE.search(value)
    if not match:
        return value, normalize_output_size(default_size)
    normalized = normalize_output_size(match.group("size"))
    prompt = value[: match.start()].strip()
    return prompt, normalized


def build_selfie_prompt(
    user_prompt: str,
    life_context: dict[str, Any] | None,
    prefix: str,
) -> str:
    """Merge life companion data as defaults while preserving user intent."""
    context = life_context if isinstance(life_context, dict) else {}
    outfit = _sanitize_selfie_text(context.get("outfit"))
    schedule = _sanitize_selfie_text(context.get("schedule"))
    image_prompt = _sanitize_selfie_text(context.get("image_prompt"))
    requested = (
        _sanitize_selfie_text(user_prompt)
        or image_prompt
        or "自然真实的生活照"
    )

    lines = [_sanitize_selfie_text(prefix)]
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
            activity = _sanitize_selfie_text(item.get("activity"))
            if time_value and activity:
                entries.append(f"{time_value} {activity[:120]}")
        if entries:
            lines.extend(["- 今日时间线：", *entries])
    if outfit or schedule or timeline:
        lines.append("用户没有指定其它穿搭或场景时，优先保持上述生活状态一致。")
    capture_policy = (
        "拍摄方式规则：把人物表现为由他人从画面外拍摄的自然生活照，而不是自拍；"
        "人物自然面对镜头或专注当前活动，双手自然活动，不做伸手举手机、对镜看屏幕或持拍摄设备的动作。"
        "镜头和拍摄设备都位于画面外，成片不得出现手机、相机、摄像头、自拍杆、屏幕、取景器或任何拍摄界面。"
        "即使用户或生活状态文字提到自拍或上述设备，也只保留动作和场景含义，改为第三人拍摄；"
        "第三人拍摄、恋人视角或定时拍摄均按上述第三人称生活照处理；"
        "人物手势和手持咖啡杯等非拍摄类日常物品遵循用户要求。"
    )
    lines.extend(["", f"用户要求（最高优先级）：{requested}", "", capture_policy])
    return "\n".join(line for line in lines if line is not None).strip()
