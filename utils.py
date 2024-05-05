from dartrs.v2 import AspectRatioTag, LengthTag, RatingTag, IdentityTag

# from https://huggingface.co/spaces/cagliostrolab/animagine-xl-3.1/blob/main/config.py
QUALITY_TAGS = {
    "default": "(masterpiece), best quality, very aesthetic, perfect face",
}
NEGATIVE_PROMPT = {
    "default": "nsfw, (low quality, worst quality:1.2), very displeasing, 3d, watermark, signature, ugly, poorly drawn",
}


IMAGE_SIZE_OPTIONS = {
    "1536x640": "<|aspect_ratio:ultra_wide|>",
    "1344x768": "<|aspect_ratio:wide|>",
    "1024x1024": "<|aspect_ratio:square|>",
    "768x1344": "<|aspect_ratio:tall|>",
    "640x1536": "<|aspect_ratio:ultra_tall|>",
}
IMAGE_SIZES = {
    "1536x640": (1536, 640),
    "1344x768": (1344, 768),
    "1024x1024": (1024, 1024),
    "768x1344": (768, 1344),
    "640x1536": (640, 1536),
}

ASPECT_RATIO_OPTIONS: dict[str, AspectRatioTag] = {
    "ultra_wide": "<|aspect_ratio:ultra_wide|>",
    "wide": "<|aspect_ratio:wide|>",
    "square": "<|aspect_ratio:square|>",
    "tall": "<|aspect_ratio:tall|>",
    "ultra_tall": "<|aspect_ratio:ultra_tall|>",
}
RATING_OPTIONS: dict[str, RatingTag] = {
    "sfw": "<|rating:sfw|>",
    "general": "<|rating:general|>",
    "sensitive": "<|rating:sensitive|>",
    "nsfw": "<|rating:nsfw|>",
    "questionable": "<|rating:questionable|>",
    "explicit": "<|rating:explicit|>",
}
LENGTH_OPTIONS: dict[str, LengthTag] = {
    "very_short": "<|length:very_short|>",
    "short": "<|length:short|>",
    "medium": "<|length:medium|>",
    "long": "<|length:long|>",
    "very_long": "<|length:very_long|>",
}
IDENTITY_OPTIONS: dict[str, IdentityTag] = {
    "none": "<|identity:none|>",
    "lax": "<|identity:lax|>",
    "strict": "<|identity:strict|>",
}


PEOPLE_TAGS = [
    *[f"1{x}" for x in ["girl", "boy", "other"]],
    *[f"{i}girls" for i in range(2, 6)],
    *[f"6+{x}s" for x in ["girl", "boy", "other"]],
    "no humans",
]
