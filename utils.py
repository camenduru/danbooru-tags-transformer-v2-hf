# from https://huggingface.co/spaces/cagliostrolab/animagine-xl-3.1/blob/main/config.py
QUALITY_TAGS = {
    "default": "(masterpiece), best quality, very aesthetic, perfect face",
}
NEGATIVE_PROMPT = {
    "default": "nsfw, (low quality, worst quality:1.2), very displeasing, 3d, watermark, signature, ugly, poorly drawn",
}


IMAGE_SIZE_OPTIONS = {
    "1536x640": "<|aspect_ratio:ultra_wide|>",
    "1216x832": "<|aspect_ratio:wide|>",
    "1024x1024": "<|aspect_ratio:square|>",
    "832x1216": "<|aspect_ratio:tall|>",
    "640x1536": "<|aspect_ratio:ultra_tall|>",
}
IMAGE_SIZES = {
    "1536x640": (1536, 640),
    "1216x832": (1216, 832),
    "1024x1024": (1024, 1024),
    "832x1216": (832, 1216),
    "640x1536": (640, 1536),
}

RATING_OPTIONS = {
    "sfw": "<|rating:sfw|>",
    "general": "<|rating:general|>",
    "sensitive": "<|rating:sensitive|>",
    "nsfw": "<|rating:nsfw|>",
    "questionable": "<|rating:questionable|>",
    "explicit": "<|rating:explicit|>",
}
LENGTH_OPTIONS = {
    "very_short": "<|length:very_short|>",
    "short": "<|length:short|>",
    "medium": "<|length:medium|>",
    "long": "<|length:long|>",
    "very_long": "<|length:very_long|>",
}
IDENTITY_OPTIONS = {
    "none": "<|identity:none|>",
    "lax": "<|identity:lax|>",
    "strict": "<|identity:strict|>",
}
