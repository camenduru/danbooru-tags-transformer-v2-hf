from dataclasses import dataclass


@dataclass
class UpsamplingOutput:
    upsampled_tags: str

    copyright_tags: str
    character_tags: str
    general_tags: str
    rating_tag: str
    aspect_ratio_tag: str
    length_tag: str
    identity_tag: str

    elapsed_time: float = 0.0
