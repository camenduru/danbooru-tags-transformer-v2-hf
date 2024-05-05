from typing import Callable
from PIL import Image

import gradio as gr

from v2 import V2UI
from diffusion import ImageGenerator, image_generation_config_ui
from output import UpsamplingOutput
from utils import (
    PEOPLE_TAGS,
    gradio_copy_text,
    COPY_ACTION_JS,
)


NORMALIZE_RATING_TAG = {
    "<|rating:sfw|>": "",
    "<|rating:general|>": "",
    "<|rating:sensitive|>": "sensitive",
    "<|rating:nsfw|>": "nsfw",
    "<|rating:questionable|>": "nsfw",
    "<|rating:explicit|>": "nsfw, explicit",
}


def example(
    copyright: str,
    character: str,
    general: str,
    rating: str,
    aspect_ratio: str,
    length: str,
    identity: str,
    image_size: str,
):
    return [
        copyright,
        character,
        general,
        rating,
        aspect_ratio,
        length,
        identity,
        image_size,
    ]


GRADIO_EXAMPLES = [
    example(
        copyright="original",
        character="",
        general="1girl, solo, upper body, :d",
        rating="general",
        aspect_ratio="tall",
        length="long",
        identity="none",
        image_size="768x1344",
    ),
    example(
        copyright="original",
        character="",
        general="1girl, solo, blue theme, limited palette",
        rating="sfw",
        aspect_ratio="ultra_wide",
        length="long",
        identity="lax",
        image_size="1536x640",
    ),
    example(
        copyright="",
        character="",
        general="4girls",
        rating="sfw",
        aspect_ratio="tall",
        length="very_long",
        identity="lax",
        image_size="768x1344",
    ),
    example(
        copyright="original",
        character="",
        general="1girl, solo, upper body, looking at viewer, profile picture",
        rating="sfw",
        aspect_ratio="square",
        length="medium",
        identity="none",
        image_size="1024x1024",
    ),
    example(
        copyright="original",
        character="",
        general="1girl, solo, chibi, upper body, looking at viewer, simple background, limited palette, square",
        rating="sfw",
        aspect_ratio="square",
        length="medium",
        identity="none",
        image_size="1024x1024",
    ),
    example(
        copyright="original",
        character="",
        general="1girl, full body, aqua eyes, baseball cap, blonde hair, closed mouth, earrings, hat, hoop earrings, jewelry, looking at viewer, shirt, short hair, solo, yellow shirt, simple background, green background",
        rating="sfw",
        aspect_ratio="tall",
        length="very_long",
        identity="strict",
        image_size="768x1344",
    ),
    example(
        copyright="",
        character="",
        general="no humans, scenery, spring (season)",
        rating="general",
        aspect_ratio="ultra_wide",
        length="medium",
        identity="lax",
        image_size="1536x640",
    ),
    example(
        copyright="",
        character="",
        general="no humans, cyberpunk, city, cityscape, building, neon lights, pixel art",
        rating="general",
        aspect_ratio="ultra_wide",
        length="medium",
        identity="lax",
        image_size="1536x640",
    ),
    example(
        copyright="sousou no frieren",
        character="frieren",
        general="1girl, solo",
        rating="general",
        aspect_ratio="tall",
        length="long",
        identity="lax",
        image_size="768x1344",
    ),
    example(
        copyright="honkai: star rail",
        character="firefly (honkai: star rail)",
        general="1girl, solo",
        rating="sfw",
        aspect_ratio="tall",
        length="medium",
        identity="lax",
        image_size="768x1344",
    ),
    example(
        copyright="honkai: star rail",
        character="silver wolf (honkai: star rail)",
        general="1girl, solo, annoyed",
        rating="sfw",
        aspect_ratio="tall",
        length="long",
        identity="lax",
        image_size="768x1344",
    ),
    example(
        copyright="chuunibyou demo koi ga shitai!",
        character="takanashi rikka",
        general="1girl, solo",
        rating="sfw",
        aspect_ratio="ultra_tall",
        length="medium",
        identity="lax",
        image_size="640x1536",
    ),
]


def animagine_xl_v3_1(output: UpsamplingOutput):
    # separate people tags (e.g. 1girl)
    people_tags = []
    other_general_tags = []
    for tag in output.general_tags.split(","):
        tag = tag.strip()
        if tag in PEOPLE_TAGS:
            people_tags.append(tag)
        else:
            other_general_tags.append(tag)

    return ", ".join(
        [
            part.strip()
            for part in [
                *people_tags,
                output.character_tags,
                output.copyright_tags,
                *other_general_tags,
                output.upsampled_tags,
                NORMALIZE_RATING_TAG[output.rating_tag],
            ]
            if part.strip() != ""
        ]
    )


def elapsed_time_format(elapsed_time: float) -> str:
    return f"Elapsed: {elapsed_time:.2f} seconds"


def parse_upsampling_output(
    upsampler: Callable[..., UpsamplingOutput],
):
    def _parse_upsampling_output(*args) -> tuple[str, str, dict, dict]:
        output = upsampler(*args)

        print(output)

        return (
            animagine_xl_v3_1(output),
            elapsed_time_format(output.elapsed_time),
            gr.update(
                interactive=True,
            ),
            gr.update(
                interactive=True,
            ),
        )

    return _parse_upsampling_output


def description_ui():
    gr.Markdown(
        """
        # Danbooru Tags Transformer V2 Demo
        """
    )


def main():

    v2 = V2UI()

    print("Loading diffusion model...")
    image_generator = ImageGenerator()
    print("Loaded.")

    with gr.Blocks() as ui:
        description_ui()

        with gr.Row():
            with gr.Column():
                v2.ui()

            with gr.Column():
                generate_btn = gr.Button(value="Generate tags", variant="primary")

                with gr.Group():
                    output_text = gr.TextArea(label="Output tags", interactive=False)
                    copy_btn = gr.Button(
                        value="Copy to clipboard",
                        interactive=False,
                    )

                elapsed_time_md = gr.Markdown(label="Elapsed time", value="")

                generate_image_btn = gr.Button(
                    value="Generate image with this prompt!",
                    interactive=False,
                )

                accordion, image_generation_config_components = (
                    image_generation_config_ui()
                )

                output_image = gr.Gallery(
                    label="Generated image",
                    show_label=True,
                    columns=1,
                    preview=True,
                    visible=True,
                )

                gr.Examples(
                    examples=GRADIO_EXAMPLES,
                    inputs=[
                        *v2.get_inputs()[1:8],
                        image_generation_config_components[0],  # image_size
                    ],
                )

        generate_btn.click(
            parse_upsampling_output(v2.on_generate),
            inputs=[
                *v2.get_inputs(),
            ],
            outputs=[output_text, elapsed_time_md, copy_btn, generate_image_btn],
        )
        copy_btn.click(gradio_copy_text, inputs=[output_text], js=COPY_ACTION_JS)
        generate_image_btn.click(
            image_generator.generate,
            inputs=[output_text, *image_generation_config_components],
            outputs=[output_image],
        )

    ui.launch()


if __name__ == "__main__":
    main()
