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
                    examples=[
                        [
                            "original",
                            "",
                            "1girl, solo, upper body, :d",
                            "general",
                            "tall",
                            "long",
                            "none",
                            "768x1344",
                        ],
                        [
                            "original",
                            "",
                            "1girl, solo, blue theme, limited palette",
                            "sfw",
                            "ultra_wide",
                            "long",
                            "lax",
                            "1536x640",
                        ],
                        [
                            "",
                            "",
                            "4girls",
                            "sfw",
                            "tall",
                            "very_long",
                            "lax",
                            "768x1344",
                        ],
                        [
                            "original",
                            "",
                            "1girl, solo, upper body, looking at viewer, profile picture",
                            "sfw",
                            "square",
                            "medium",
                            "none",
                            "1024x1024",
                        ],
                        [
                            "",
                            "",
                            "no humans, scenery, spring (season)",
                            "general",
                            "ultra_wide",
                            "medium",
                            "lax",
                            "1536x640",
                        ],
                        [
                            "sousou no frieren",
                            "frieren",
                            "1girl, solo",
                            "general",
                            "tall",
                            "long",
                            "lax",
                            "768x1344",
                        ],
                        [
                            "honkai: star rail",
                            "firefly (honkai: star rail)",
                            "1girl, solo",
                            "sfw",
                            "tall",
                            "medium",
                            "lax",
                            "768x1344",
                        ],
                        [
                            "honkai: star rail",
                            "silver wolf (honkai: star rail)",
                            "1girl, solo, annoyed",
                            "sfw",
                            "tall",
                            "long",
                            "lax",
                            "768x1344",
                        ],
                        [
                            "chuunibyou demo koi ga shitai!",
                            "takanashi rikka",
                            "1girl, solo",
                            "sfw",
                            "ultra_tall",
                            "medium",
                            "lax",
                            "640x1536",
                        ],
                    ],
                    inputs=[
                        *v2.get_inputs()[1:8],
                        image_generation_config_components[0],  # image_size
                    ],
                )

        v2.get_generate_btn().click(
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
