from typing import Callable
from PIL import Image

import gradio as gr

from v2 import V2UI
from diffusion import ImageGenerator
from output import UpsamplingOutput
from utils import QUALITY_TAGS, NEGATIVE_PROMPT, IMAGE_SIZE_OPTIONS, PEOPLE_TAGS


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
    def _parse_upsampling_output(*args) -> tuple[
        str,
        str,
        dict,
    ]:
        output = upsampler(*args)

        print(output)

        return (
            animagine_xl_v3_1(output),
            elapsed_time_format(output.elapsed_time),
            gr.update(
                interactive=True,
            ),
        )

    return _parse_upsampling_output


def image_generation_config_ui():
    with gr.Accordion(label="Image generation config", open=False) as accordion:
        image_size = gr.Radio(
            label="Image size",
            choices=list(IMAGE_SIZE_OPTIONS.keys()),
            value=list(IMAGE_SIZE_OPTIONS.keys())[3],  # tall
        )

        quality_tags = gr.Textbox(
            label="Quality tags",
            placeholder=QUALITY_TAGS["default"],
            value=QUALITY_TAGS["default"],
        )
        negative_prompt = gr.Textbox(
            label="Negative prompt",
            placeholder=NEGATIVE_PROMPT["default"],
            value=NEGATIVE_PROMPT["default"],
        )

        num_inference_steps = gr.Slider(
            label="Num inference steps",
            minimum=20,
            maximum=30,
            step=1,
            value=25,
        )
        guidance_scale = gr.Slider(
            label="Guidance scale",
            minimum=0.0,
            maximum=10.0,
            step=0.5,
            value=7.0,
        )

    return accordion, [
        image_size,
        quality_tags,
        negative_prompt,
        num_inference_steps,
        guidance_scale,
    ]


def description_ui():
    gr.Markdown(
        """
        # Danbooru Tags Transformer V2 Demo
        """
    )


def main():

    v2 = V2UI()

    print("Loading diffusion model...")
    # image_generator = ImageGenerator()
    print("Loaded.")

    with gr.Blocks() as ui:
        description_ui()

        with gr.Row():
            with gr.Column():
                v2.ui()

            with gr.Column():
                output_text = gr.TextArea(label="Output tags", interactive=False)

                elapsed_time_md = gr.Markdown(label="Elapsed time", value="")

                generate_image_btn = gr.Button(
                    value="Generate image with this prompt!",
                )

                accordion, image_generation_config_components = (
                    image_generation_config_ui()
                )

                output_image = gr.Gallery(
                    label="Output image",
                    columns=1,
                    preview=True,
                    show_label=False,
                    visible=False,
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
                        ],
                        [
                            "original",
                            "",
                            "1girl, solo, blue theme, limited palette",
                            "sfw",
                            "ultra_wide",
                            "long",
                            "lax",
                        ],
                        [
                            "",
                            "",
                            "4girls",
                            "sfw",
                            "tall",
                            "very_long",
                            "lax",
                        ],
                        [
                            "original",
                            "",
                            "1girl, solo, upper body, looking at viewer, profile picture",
                            "sfw",
                            "square",
                            "medium",
                            "none",
                        ],
                        [
                            "",
                            "",
                            "no humans, scenery, spring (season)",
                            "general",
                            "ultra_wide",
                            "medium",
                            "lax",
                        ],
                        [
                            "sousou no frieren",
                            "frieren",
                            "1girl, solo",
                            "general",
                            "tall",
                            "long",
                            "lax",
                        ],
                        [
                            "honkai: star rail",
                            "silver wolf (honkai: star rail)",
                            "1girl, solo, annoyed",
                            "sfw",
                            "tall",
                            "long",
                            "lax",
                        ],
                        [
                            "chuunibyou demo koi ga shitai!",
                            "takanashi rikka",
                            "1girl, solo",
                            "sfw",
                            "ultra_tall",
                            "medium",
                            "lax",
                        ],
                    ],
                    inputs=[
                        *v2.get_inputs()[1:8],
                    ],
                )

        v2.get_generate_btn().click(
            parse_upsampling_output(v2.on_generate),
            inputs=[
                *v2.get_inputs(),
            ],
            outputs=[output_text, elapsed_time_md, generate_image_btn],
        )

    ui.launch()


if __name__ == "__main__":
    main()
