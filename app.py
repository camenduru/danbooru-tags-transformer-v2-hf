from typing import Callable
from PIL import Image

import gradio as gr

from v2 import V2UI
from diffusion import ImageGenerator
from output import UpsamplingOutput
from utils import QUALITY_TAGS, NEGATIVE_PROMPT, IMAGE_SIZE_OPTIONS, IMAGE_SIZES


def animagine_xl_v3_1(output: UpsamplingOutput):
    return ", ".join(
        [
            part.strip()
            for part in [
                output.character_tags,
                output.copyright_tags,
                output.general_tags,
                output.upsampled_tags,
                (
                    output.rating_tag
                    if output.rating_tag not in ["<|rating:sfw|>", "<|rating:general|>"]
                    else ""
                ),
            ]
            if part.strip() != ""
        ]
    )


def elapsed_time_format(elapsed_time: float) -> str:
    return f"Elapsed: {elapsed_time:.2f} seconds"


def parse_upsampling_output(
    upsampler: Callable[..., UpsamplingOutput],
    image_generator: Callable[..., Image.Image],
):
    def _parse_upsampling_output(
        generate_image: bool, *args
    ) -> tuple[str, str, Image.Image | None]:
        output = upsampler(*args)

        print(output)

        if not generate_image:
            return (
                animagine_xl_v3_1(output),
                elapsed_time_format(output.elapsed_time),
                None,
            )

        # generate image
        [
            image_size_option,
            quality_tags,
            negative_prompt,
            num_inference_steps,
            guidance_scale,
        ] = args[
            7:
        ]  # remove the first 7 arguments for upsampler
        width, height = IMAGE_SIZES[image_size_option]
        image = image_generator(
            ", ".join([animagine_xl_v3_1(output), quality_tags]),
            negative_prompt,
            height,
            width,
            num_inference_steps,
            guidance_scale,
        )

        return (
            animagine_xl_v3_1(output),
            elapsed_time_format(output.elapsed_time),
            image,
        )

    return _parse_upsampling_output


def toggle_visible_output_image(generate_image: bool):
    return gr.update(
        visible=generate_image,
    )


def image_generation_config_ui():
    with gr.Accordion(label="Image generation config", open=True) as accordion:
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
    image_generator = ImageGenerator()
    print("Loaded.")

    with gr.Blocks() as ui:
        description_ui()

        with gr.Row():
            with gr.Column():
                v2.ui()

                generate_image_check = gr.Checkbox(
                    label="Also generate image", value=True
                )

                accordion, image_generation_config_components = (
                    image_generation_config_ui()
                )

            with gr.Column():
                output_text = gr.TextArea(label="Output tags", interactive=False)

                elapsed_time_md = gr.Markdown(label="Elapsed time", value="")

                output_image = gr.Gallery(
                    label="Output image",
                    columns=1,
                    preview=True,
                    show_label=False,
                    visible=True,
                )

                gr.Examples(
                    examples=[
                        ["original", "", "1girl, solo", "832x1216"],
                        ["original", "", "3girls, 2boys", "1536x640"],
                        [
                            "sousou no frieren",
                            "frieren (sousou no frieren)",
                            "1girl, solo",
                            "832x1216",
                        ],
                        [
                            "honkai: star rail",
                            "silver wolf (honkai: star rail)",
                            "1girl, solo",
                            "832x1216",
                        ],
                        [
                            "bocchi the rock!",
                            "gotoh hitori, kita ikuyo, ijichi nijika, yamada ryo",
                            "4girls, multiple girls",
                            "1216x832",
                        ],
                        [
                            "chuunibyou demo koi ga shitai!",
                            "takanashi rikka",
                            "1girl, solo",
                            "640x1536",
                        ],
                    ],
                    inputs=[
                        *v2.get_inputs()[1:4],
                        image_generation_config_components[0],  # image size
                    ],
                )

        v2.get_generate_btn().click(
            parse_upsampling_output(v2.on_generate, image_generator.generate),
            inputs=[
                generate_image_check,
                *v2.get_inputs(),
                *image_generation_config_components,
            ],
            outputs=[output_text, elapsed_time_md, output_image],
        )
        generate_image_check.change(
            toggle_visible_output_image,
            inputs=[generate_image_check],
            outputs=[output_image],
        )

    ui.launch()


if __name__ == "__main__":
    main()
