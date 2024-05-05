from PIL import Image

import torch

from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)
from diffusers.schedulers.scheduling_euler_ancestral_discrete import (
    EulerAncestralDiscreteScheduler,
)
from diffusers.models.attention_processor import AttnProcessor2_0

try:
    import spaces
except ImportError:

    class spaces:
        def GPU(*args, **kwargs):
            return lambda x: x


import gradio as gr
from utils import NEGATIVE_PROMPT, IMAGE_SIZE_OPTIONS, QUALITY_TAGS, IMAGE_SIZES


device = "cuda"
model_name: str = "cagliostrolab/animagine-xl-3.1"
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
    add_watermarker=False,
    custom_pipeline="lpw_stable_diffusion_xl",
)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
    model_name,
    subfolder="scheduler",
)

# sdpa
pipe.unet.set_attn_processor(AttnProcessor2_0())

pipe.to(device)


def image_generation_config_ui():
    with gr.Accordion(label="Image generation config", open=False) as accordion:
        image_size = gr.Radio(
            label="Image size",
            choices=list(IMAGE_SIZE_OPTIONS.keys()),
            value=list(IMAGE_SIZE_OPTIONS.keys())[3],
            interactive=True,
        )

        quality_tags = gr.Textbox(
            label="Quality tags",
            placeholder=QUALITY_TAGS["default"],
            value=QUALITY_TAGS["default"],
            interactive=True,
        )
        negative_prompt = gr.Textbox(
            label="Negative prompt",
            placeholder=NEGATIVE_PROMPT["default"],
            value=NEGATIVE_PROMPT["default"],
            interactive=True,
        )

        num_inference_steps = gr.Slider(
            label="Num inference steps",
            minimum=20,
            maximum=30,
            step=1,
            value=25,
            interactive=True,
        )
        guidance_scale = gr.Slider(
            label="Guidance scale",
            minimum=0.0,
            maximum=10.0,
            step=0.5,
            value=7.0,
            interactive=True,
        )

    return accordion, [
        image_size,
        quality_tags,
        negative_prompt,
        num_inference_steps,
        guidance_scale,
    ]


class ImageGenerator:
    # pipe: StableDiffusionXLPipeline

    def __init__(self, model_name: str = "cagliostrolab/animagine-xl-3.1"):
        pass

    @spaces.GPU()
    def generate(
        self,
        prompt: str,
        image_size: str = "768x1344",
        quality_tags: str = QUALITY_TAGS["default"],  # Light v3.1
        negative_prompt: str = NEGATIVE_PROMPT["default"],  # Light v3.1
        num_inference_steps: int = 25,
        guidance_scale: float = 7.0,
    ) -> Image.Image:
        width, height = IMAGE_SIZES[image_size]

        prompt = ", ".join([prompt, quality_tags])

        print("prompt", prompt)
        print("negative_prompt", negative_prompt)
        print("height", height)
        print("width", width)
        print("num_inference_steps", num_inference_steps)
        print("guidance_scale", guidance_scale)

        return pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images
