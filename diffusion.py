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


from utils import NEGATIVE_PROMPT


class ImageGenerator:
    pipe: StableDiffusionXLPipeline

    def __init__(self, model_name: str = "cagliostrolab/animagine-xl-3.1"):
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            use_safetensors=True,
            add_watermarker=False,
            custom_pipeline="lpw_stable_diffusion_xl",
        )
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
            model_name,
            subfolder="scheduler",
        )

        # xformers
        # self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.unet.set_attn_processor(AttnProcessor2_0())

        try:
            self.pipe = torch.compile(self.pipe)
        except Exception as e:
            print("torch.compile is not supported on this system")

        self.pipe.to("cuda")

    @torch.no_grad()
    @spaces.GPU(duration=30)
    def generate(
        self,
        prompt: str,
        negative_prompt: str = NEGATIVE_PROMPT["default"],  # Light v3.1
        height: int = 1152,
        width: int = 896,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.0,
    ) -> Image.Image:
        print("prompt", prompt)
        print("negative_prompt", negative_prompt)
        print("height", height)
        print("width", width)
        print("num_inference_steps", num_inference_steps)
        print("guidance_scale", guidance_scale)

        return self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images
