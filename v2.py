import time
import os

import torch

from dartrs.v2 import (
    V2Model,
    MixtralModel,
    compose_prompt,
)
from dartrs.dartrs import DartTokenizer
from dartrs.utils import get_generation_config


import gradio as gr
from gradio.components import Component

try:
    import spaces
except ImportError:

    class spaces:
        def GPU(*args, **kwargs):
            return lambda x: x


from output import UpsamplingOutput
from utils import ASPECT_RATIO_OPTIONS, RATING_OPTIONS, LENGTH_OPTIONS, IDENTITY_OPTIONS

HF_TOKEN = os.getenv("HF_TOKEN", None)

ALL_MODELS = {
    "dart-v2-mixtral-160m-sft-8": {
        "repo": "p1atdev/dart-v2-mixtral-160m-sft-8",
        "type": "sft",
        "class": MixtralModel,
    },
}


def prepare_models(model_config: dict):
    model_name = model_config["repo"]
    tokenizer = DartTokenizer.from_pretrained(model_name, auth_token=HF_TOKEN)
    model = model_config["class"].from_pretrained(model_name, auth_token=HF_TOKEN)

    return {
        "tokenizer": tokenizer,
        "model": model,
    }


# def normalize_tags(tokenizer: PreTrainedTokenizerBase, tags: str):
#     """Just remove unk tokens."""
#     return ", ".join(
#         tokenizer.batch_decode(
#             [
#                 token
#                 for token in tokenizer.encode_plus(
#                     tags.strip(),
#                     return_tensors="pt",
#                 ).input_ids[0]
#                 if int(token) != tokenizer.unk_token_id
#             ],
#             skip_special_tokens=True,
#         )
#     )


@torch.no_grad()
@spaces.GPU(duration=5)
def generate_tags(
    model: V2Model,
    tokenizer: DartTokenizer,
    prompt: str,
    ban_token_ids: list[int],
):
    output = model.generate(
        get_generation_config(
            prompt,
            tokenizer=tokenizer,
            temperature=1,
            top_p=0.9,
            top_k=100,
            max_new_tokens=256,
            ban_token_ids=ban_token_ids,
        ),
    )

    return output


class V2UI:
    model_name: str | None = None
    model: V2Model
    tokenizer: DartTokenizer

    input_components: list[Component] = []
    generate_btn: gr.Button

    def on_generate(
        self,
        model_name: str,
        copyright_tags: str,
        character_tags: str,
        general_tags: str,
        rating_option: str,
        aspect_ratio_option: str,
        length_option: str,
        identity_option: str,
        ban_tags: str,
        *args,
    ) -> UpsamplingOutput:
        if self.model_name is None or self.model_name != model_name:
            models = prepare_models(ALL_MODELS[model_name])
            self.model = models["model"]
            self.tokenizer = models["tokenizer"]
            self.model_name = model_name

        # normalize tags
        # copyright_tags = normalize_tags(self.tokenizer, copyright_tags)
        # character_tags = normalize_tags(self.tokenizer, character_tags)
        # general_tags = normalize_tags(self.tokenizer, general_tags)

        rating_tag = RATING_OPTIONS[rating_option]
        aspect_ratio_tag = ASPECT_RATIO_OPTIONS[aspect_ratio_option]
        length_tag = LENGTH_OPTIONS[length_option]
        identity_tag = IDENTITY_OPTIONS[identity_option]
        ban_token_ids = self.tokenizer.encode(ban_tags.strip())

        prompt = compose_prompt(
            prompt=general_tags,
            copyright=copyright_tags,
            character=character_tags,
            rating=rating_tag,
            aspect_ratio=aspect_ratio_tag,
            length=length_tag,
            identity=identity_tag,
        )

        start = time.time()
        upsampled_tags = generate_tags(
            self.model,
            self.tokenizer,
            prompt,
            ban_token_ids,
        )
        elapsed_time = time.time() - start

        return UpsamplingOutput(
            upsampled_tags=upsampled_tags,
            copyright_tags=copyright_tags,
            character_tags=character_tags,
            general_tags=general_tags,
            rating_tag=rating_tag,
            aspect_ratio_tag=aspect_ratio_tag,
            length_tag=length_tag,
            identity_tag=identity_tag,
            elapsed_time=elapsed_time,
        )

    def ui(self):
        input_copyright = gr.Textbox(
            label="Copyright tags",
            placeholder="vocaloid",
        )
        input_character = gr.Textbox(
            label="Character tags",
            placeholder="hatsune miku",
        )
        input_general = gr.TextArea(
            label="General tags",
            lines=4,
            placeholder="1girl, ...",
            value="1girl, solo",
        )

        input_rating = gr.Radio(
            label="Rating",
            choices=list(RATING_OPTIONS.keys()),
            value="general",
        )
        input_aspect_ratio = gr.Radio(
            label="Aspect ratio",
            choices=["ultra_wide", "wide", "square", "tall", "ultra_tall"],
            value="tall",
        )
        input_length = gr.Radio(
            label="Length",
            choices=list(LENGTH_OPTIONS.keys()),
            value="long",
        )
        input_identity = gr.Radio(
            label="Keep identity level",
            choices=list(IDENTITY_OPTIONS.keys()),
            value="none",
        )

        with gr.Accordion(label="Advanced options", open=False):
            input_ban_tags = gr.Textbox(
                label="Ban tags",
                placeholder="alternate costumen, ...",
            )

        model_name = gr.Dropdown(
            label="Model",
            choices=list(ALL_MODELS.keys()),
            value=list(ALL_MODELS.keys())[0],
        )

        self.generate_btn = gr.Button(value="Generate", variant="primary")

        self.input_components = [
            model_name,
            input_copyright,
            input_character,
            input_general,
            input_rating,
            input_aspect_ratio,
            input_length,
            input_identity,
            input_ban_tags,
        ]

    def get_generate_btn(self) -> gr.Button:
        return self.generate_btn

    def get_inputs(self) -> list[Component]:
        return self.input_components
