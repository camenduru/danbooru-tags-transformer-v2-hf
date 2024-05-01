import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

import gradio as gr
from gradio.components import Component

try:
    import spaces
except ImportError:

    class spaces:
        def GPU(*args, **kwargs):
            return lambda x: x


from output import UpsamplingOutput
from utils import IMAGE_SIZE_OPTIONS, RATING_OPTIONS, LENGTH_OPTIONS, IDENTITY_OPTIONS

ALL_MODELS = {
    "dart-v2-llama-100m-sft": {
        "repo": "p1atdev/dart-v2-llama-100m-sft",
        "type": "sft",
    },
    "dart-v2-mistral-100m-sft": {
        "repo": "p1atdev/dart-v2-mistral-100m-sft",
        "type": "sft",
    },
    "dart-v2-mixtral-160m-sft": {
        "repo": "p1atdev/dart-v2-mixtral-160m-sft",
        "type": "sft",
    },
}


def prepare_models(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    return {
        "tokenizer": tokenizer,
        "model": model,
    }


def normalize_tags(tokenizer: PreTrainedTokenizerBase, tags: str):
    """Just remove unk tokens."""
    return ", ".join(
        tokenizer.batch_decode(
            [
                token
                for token in tokenizer.encode_plus(
                    tags.strip(),
                    return_tensors="pt",
                ).input_ids[0]
                if int(token) != tokenizer.unk_token_id
            ],
            skip_special_tokens=True,
        )
    )


def compose_prompt(
    copyright: str = "",
    character: str = "",
    general: str = "",
    rating: str = "<|rating:sfw|>",
    aspect_ratio: str = "<|aspect_ratio:tall|>",
    length: str = "<|length:long|>",
    identity: str = "<|identity:none|>",
):
    prompt = (
        f"<|bos|>"
        f"<copyright>{copyright.strip()}</copyright>"
        f"<character>{character.strip()}</character>"
        f"{rating}{aspect_ratio}{length}"
        f"<general>{general.strip()}{identity}<|input_end|>"
    )

    return prompt


@torch.no_grad()
@spaces.GPU(duration=5)
def generate_tags(
    model,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
):
    print(  # debug
        tokenizer.tokenize(
            prompt,
            add_special_tokens=False,
        )
    )
    input_ids = tokenizer.encode_plus(prompt, return_tensors="pt").input_ids
    output = model.generate(
        input_ids.to(model.device),
        do_sample=True,
        temperature=1,
        top_p=0.9,
        top_k=100,
        num_beams=1,
        num_return_sequences=1,
        max_length=256,
    )

    # remove input tokens
    pure_output_ids = output[0][len(input_ids[0]) :]

    return ", ".join(
        [
            token
            for token in tokenizer.batch_decode(
                pure_output_ids, skip_special_tokens=True
            )
            if token.strip() != ""
        ]
    )


class V2UI:
    model_name: str | None = None
    model: AutoModelForCausalLM
    tokenizer: PreTrainedTokenizerBase

    input_components: list[Component] = []
    generate_btn: gr.Button

    def on_generate(
        self,
        model_name: str,
        copyright_tags: str,
        character_tags: str,
        general_tags: str,
        rating_option: str,
        # aspect_ratio_option: str,
        length_option: str,
        identity_option: str,
        image_size: str,  # this is from image generation config
        *args,
    ) -> UpsamplingOutput:
        if self.model_name is None or self.model_name != model_name:
            models = prepare_models(ALL_MODELS[model_name]["repo"])
            self.model = models["model"]
            self.tokenizer = models["tokenizer"]
            self.model_name = model_name

        # normalize tags
        copyright_tags = normalize_tags(self.tokenizer, copyright_tags)
        character_tags = normalize_tags(self.tokenizer, character_tags)
        general_tags = normalize_tags(self.tokenizer, general_tags)

        rating_tag = RATING_OPTIONS[rating_option]
        aspect_ratio_tag = IMAGE_SIZE_OPTIONS[image_size]
        length_tag = LENGTH_OPTIONS[length_option]
        identity_tag = IDENTITY_OPTIONS[identity_option]

        prompt = compose_prompt(
            copyright=copyright_tags,
            character=character_tags,
            general=general_tags,
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
            value="1girl",
        )

        input_rating = gr.Radio(
            label="Rating",
            choices=list(RATING_OPTIONS.keys()),
            value="general",
        )
        # input_aspect_ratio = gr.Radio(
        #     label="Aspect ratio",
        #     choices=["ultra_wide", "wide", "square", "tall", "ultra_tall"],
        #     value="tall",
        # )
        input_length = gr.Radio(
            label="Length",
            choices=list(LENGTH_OPTIONS.keys()),
            value="long",
        )
        input_identity = gr.Radio(
            label="Keep identity level",
            choices=list(IDENTITY_OPTIONS.keys()),
            value="lax",
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
            # input_aspect_ratio,
            input_length,
            input_identity,
        ]

    def get_generate_btn(self) -> gr.Button:
        return self.generate_btn

    def get_inputs(self) -> list[Component]:
        return self.input_components
