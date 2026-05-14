import os
from typing import List, Sequence, Union, Optional
from PIL import Image

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from lakonlab.utils.io_utils import hf_model_loader


DEFAULT_TEXT_ONLY_PATH = os.path.abspath(os.path.join(__file__, '../system_prompts/default_text_only.txt'))
DEFAULT_WITH_IMAGES_PATH = os.path.abspath(os.path.join(__file__, '../system_prompts/default_with_images.txt'))


class Qwen3VLPromptRewriter:

    def __init__(
            self,
            model_name_or_path="Qwen/Qwen3-VL-8B-Instruct",
            torch_dtype='bfloat16',
            device_map="auto",
            max_new_tokens_default=128,
            system_prompt_text_only=None,
            system_prompt_wigh_images=None,
            **kwargs):
        if torch_dtype is not None:
            kwargs.update(torch_dtype=getattr(torch, torch_dtype))
        self.model = hf_model_loader(
            Qwen3VLForConditionalGeneration,
            model_name_or_path,
            device_map=device_map,
            **kwargs)
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        # Left padding is safer for batched generation
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"
        self.max_new_tokens_default = max_new_tokens_default
        if system_prompt_text_only is None:
            system_prompt_text_only = open(DEFAULT_TEXT_ONLY_PATH, 'r').read()
        if system_prompt_wigh_images is None:
            system_prompt_wigh_images = open(DEFAULT_WITH_IMAGES_PATH, 'r').read()
        self.system_prompt_text_only = system_prompt_text_only
        self.system_prompt_wigh_images = system_prompt_wigh_images

    @torch.inference_mode()
    def _generate_from_messages(
            self,
            batch_messages: Sequence[Sequence[dict]],
            max_new_tokens: Optional[int] = None,
            **kwargs) -> List[str]:
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens_default

        inputs = self.processor.apply_chat_template(
            batch_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        inputs.pop("token_type_ids", None)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            **kwargs)

        input_ids = inputs["input_ids"]
        tokenizer = self.processor.tokenizer
        outputs: List[str] = []

        # Decode only the new tokens after each input sequence
        for in_ids, out_ids in zip(input_ids, generated_ids):
            trimmed_ids = out_ids[len(in_ids):]
            text = tokenizer.decode(
                trimmed_ids.tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            outputs.append(text.strip())

        return outputs

    def rewrite_text_batch(
            self,
            prompts: Sequence[str],
            max_new_tokens: Optional[int] = None,
            top_p=0.6,
            top_k=40,
            temperature=0.5,
            repetition_penalty=1.0,
            **kwargs) -> List[str]:
        """
        Rewrite a batch of text-only prompts into detailed prompts.
        """
        batch_messages = []
        for p in prompts:
            conv = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": self.system_prompt_text_only},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": p},
                    ],
                },
            ]
            batch_messages.append(conv)

        return self._generate_from_messages(
            batch_messages,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            **kwargs,
        )

    def rewrite_edit_batch(
            self,
            image: Sequence[Union[str, 'Image.Image', Sequence[Union[str, 'Image.Image']]]],
            edit_requests: Sequence[str],
            max_new_tokens: Optional[int] = None,
            top_p=0.5,
            top_k=20,
            temperature=0.4,
            repetition_penalty=1.0,
            **kwargs) -> List[str]:
        """
        Rewrite a batch of (image, edit-request) pairs into concise edit instructions.
        """
        if len(image) != len(edit_requests):
            raise ValueError("image and edit_requests must have the same length")

        batch_messages = []
        for imgs, req in zip(image, edit_requests):
            if isinstance(imgs, (str, Image.Image)):
                img_list = [imgs]
            else:
                img_list = list(imgs)

            user_content = []
            for im in img_list:
                user_content.append({"type": "image", "image": im})
            user_content.append({"type": "text", "text": req})

            conv = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": self.system_prompt_wigh_images},
                    ],
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ]
            batch_messages.append(conv)

        return self._generate_from_messages(
            batch_messages,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            **kwargs,
        )
