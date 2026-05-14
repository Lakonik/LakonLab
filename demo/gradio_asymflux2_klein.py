import math
import argparse

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import gradio as gr
from mmcv.runner import set_random_seed
from lakonlab.models.architectures import OklabColorEncoder
from lakonlab.models.diffusions.schedulers import FlowAdapterScheduler
from lakonlab.pipelines.pipeline_pixelflux2_klein import PixelFlux2KleinPipeline
from lakonlab.ui.gradio.create_text_to_img import create_interface_text_to_img
from lakonlab.pipelines.prompt_rewriters.qwen3_vl import Qwen3VLPromptRewriter


DEFAULT_PROMPT = 'Restored color photo from the 1900s. A middle-aged man with cybernetic metal hands is sitting on an old wooden chair and reading the newspaper. The newspaper has the prominent headline "AsymFLOW RELEASED" in large bold font. Close-up shot focusing on the newspaper.'
DEFAULT_NEG_PROMPT = 'Low quality, worst quality, blurry, deformed, bad anatomy, unclear text'

SYSTEM_PROMPT_TEXT_ONLY_PATH = 'lakonlab/pipelines/prompt_rewriters/system_prompts/default_text_only.txt'


def parse_args():
    parser = argparse.ArgumentParser(description='AsymFLUX.2-klein Gradio Demo')
    parser.add_argument('--share', action='store_true', help='Enable Gradio sharing')
    parser.add_argument('--use-rewriter', action='store_true', help='Enable prompt rewriter UI and model loading')
    return parser.parse_args()


def main():
    args = parse_args()

    pipe = PixelFlux2KleinPipeline.from_pretrained(
        'black-forest-labs/FLUX.2-klein-base-9B',
        vae=OklabColorEncoder(
            use_affine_norm=True,
            mean=(0.56, 0.0, 0.01),
            std=0.16),
        scheduler=FlowAdapterScheduler(
            shift=17.0,
            use_dynamic_shifting=True,
            base_seq_len=1024 ** 2,
            max_seq_len=2048 ** 2,
            base_logshift=math.log(17.0),
            max_logshift=math.log(34.0),
            dynamic_shifting_type='sqrt',
            base_scheduler='UniPCMultistep'),
        torch_dtype=torch.bfloat16)
    pipe.load_lakonlab_adapter(
        'Lakonik/AsymFLUX.2-klein-9B',
        target_module_name='transformer')
    pipe = pipe.to('cuda')

    if args.use_rewriter:
        prompt_rewriter = Qwen3VLPromptRewriter(
            device_map="cuda",
            system_prompt_text_only=open(SYSTEM_PROMPT_TEXT_ONLY_PATH, 'r').read(),
            max_new_tokens_default=512,
        )

        def run_rewrite_prompt(seed, prompt, rewrite_prompt, progress=gr.Progress(track_tqdm=True)):
            if rewrite_prompt:
                set_random_seed(seed)
                progress(0.05, desc="Rewriting prompt...")
                final_prompt = prompt_rewriter.rewrite_text_batch([prompt])[0]
                return final_prompt, None
            else:
                return '', None

    else:
        run_rewrite_prompt = None

    def generate(
            seed, prompt, negative_prompt, width, height, steps, guidance_scale,
            rewrite_prompt=False, rewritten_prompt='',
            progress=gr.Progress(track_tqdm=True)):
        return pipe(
            prompt=rewritten_prompt if rewrite_prompt else prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),
        ).images[0]

    with gr.Blocks(analytics_enabled=False,
                   title='AsymFLUX.2-klein Demo',
                   css_paths='lakonlab/ui/gradio/style.css'
                   ) as demo:

        md_txt = '# AsymFLUX.2-klein Demo\n\n' \
                 'Pixel-space text-to-image generation demo of the paper [Asymmetric Flow Models](https://arxiv.org/abs/2605.12964). ' \
                 '**Base model:** [FLUX.2 klein Base 9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B). **Code:** [https://github.com/Lakonik/LakonLab](https://github.com/Lakonik/LakonLab).\n' \
                 '<br> Use and distribution of this app are governed by the [FLUX Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B/blob/main/LICENSE.md).'
        gr.Markdown(md_txt)

        create_interface_text_to_img(
            generate,
            prompt=DEFAULT_PROMPT,
            negative_prompt=DEFAULT_NEG_PROMPT,
            steps=38,
            min_steps=4,
            max_steps=50,
            guidance_scale=4.0,
            height=1280,
            width=960,
            create_negative_prompt=True,
            create_prompt_rewrite=args.use_rewriter,
            args=['last_seed', 'prompt', 'negative_prompt', 'width', 'height', 'steps', 'guidance_scale']
                 + (['rewrite_prompt', 'rewritten_prompt'] if args.use_rewriter else []),
            rewrite_prompt_api=run_rewrite_prompt,
            rewrite_prompt_args=['last_seed', 'prompt', 'rewrite_prompt'])
        demo.queue().launch(share=args.share)


if __name__ == "__main__":
    main()
