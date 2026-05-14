import argparse
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import gradio as gr
from diffusers import FluxPipeline
from lakonlab.models.diffusions.schedulers.flow_map_sde import FlowMapSDEScheduler
from lakonlab.ui.gradio.create_text_to_img import create_interface_text_to_img
from lakonlab.pipelines.pipeline_piflux import PiFluxPipeline


def parse_args():
    parser = argparse.ArgumentParser(description='pi-FLUX Gradio Demo')
    parser.add_argument('--share', action='store_true', help='Enable Gradio sharing')
    return parser.parse_args()


DEFAULT_PROMPT = ('A portrait photo of a kangaroo wearing an orange hoodie and blue sunglasses standing in front of '
                  'the Sydney Opera House holding a sign on the chest that says "Welcome Friends"')


def main():
    args = parse_args()

    base_pipe = FluxPipeline.from_pretrained(
        'black-forest-labs/FLUX.1-dev',
        torch_dtype=torch.bfloat16)
    base_pipe = base_pipe.to('cuda')
    scheduler = FlowMapSDEScheduler.from_config(
        base_pipe.scheduler.config, shift=3.2, use_dynamic_shifting=False, final_step_size_scale=0.5)

    pipe_4nfe = PiFluxPipeline(
        transformer=base_pipe.transformer,
        vae=base_pipe.vae,
        text_encoder=base_pipe.text_encoder,
        text_encoder_2=base_pipe.text_encoder_2,
        tokenizer=base_pipe.tokenizer,
        tokenizer_2=base_pipe.tokenizer_2,
        scheduler=scheduler)
    pipe_4nfe.load_lakonlab_adapter(
        'Lakonik/pi-FLUX.1',
        subfolder='gmflux_k8_piid_4step',
        target_module_name='transformer')

    pipe_8nfe = PiFluxPipeline(
        transformer=base_pipe.transformer,
        vae=base_pipe.vae,
        text_encoder=base_pipe.text_encoder,
        text_encoder_2=base_pipe.text_encoder_2,
        tokenizer=base_pipe.tokenizer,
        tokenizer_2=base_pipe.tokenizer_2,
        scheduler=scheduler)
    pipe_8nfe.load_lakonlab_adapter(
        'Lakonik/pi-FLUX.1',
        subfolder='gmflux_k8_piid_8step',
        target_module_name='transformer')

    del base_pipe

    def generate(
            seed, prompt, width, height, steps,
            progress=gr.Progress(track_tqdm=True)):
        assert steps in [4, 8], 'Only 4 or 8 steps are supported.'
        pipe = pipe_4nfe if steps == 4 else pipe_8nfe
        return pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            generator=torch.Generator().manual_seed(seed),
        ).images[0]

    with gr.Blocks(analytics_enabled=False,
                   title='pi-FLUX Demo',
                   css_paths='lakonlab/ui/gradio/style.css'
                   ) as demo:

        md_txt = '# pi-FLUX Demo\n\n' \
                 'Official demo of the paper [pi-Flow: Policy-Based Few-Step Generation via Imitation Distillation](https://arxiv.org/abs/2510.14974). ' \
                 '**Base model:** [FLUX.1 dev](https://huggingface.co/black-forest-labs/FLUX.1-dev). **Fast policy:** GMFlow. **Code:** [https://github.com/Lakonik/piFlow](https://github.com/Lakonik/piFlow).\n' \
                 '<br> Use and distribution of this app are governed by the [FLUX.1 [dev] Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md).'
        gr.Markdown(md_txt)

        create_interface_text_to_img(
            generate,
            prompt=DEFAULT_PROMPT,
            steps=4, min_steps=4, max_steps=8, steps_slider_step=4, guidance_scale=None,
            args=['last_seed', 'prompt', 'width', 'height', 'steps'])
        demo.queue().launch(share=args.share)


if __name__ == "__main__":
    main()
