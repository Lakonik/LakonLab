import argparse
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import gradio as gr
from lakonlab.models.diffusions.schedulers.flow_map_sde import FlowMapSDEScheduler
from lakonlab.ui.gradio.create_text_to_img import create_interface_text_to_img
from lakonlab.pipelines.pipeline_piqwen import PiQwenImagePipeline


def parse_args():
    parser = argparse.ArgumentParser(description='pi-Qwen Gradio Demo')
    parser.add_argument('--share', action='store_true', help='Enable Gradio sharing')
    return parser.parse_args()


DEFAULT_PROMPT = ('Photo of a coffee shop entrance featuring a chalkboard sign reading "π-Qwen Coffee 😊 $2 per cup," '
                  'with a neon light beside it displaying "π-通义千问". Next to it hangs a poster showing a beautiful '
                  'Chinese woman, and beneath the poster is written "e≈2.71828-18284-59045-23536-02874-71352".')


def main():
    args = parse_args()

    pipe = PiQwenImagePipeline.from_pretrained(
        'Qwen/Qwen-Image',
        torch_dtype=torch.bfloat16)
    pipe.load_lakonlab_adapter(
        'Lakonik/pi-Qwen-Image',
        subfolder='gmqwen_k8_piid_4step',
        target_module_name='transformer')
    pipe.scheduler = FlowMapSDEScheduler.from_config(  # use fixed shift=3.2
        pipe.scheduler.config, shift=3.2, use_dynamic_shifting=False, final_step_size_scale=0.5)
    pipe = pipe.to('cuda')

    def generate(
            seed, prompt, width, height, steps,
            progress=gr.Progress(track_tqdm=True)):
        return pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            generator=torch.Generator().manual_seed(seed),
        ).images[0]

    with gr.Blocks(analytics_enabled=False,
                   title='pi-Qwen Demo',
                   css_paths='lakonlab/ui/gradio/style.css'
                   ) as demo:

        md_txt = '# pi-Qwen Demo\n\n' \
                 'Official demo of the paper [pi-Flow: Policy-Based Few-Step Generation via Imitation Distillation](https://arxiv.org/abs/2510.14974). ' \
                 '**Base model:** [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image). **Fast policy:** GMFlow. **Code:** [https://github.com/Lakonik/piFlow](https://github.com/Lakonik/piFlow).'
        gr.Markdown(md_txt)

        create_interface_text_to_img(
            generate,
            prompt=DEFAULT_PROMPT,
            steps=4, guidance_scale=None,
            args=['last_seed', 'prompt', 'width', 'height', 'steps'])
        demo.queue().launch(share=args.share)


if __name__ == "__main__":
    main()
