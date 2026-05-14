import gradio as gr
from .shared_opts import create_base_opts, create_generate_bar, set_seed, create_prompt_opts, create_image_size_bar


def create_interface_text_to_img(
        api, prompt='', negative_prompt='', seed=42, steps=32, min_steps=4, max_steps=50, steps_slider_step=1,
        height=768, width=1360, hw_slider_step=16,
        guidance_scale=None, temperature=None, api_name='text_to_img',
        create_negative_prompt=False,
        create_prompt_rewrite=False, rewrite_prompt=False,
        args=['last_seed', 'prompt', 'width', 'height', 'steps', 'guidance_scale'],
        rewrite_prompt_api=None, rewrite_prompt_args=['last_seed', 'prompt', 'rewrite_prompt']):
    var_dict = dict()
    with gr.Blocks(analytics_enabled=False) as interface:
        var_dict['output_image'] = gr.Image(
            type='pil', image_mode='RGB', label='Output image', interactive=False, elem_classes=['vh-img', 'vh-img-700'])
        create_prompt_opts(
            var_dict, create_negative_prompt=create_negative_prompt, prompt=prompt, negatove_prompt=negative_prompt)
        if create_prompt_rewrite:
            var_dict['rewrite_prompt'] = gr.Checkbox(
                label='Rewrite prompt', value=rewrite_prompt, container=False)
            with gr.Accordion("Rewritten prompt", open=False, elem_classes=['custom-spacing']):
                var_dict['rewritten_prompt'] = gr.Textbox(
                    lines=4, interactive=False, show_label=False, container=False)
        with gr.Column(variant='compact', elem_classes=['custom-spacing']):
            create_image_size_bar(
                var_dict, height=height, width=width, hw_slider_step=hw_slider_step)
        create_generate_bar(var_dict, text='Generate', seed=seed)
        create_base_opts(
            var_dict,
            steps=steps,
            min_steps=min_steps,
            max_steps=max_steps,
            steps_slider_step=steps_slider_step,
            guidance_scale=guidance_scale,
            temperature=temperature)

        if create_prompt_rewrite:
            assert rewrite_prompt_api is not None
            var_dict['run_btn'].click(
                fn=set_seed,
                inputs=var_dict['seed'],
                outputs=var_dict['last_seed'],
                show_progress=False,
                api_name=False
            ).success(
                fn=rewrite_prompt_api,
                inputs=[var_dict[arg] for arg in rewrite_prompt_args],
                outputs=[var_dict['rewritten_prompt'], var_dict['output_image']],
                concurrency_id='default_group', api_name=False
            ).success(
                fn=api,
                inputs=[var_dict[arg] for arg in args],
                outputs=var_dict['output_image'],
                concurrency_id='default_group', api_name=api_name
            )
        else:
            var_dict['run_btn'].click(
                fn=set_seed,
                inputs=var_dict['seed'],
                outputs=var_dict['last_seed'],
                show_progress=False,
                api_name=False
            ).success(
                fn=api,
                inputs=[var_dict[arg] for arg in args],
                outputs=var_dict['output_image'],
                concurrency_id='default_group', api_name=api_name
            )

    return interface, var_dict
