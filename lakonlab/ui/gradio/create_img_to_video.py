import gradio as gr
from .shared_opts import create_base_opts, create_generate_bar, set_seed, create_prompt_opts


def create_interface_img_to_video(
        api, prompt='', seed=42, steps=4, min_steps=2, max_steps=4, steps_slider_step=2,
        aspect_ratios=('16:9', '1:1', '9:16'), aspect_ratio='16:9',
        megapixels=1.0, min_megapixels=0.25, max_megapixels=1.0, megapixels_slider_step=0.05,
        duration=5.0, min_duration=5.0, max_duration=15.0, duration_slider_step=0.5,
        guidance_scale=None, temperature=None, remove_decoder=None, examples=None, api_name='img_to_video',
        args=['last_seed', 'prompt', 'in_image', 'in_last_image', 'aspect_ratio', 'megapixels', 'duration', 'steps']):
    var_dict = dict()
    with gr.Blocks(analytics_enabled=False) as interface:
        with gr.Row():
            with gr.Column():
                with gr.Accordion('Keyframes (optional)', open=True, elem_classes=['custom-spacing']):
                    with gr.Row():
                        var_dict['in_image'] = gr.Image(label='First frame', type='pil', image_mode='RGB')
                        var_dict['in_last_image'] = gr.Image(label='Last frame', type='pil', image_mode='RGB')

                create_prompt_opts(var_dict, create_negative_prompt=False, prompt=prompt)

                with gr.Row(variant='compact', elem_classes=['custom-spacing']):
                    var_dict['duration'] = gr.Slider(
                        min_duration, max_duration, value=duration, step=duration_slider_step,
                        label='Duration (s)', scale=2, elem_classes=['force-hide-container'])
                    var_dict['megapixels'] = gr.Slider(
                        min_megapixels, max_megapixels, value=megapixels, step=megapixels_slider_step,
                        label='Megapixels', scale=2, elem_classes=['force-hide-container'])
                    var_dict['aspect_ratio'] = gr.Dropdown(
                        list(aspect_ratios), value=aspect_ratio, label='Aspect ratio', scale=1,
                        elem_classes=['force-hide-container', 'compact-dropdown'])

                create_generate_bar(var_dict, text='Generate', seed=seed)

                create_base_opts(
                    var_dict,
                    steps=steps,
                    min_steps=min_steps,
                    max_steps=max_steps,
                    steps_slider_step=steps_slider_step,
                    guidance_scale=guidance_scale,
                    temperature=temperature,
                    remove_decoder=remove_decoder)

            with gr.Column():
                var_dict['output_video'] = gr.Video(
                    label='Output video', interactive=False, autoplay=True, elem_classes=['vh-img', 'vh-img-1000'])
                if examples:  # dicts of a 'label' and input values keyed like var_dict
                    keys = [k for k in examples[0] if k != 'label']
                    gr.Examples(
                        [[example['label']] + [example[k] for k in keys] for example in examples],
                        [gr.Textbox(label='Example', visible=False)] + [var_dict[k] for k in keys],
                        label='Examples')

        var_dict['run_btn'].click(
            fn=set_seed,
            inputs=var_dict['seed'],
            outputs=var_dict['last_seed'],
            show_progress=False,
            api_visibility='private'
        ).success(
            fn=api,
            inputs=[var_dict[arg] for arg in args],
            outputs=var_dict['output_video'],
            concurrency_id='default_group', api_name=api_name
        )

    return interface, var_dict
