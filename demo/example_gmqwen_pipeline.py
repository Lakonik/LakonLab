import torch
from lakonlab.models.diffusions.schedulers import FlowMapSDEScheduler
from lakonlab.pipelines.pipeline_piqwen import PiQwenImagePipeline

pipe = PiQwenImagePipeline.from_pretrained(
    'Qwen/Qwen-Image',
    torch_dtype=torch.bfloat16)
adapter_name = pipe.load_lakonlab_adapter(  # you may later call `pipe.set_adapters([adapter_name, ...])` to combine other adapters (e.g., style LoRAs)
    'Lakonik/pi-Qwen-Image',
    subfolder='gmqwen_k8_piid_4step',
    target_module_name='transformer')
pipe.scheduler = FlowMapSDEScheduler.from_config(  # use fixed shift=3.2
    pipe.scheduler.config, shift=3.2, use_dynamic_shifting=False, final_step_size_scale=0.5)
pipe = pipe.to('cuda')

out = pipe(
    prompt='Photo of a coffee shop entrance featuring a chalkboard sign reading "π-Qwen Coffee 😊 $2 per cup," with a neon '
           'light beside it displaying "π-通义千问". Next to it hangs a poster showing a beautiful Chinese woman, '
           'and beneath the poster is written "e≈2.71828-18284-59045-23536-02874-71352".',
    width=1920,
    height=1080,
    num_inference_steps=4,
    generator=torch.Generator().manual_seed(42),
).images[0]
out.save('gmqwen_4nfe.png')
