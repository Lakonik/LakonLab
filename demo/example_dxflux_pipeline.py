import torch
from lakonlab.models.diffusions.schedulers import FlowMapSDEScheduler
from lakonlab.pipelines.pipeline_piflux import PiFluxPipeline

pipe = PiFluxPipeline.from_pretrained(
    'black-forest-labs/FLUX.1-dev',
    policy_type='DX',
    policy_kwargs=dict(
        segment_size=1 / 3.5,  # 1 / (nfe - 1 + final_step_size_scale)
        shift=3.2),
    torch_dtype=torch.bfloat16)
adapter_name = pipe.load_lakonlab_adapter(  # you may later call `pipe.set_adapters([adapter_name, ...])` to combine other adapters (e.g., style LoRAs)
    'Lakonik/pi-FLUX.1',
    subfolder='dxflux_n10_piid_4step',
    target_module_name='transformer')
pipe.scheduler = FlowMapSDEScheduler.from_config(  # use fixed shift=3.2
    pipe.scheduler.config, shift=3.2, use_dynamic_shifting=False, final_step_size_scale=0.5)
pipe = pipe.to('cuda')

out = pipe(
    prompt='A portrait photo of a kangaroo wearing an orange hoodie and blue sunglasses standing in front of the Sydney Opera House holding a sign on the chest that says "Welcome Friends"',
    width=1360,
    height=768,
    num_inference_steps=4,
    generator=torch.Generator().manual_seed(42),
).images[0]
out.save('dxflux_4nfe.png')
