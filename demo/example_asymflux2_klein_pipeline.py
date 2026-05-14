import math
import torch
from lakonlab.models.architectures import OklabColorEncoder
from lakonlab.models.diffusions.schedulers import FlowAdapterScheduler
from lakonlab.pipelines.pipeline_pixelflux2_klein import PixelFlux2KleinPipeline

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
adapter_name = pipe.load_lakonlab_adapter(  # you may later call `pipe.set_adapters([adapter_name, ...])` to combine other adapters (e.g., style LoRAs)
    'Lakonik/AsymFLUX.2-klein-9B',
    target_module_name='transformer')
pipe = pipe.to('cuda')

# Text-to-image generation example
prompt = 'Restored color photo from the 1900s. A middle-aged man with cybernetic metal hands is sitting on an old wooden chair and reading the newspaper. The newspaper has the prominent headline "AsymFLOW RELEASED" in large bold font. Close-up shot focusing on the newspaper.'
neg_prompt = 'Low quality, worst quality, blurry, deformed, bad anatomy, unclear text'
out = pipe(
    prompt=prompt,
    negative_prompt=neg_prompt,
    width=960,
    height=1280,
    num_inference_steps=38,
    guidance_scale=4.0,
    generator=torch.Generator().manual_seed(42),
).images[0]
out.save('asymflux2_klein.png')
