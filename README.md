# LakonLab: Official Codebase for AsymFlow, pi-Flow, and GMFlow

Official PyTorch implementation of the papers:

- **Asymmetric Flow Models [[README](docs/AsymFlow.md)]**
    <br>
    arXiv 2026
    <br>
    [Hansheng Chen](https://lakonik.github.io/),
    [Jan Ackermann](https://janackermann.info/),
    [Minseo Kim](https://soniaminseokim.github.io/),
    [Gordon Wetzstein](http://web.stanford.edu/~gordonwz/), 
    [Leonidas Guibas](https://geometry.stanford.edu/?member=guibas)<br>
    Stanford University
    <br>
    [Project Page](https://hanshengchen.com/asymflow) | [arXiv](https://arxiv.org/abs/2605.12964) | [ComfyUI](https://github.com/Lakonik/ComfyUI-piFlow) | [AsymFLUX.2 klein Demo🤗](https://huggingface.co/spaces/Lakonik/AsymFLUX.2-klein)
  
    <img src="docs/assets/asymflow/asymflow_teaser.jpg" width="250" alt=""/>    

- **pi-Flow: Policy-Based Few-Step Generation via Imitation Distillation [[README](docs/piFlow.md)]**
    <br>
    In ICLR 2026
    <br>
    [Hansheng Chen](https://lakonik.github.io/)<sup>1</sup>, 
    [Kai Zhang](https://kai-46.github.io/website/)<sup>2</sup>,
    [Hao Tan](https://research.adobe.com/person/hao-tan/)<sup>2</sup>,
    [Leonidas Guibas](https://geometry.stanford.edu/?member=guibas)<sup>1</sup>,
    [Gordon Wetzstein](http://web.stanford.edu/~gordonwz/)<sup>1</sup>, 
    [Sai Bi](https://sai-bi.github.io/)<sup>2</sup><br>
    <sup>1</sup>Stanford University, <sup>2</sup>Adobe Research
    <br>
    [arXiv](https://arxiv.org/abs/2510.14974) | [ComfyUI](https://github.com/Lakonik/ComfyUI-piFlow) | [pi-Qwen Demo🤗](https://huggingface.co/spaces/Lakonik/pi-Qwen) | [pi-FLUX Demo🤗](https://huggingface.co/spaces/Lakonik/pi-FLUX.1) | [pi-FLUX.2 Demo🤗](https://huggingface.co/spaces/Lakonik/pi-FLUX.2)

    <img src="docs/assets/piflow/piflow_teaser.jpg" width="250" alt=""/>

- **Gaussian Mixture Flow Matching Models [[README](docs/GMFlow.md)]**
    <br>
    In ICML 2025
    <br>
    [Hansheng Chen](https://lakonik.github.io/)<sup>1</sup>, 
    [Kai Zhang](https://kai-46.github.io/website/)<sup>2</sup>,
    [Hao Tan](https://research.adobe.com/person/hao-tan/)<sup>2</sup>,
    [Zexiang Xu](https://zexiangxu.github.io/)<sup>3</sup>, 
    [Fujun Luan](https://research.adobe.com/person/fujun/)<sup>2</sup>,
    [Leonidas Guibas](https://geometry.stanford.edu/?member=guibas)<sup>1</sup>,
    [Gordon Wetzstein](http://web.stanford.edu/~gordonwz/)<sup>1</sup>, 
    [Sai Bi](https://sai-bi.github.io/)<sup>2</sup><br>
    <sup>1</sup>Stanford University, <sup>2</sup>Adobe Research, <sup>3</sup>Hillbot
    <br>
    [arXiv](https://arxiv.org/abs/2504.05304)
    
    <img src="docs/assets/gmflow/gmdit.png" width="250" alt=""/>
    <br>
    <img src="docs/assets/gmflow/gmdit_results.png" width="250" alt=""/>

## 🔥News

- [May 20, 2026] AsymFlow is now supported in our [ComfyUI extension](https://github.com/Lakonik/ComfyUI-piFlow).

- [May 14, 2026] [AsymFlow](docs/AsymFlow.md) is released.

- [Dec 12, 2025] pi-FLUX.2 is now available for 4-step image generation and editing. Check out the [pi-FLUX.2 Demo🤗](https://huggingface.co/spaces/Lakonik/pi-FLUX.2). Please re-install the latest version of LakonLab (this repository) to use pi-FLUX.2.

- [Nov 7, 2025] [ComfyUI-piFlow](https://github.com/Lakonik/ComfyUI-piFlow) is now available. Supports 4-step sampling of Qwen-Image and Flux.1 dev using 8-bit models on a single consumer-grade GPU, powered by [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

## Installation

The code has been tested in the following environment:

- Linux (tested on Ubuntu 20 and above)
- [PyTorch](https://pytorch.org/get-started/previous-versions/) 2.6+

With the above prerequisites, run `pip install -e . --no-build-isolation` from the repository root to install the LakonLab codebase and its dependencies.

An example of installation commands is shown below:

```bash
# Move to this repository (the folder with setup.py) after cloning
cd <PATH_TO_YOUR_LOCAL_REPO>

# Create uv environment
uv venv --python 3.10
source .venv/bin/activate

# Install Pytorch. Goto https://pytorch.org/get-started/previous-versions/ to select the appropriate version
uv pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128

# Install LakonLab in editable mode
uv pip install -e . --no-build-isolation
```

Additional notes:
<br>
To access FLUX models, please accept the [FLUX.2 klein Base 9B conditions](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) and [FLUX.1 dev conditions](https://huggingface.co/black-forest-labs/FLUX.1-dev), and then run `hf auth login` to login with your HuggingFace account.

## Codebase

<img src="docs/assets/lakonlab.png" width="200"  alt=""/>

**LakonLab** is a high-performance codebase for experimenting with large diffusion models. Key features of LakonLab include:
- **Performance optimizations**: Seamless switching between DDP, FSDP, and FSDP2, all supporting gradient accumulation and mixed precision.
- **Weight tying**: For LoRA fine-tuning, the base weights of the teacher, student, and EMA models are tied, sharing the same underlying memory. This is compatible with DDP and FSDP.
- **Advanced flow solvers**:
  - [FlowSDEScheduler](lakonlab/models/diffusions/schedulers/flow_sde.py): Generic flow SDE solver with an adjustable [diffusion coefficient](https://arxiv.org/pdf/2306.02063). `h=0` corresponds to a flow ODE; `h=1` corresponds to a standard flow SDE; `h='inf'` corresponds to the re-noising sampler in the original [consistency models](https://arxiv.org/pdf/2303.01469). Powers the GM-SDE solver in GMFlow.
  - [FlowMapSDEScheduler](lakonlab/models/diffusions/schedulers/flow_map_sde.py): Generic flow SDE solver for few-step flow map models, similar to above.
- **Storage backends**: Most I/O operations (e.g., dataloaders, checkpoint I/O) support both local filesystems and AWS S3. In addition, model checkpoints can be loaded from HuggingFace (link format `huggingface://<HF_REPO_NAME>/<PATH_TO_MODEL>`) and HTTP/HTTPS URLs directly.
- **Streamlined training and evaluation**: Supports online evaluation using common [metrics](lakonlab/evaluation/metrics.py), including FID, KID, IS, Precision, Recall, CLIP similarity, [VQAScore](https://github.com/linzhiqiu/t2v_metrics), [HPSv2](https://github.com/tgxs002/HPSv2), and [HPSv3](https://github.com/MizzenAI/HPSv3). Supports exporting results to offline evaluators, including [HPSv3 Benchmark](https://github.com/MizzenAI/HPSv3), [DPG-Bench](https://github.com/TencentQQGYLab/ELLA) and [GenEval](https://github.com/djghosh13/geneval).
- **3rd-party model inference reproduction**:
  
  - ImageNet 256x256 models with [ADM evaluation](https://github.com/openai/guided-diffusion/tree/main/evaluations):

    | Model                                                             | FID (reproduced) | FID (official) |
    |-------------------------------------------------------------------|:----------------:|:--------------:|
    | [SiT-XL/2](configs/misc/sit_imagenet_test.py)                     |       2.05       |      2.06      |
    | [JiT-H/16](configs/misc/jit_h_16_imagenet_test.py)                |       1.90       |       -        |
    | [DiT-XL RAE (unguided)](configs/misc/rae_dinov2_imagenet_test.py) |       1.50       |      1.51      |
    | [REPA-XL/2](configs/misc/repa_imagenet_test.py)                   |       1.38       |      1.42      |
    | [REPA-E-XL VAVAE](configs/misc/repae_imagenet_test.py)            |       1.12       |      1.12      |

    Note: <br>
    We use BF16 inference for all models except RAE. <br>
    Original JiT paper uses its own evaluation protocol that differs from ADM evaluation.

  - Text-to-image models:

    See examples in [configs/misc](configs/misc) and [lakonlab/models/architectures/diffusers](lakonlab/models/architectures/diffusers).

LakonLab uses the configuration system and code structure from [MMCV](https://github.com/open-mmlab/mmcv).

## Citation
```
@article{asymflow,
  title={Asymmetric Flow Models},
  author={Hansheng Chen and Jan Ackermann and Minseo Kim and Gordon Wetzstein and Leonidas Guibas},
  url={https://arxiv.org/abs/2605.12964},
  journal={arXiv preprint arXiv:2605.12964},
  year={2026},
}

@article{piflow,
  title={pi-Flow: Policy-Based Few-Step Generation via Imitation Distillation}, 
  author={Hansheng Chen and Kai Zhang and Hao Tan and Leonidas Guibas and Gordon Wetzstein and Sai Bi},
  url={https://arxiv.org/abs/2510.14974}, 
  journal={arXiv preprint arXiv:2510.14974},
  year={2025},
}

@article{gmflow,
  title={Gaussian Mixture Flow Matching Models},
  author={Hansheng Chen and Kai Zhang and Hao Tan and Zexiang Xu and Fujun Luan and Leonidas Guibas and Gordon Wetzstein and Sai Bi},
  url={https://arxiv.org/abs/2504.05304}, 
  journal={arXiv preprint arXiv:2504.05304},
  year={2025},
}
```
