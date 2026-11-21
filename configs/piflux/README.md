## Distilling FLUX

### Before Training: Data Preparation

We release the [prompt-only dataset](https://huggingface.co/datasets/Lakonik/t2i-prompts-3m) to reproduce the data-free training (without real images) in the paper. It is highly recommended to preprocess the prompts into prompt embeddings using the text encoder before training. Run the following command to preprocess the prompts using DDP on 1 node with 8 GPUs:
```bash
torchrun --nnodes=1 --nproc_per_node=8 tools/datasets/cache_image_prompt_data.py configs/piflux/gmflux_k8_datafree_piid_4step_32gpus.py --text-encoder configs/piflux/_text_encoder.py --max-size 768128 --launcher pytorch --diff_seed
```
By default, the preprocessed data will be saved to `data/t2i_prompts_3m/preproc_flux/`, which requires 1.1TB of storage space. You can change the `data_root` in the [config file](_data_trainval.py) to save the data to a different location (AWS S3 URLs are supported).

If you choose not to preprocess the prompts, please modify the [training config file](gmflux_k8_datafree_piid_4step_32gpus.py) to add `'./_text_encoder.py'` to the `_base_` list, which will load the text encoder during training.

### Training

Run the following command to train the GM-FLUX model under the data-free setting using DDP on 4 nodes with 8 GPUs each:
```bash
torchrun --nnodes=4 --nproc_per_node=8 --node_rank=<NODE_RANK> --master_addr=<MASTER_ADDR> --master_port=<MASTER_PORT> tools/train.py configs/piflux/gmflux_k8_datafree_piid_4step_32gpus.py --launcher pytorch --diff_seed
```

The above config specifies a training batch size of 2 images per GPU, and the gradients are accumulated across 4 student steps, effectively making the batch size 8 images per GPU. This requires ~65GB of VRAM per GPU. FSDP can be enabled to reduce VRAM usage to ~16GB per GPU. To use FSDP, modify the [training config file](gmflux_k8_datafree_piid_4step_32gpus.py) to replace `'./_ddp_train.py'` with `'./_fsdp_train.py'` in the `_base_` list.

32 GPUs are required to reproduce the total batch size of 256 in the paper. If you do not care about exact reproduction, using a smaller batch size with fewer GPUs is totally fine.

Note that pi-FLUX adopts scheduled trajectory mixing during training. Early checkpoints will produce blurry samples, which is the intended behavior. If you want to get high-quality samples in fewer training iterations, modify the `num_decay_iters` in the [training config file](gmflux_k8_datafree_piid_4step_32gpus.py) to a smaller value (`num_decay_iters=0` disables trajectory mixing).

### Evaluation (COCO and HPSv2)

Before evaluation, run the following command to preprocess the evaluation prompts (similar to the [Before Training: Data Preparation](#before-training-data-preparation) step):
```bash
torchrun --nnodes=1 --nproc_per_node=8 tools/datasets/cache_image_prompt_data.py configs/piflux/gmflux_k8_datafree_piid_4step_test.py --text-encoder configs/piflux/_text_encoder.py --launcher pytorch --diff_seed
```

Run the following command to evaluate a pretrained model (downloaded automatically) using DDP on 1 node with 8 GPUs:
```bash
torchrun --nnodes=1 --nproc_per_node=8 tools/test.py <PATH_TO_CONFIG> --launcher pytorch --diff_seed
```
where `<PATH_TO_CONFIG>` can be one of the following:
- `configs/piflux/gmflux_k8_piid_4step_test.py` (4-NFE GM-FLUX)
- `configs/piflux/gmflux_k8_datafree_piid_4step_test.py` (4-NFE GM-FLUX, data-free)
- `configs/piflux/dxflux_n10_piid_4step_test.py` (4-NFE DX-FLUX)
- `configs/piflux/gmflux_k8_piid_8step_test.py` (8-NFE GM-FLUX)

To enable FSDP evaluation, please add `'./_fsdp_test.py'` to the `_base_` list in the corresponding config file.

To evaluate a custom checkpoint instead of the official model, we provide two options.

*Option A.*

Run the export script to convert the checkpoint to diffusers safetensors:
```bash
python tools/checkpoints/export_piflow_to_diffusers.py <PATH_TO_CONFIG> --ckpt <PATH_TO_CKPT> --out-dir <OUTPUT_DIR>
```
Then, modify the test config file to set `pretrained_adapter` to `<OUTPUT_DIR>/diffusion_pytorch_model.safetensors`. Finally, run the evaluation command as above.

*Option B.*

Copy the `use_lora`, `lora_target_modules`, and `lora_rank` settings from the train config file to the test config file, remove the `pretrained_adapter` line, and then run the evaluation command with an additional `--ckpt <PATH_TO_CKPT>` argument:
```bash
torchrun --nnodes=1 --nproc_per_node=8 tools/test.py <PATH_TO_CONFIG> --ckpt <PATH_TO_CKPT> --launcher pytorch --diff_seed
```
