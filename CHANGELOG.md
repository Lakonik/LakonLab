# Changelog

## [0.2.0] - 2026-05-12

### Added
- AsymFlow release.
- JiT, RAE, and REPA-E model support and test configs.
- DPGBench, GenEval, and HPSv3 benchmark exporters.
- Media viewer pagination.

### Changed
- **Breaking:** Replace the MMGeneration model registry dependency with local LakonLab registries.
- **Breaking:** Move model architectures from `lakonlab.models.architecture` to `lakonlab.models.architectures`.
- Update FSDP/DDP wrappers, checkpoint loading/saving, runner utilities, and S3 retry/refresh behavior.

### Fixed
- Fixed multiple bugs and edge cases.

## [0.1.2] - 2026-01-18

### Added
- Support for loading prompt list from plain text files in `ImagePromptDataset`. Example:
  ```
  prompt_dataset_kwargs=dict(
      path='text',
      data_files='https://raw.githubusercontent.com/ModelTC/Qwen-Image-Lightning/refs/heads/main/examples/prompt_list.txt',
      split="train"),
  ```
- Support for `spawn` start method for S3 dataloader workers.

### Changed
- **Breaking:** Teacher guidance parameters are now specified in the `teacher_test_cfg` section of the configuration file. Please update your configuration files accordingly.
- Change `S3Backend` default settings:
  - Set default `AWS_REGION` to None,
  - Disable anonymous mode by default.

### Fixed
- Fix the implementation of orthogonal guidance in GaussianFlow.
- Fix a bug with guidance interval.
- Fix `tools/train.py` crashing when using `--launcher slurm`.
- Fix a bug in `tools/test.py` that caused the flag `--reuse-viz` to have no effect.
- Fix a bug when running `tools/cache_image_prompt_data.py` with `--skip-existing` flag in distributed mode.
- Fix several bugs that could cause distributed runs to hang due to I/O synchronization issues.
- Fix a bug that visualization images may not be saved in certain file systems due to unsupported filenames.
- Fix a version compatibility issue in the example GMDiT script in `configs/gmflow/README.md`.

## [0.1.1] - 2025-12-18

### Changed
- **Breaking:** Rename the pretrained HuggingFace model argument `from_pretrained` to `model_name_or_path`. Please update any custom configurations or scripts accordingly.

### Fixed
- **Important:** Fix a GMFlow batching bug introduced by the recent numerical-stability change.
- Fix loading pi-Flow adapters for TorchAO-quantized base models.
- Fix a rare bug that could cause distributed runs to hang when loading pretrained HuggingFace models.

## [0.1.0] - 2025-12-12

### Added
- FLUX.2 integration and `PiFlux2Pipeline` with example demos.
- `Qwen3VLPromptRewriter` for prompt rewriting.
- Dataset and data-caching enhancements:
  - Support for `condition_images` and `condition_latents` for image-conditioned generation.
  - Improved image rescaling.
  - `ConcatDataset` for combining multiple datasets.
  - `--skip-existing` option in `cache_image_prompt_data.py` for resuming interrupted caching.
- Support for loading quantized base models in pi-Flow pipelines.
- Support for non-local storage backends in `save_inference_weights.py`.
- Support for loading sharded safetensors from the local filesystem ([#17](https://github.com/Lakonik/piFlow/issues/17)).

### Changed
- **Breaking:** Switch all pi-Flow model schedulers to [`FlowMapSDEScheduler`](lakonlab/models/diffusions/schedulers/flow_map_sde.py), which supports both deterministic (`h = 0`) and stochastic (`h > 0`) sampling.
- Reduce peak memory usage during initialization when distilling large models with LoRA.
- Bump Gradio to `5.49.0` for web demos.
- Improve GMFlow numerical stability.

### Fixed
- Reduce the risk of CUDA OOM when saving large checkpoints.
- Fix `S3Backend.list_dir_or_file`.
- Fix errors when loading pretrained HuggingFace models under unstable network conditions.
- Fix test data split configurations.
