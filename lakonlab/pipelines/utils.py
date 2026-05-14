# Copyright (c) 2026 Hansheng Chen

import os
from typing import Union, Optional

import torch
import accelerate
import diffusers
from diffusers.models import AutoModel
from diffusers.models.modeling_utils import (
    load_state_dict,
    _LOW_CPU_MEM_USAGE_DEFAULT,
    no_init_weights,
    ContextManagers
)
from diffusers.utils import (
    SAFETENSORS_WEIGHTS_NAME,
    WEIGHTS_NAME,
    _add_variant,
    _get_model_file,
    is_accelerate_available,
    is_torch_version,
    logging,
)
from diffusers.loaders.peft import _SET_ADAPTER_SCALE_FN_MAPPING
from diffusers.quantizers import DiffusersAutoQuantizer
from diffusers.utils.torch_utils import empty_device_cache
from lakonlab.models.architectures.gmflow.gmflux import _GMFluxTransformer2DModel
from lakonlab.models.architectures.gmflow.gmqwen import _GMQwenImageTransformer2DModel
from lakonlab.models.architectures.gmflow.gmflux2 import _GMFlux2Transformer2DModel
from lakonlab.models.architectures.asymflow.asymflux2 import _AsymFlux2Transformer2DModel


LOCAL_CLASS_MAPPING = {
    "GMFluxTransformer2DModel": _GMFluxTransformer2DModel,
    "GMQwenImageTransformer2DModel": _GMQwenImageTransformer2DModel,
    "GMFlux2Transformer2DModel": _GMFlux2Transformer2DModel,
    "AsymFlux2Transformer2DModel": _AsymFlux2Transformer2DModel,
}

_SET_ADAPTER_SCALE_FN_MAPPING.update(
    _GMFluxTransformer2DModel=lambda model_cls, weights: weights,
    _GMQwenImageTransformer2DModel=lambda model_cls, weights: weights,
    _GMFlux2Transformer2DModel=lambda model_cls, weights: weights,
    _AsymFlux2Transformer2DModel=lambda model_cls, weights: weights,
)

logger = logging.get_logger(__name__)


def assign_param(module, tensor_name: str, param: torch.nn.Parameter):
    if "." in tensor_name:
        splits = tensor_name.split(".")
        for split in splits[:-1]:
            new_module = getattr(module, split)
            if new_module is None:
                raise ValueError(f"{module} has no attribute {split}.")
            module = new_module
        tensor_name = splits[-1]
    module._parameters[tensor_name] = param


class LakonLabMixin:

    def load_piflow_adapter(self, *args, **kwargs):
        logger.warning(
            "`load_piflow_adapter` is deprecated and will be removed in a future release. "
            "Use `load_lakonlab_adapter` instead.",
        )
        return self.load_lakonlab_adapter(*args, **kwargs)

    def load_lakonlab_adapter(
        self,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        target_module_name: str = "transformer",
        adapter_name: Optional[str] = None,
        **kwargs
    ):
        r"""
        Load a PiFlow adapter from a pretrained model repository into the target module.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                Can be either:

                    - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing the model weights saved
                      with [`~ModelMixin.save_pretrained`].

            target_module_name (`str`, *optional*, defaults to `"transformer"`):
                The module name in the model to load the PiFlow adapter into.
            adapter_name (`str`, *optional*):
                The name to assign to the loaded adapter. If not provided, it defaults to
                `"{target_module_name}_piflow"`.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            subfolder (`str`, *optional*, defaults to `""`):
                The subfolder location of a model file within a larger model repository on the Hub or locally.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
            variant (`str`, *optional*):
                Load weights from a specified `variant` filename such as `"fp16"` or `"ema"`. This is ignored when
                loading `from_flax`.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the `safetensors` weights are downloaded if they're available **and** if the
                `safetensors` library is installed. If set to `True`, the model is forcibly loaded from `safetensors`
                weights. If set to `False`, `safetensors` weights are not loaded.
            disable_mmap ('bool', *optional*, defaults to 'False'):
                Whether to disable mmap when loading a Safetensors model. This option can perform better when the model
                is on a network mount or hard drive, which may not handle the seeky-ness of mmap very well.

        Returns:
            `str` or `None`: The name assigned to the loaded adapter, or `None` if no LoRA weights were found.
        """
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)
        variant = kwargs.pop("variant", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        disable_mmap = kwargs.pop("disable_mmap", False)

        allow_pickle = False
        if use_safetensors is None:
            use_safetensors = True
            allow_pickle = True

        if low_cpu_mem_usage and not is_accelerate_available():
            low_cpu_mem_usage = False
            logger.warning(
                "Cannot initialize model with low cpu memory usage because `accelerate` was not found in the"
                " environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install"
                " `accelerate` for faster and less memory-intense model loading. You can do so with: \n```\npip"
                " install accelerate\n```\n."
            )

        if low_cpu_mem_usage is True and not is_torch_version(">=", "1.9.0"):
            raise NotImplementedError(
                "Low memory initialization requires torch >= 1.9.0. Please either update your PyTorch version or set"
                " `low_cpu_mem_usage=False`."
            )

        user_agent = {
            "diffusers": diffusers.__version__,
            "file_type": "model",
            "framework": "pytorch",
        }

        # 1. Determine model class from config

        load_config_kwargs = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "proxies": proxies,
            "token": token,
            "local_files_only": local_files_only,
            "revision": revision,
        }

        config = AutoModel.load_config(pretrained_model_name_or_path, subfolder=subfolder, **load_config_kwargs)

        orig_class_name = config["_class_name"]

        if orig_class_name in LOCAL_CLASS_MAPPING:
            model_cls = LOCAL_CLASS_MAPPING[orig_class_name]

        else:
            load_config_kwargs.update({"subfolder": subfolder})

            from diffusers.pipelines.pipeline_loading_utils import ALL_IMPORTABLE_CLASSES, get_class_obj_and_candidates

            model_cls, _ = get_class_obj_and_candidates(
                library_name="diffusers",
                class_name=orig_class_name,
                importable_classes=ALL_IMPORTABLE_CLASSES,
                pipelines=None,
                is_pipeline_module=False,
            )

        if model_cls is None:
            raise ValueError(f"Can't find a model linked to {orig_class_name}.")

        # 2. Get model file

        model_file = None

        if use_safetensors:
            try:
                model_file = _get_model_file(
                    pretrained_model_name_or_path,
                    weights_name=_add_variant(SAFETENSORS_WEIGHTS_NAME, variant),
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent=user_agent,
                )

            except IOError as e:
                logger.error(f"An error occurred while trying to fetch {pretrained_model_name_or_path}: {e}")
                if not allow_pickle:
                    raise
                logger.warning(
                    "Defaulting to unsafe serialization. Pass `allow_pickle=False` to raise an error instead."
                )

        if model_file is None:
            model_file = _get_model_file(
                pretrained_model_name_or_path,
                weights_name=_add_variant(WEIGHTS_NAME, variant),
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
            )

        assert model_file is not None, \
            f"Could not find adapter weights for {pretrained_model_name_or_path}."

        # 3. Initialize model

        base_module = getattr(self, target_module_name)

        torch_dtype = base_module.dtype
        device = base_module.device
        dtype_orig = model_cls._set_default_torch_dtype(torch_dtype)

        # load the state dict early to determine keep_in_fp32_modules
        #######################################
        overwrite_state_dict = dict()
        lora_state_dict = dict()

        adapter_state_dict = load_state_dict(model_file, disable_mmap=disable_mmap)
        for k in adapter_state_dict.keys():
            adapter_state_dict[k] = adapter_state_dict[k].to(dtype=torch_dtype, device=device)
            if "lora" in k:
                lora_state_dict[k.removeprefix(f"{target_module_name}.")] = adapter_state_dict[k]
            else:
                overwrite_state_dict[k.removeprefix(f"{target_module_name}.")] = adapter_state_dict[k]

        # determine initial quantization config.
        #######################################
        pre_quantized = ("quantization_config" in base_module.config
                         and base_module.config["quantization_config"] is not None)
        if pre_quantized:
            config["quantization_config"] = base_module.config.quantization_config
            hf_quantizer = DiffusersAutoQuantizer.from_config(
                config["quantization_config"], pre_quantized=True
            )

            hf_quantizer.validate_environment(torch_dtype=torch_dtype)
            torch_dtype = hf_quantizer.update_torch_dtype(torch_dtype)

            user_agent["quant"] = hf_quantizer.quantization_config.quant_method.value

            # Force-set to `True` for more mem efficiency
            if low_cpu_mem_usage is None:
                low_cpu_mem_usage = True
                logger.info("Set `low_cpu_mem_usage` to True as `hf_quantizer` is not None.")
            elif not low_cpu_mem_usage:
                raise ValueError("`low_cpu_mem_usage` cannot be False or None when using quantization.")

        else:
            hf_quantizer = None

        # Check if `_keep_in_fp32_modules` is not None
        use_keep_in_fp32_modules = model_cls._keep_in_fp32_modules is not None and (
            hf_quantizer is None or getattr(hf_quantizer, "use_keep_in_fp32_modules", False)
        )

        if use_keep_in_fp32_modules:
            keep_in_fp32_modules = model_cls._keep_in_fp32_modules
            if not isinstance(keep_in_fp32_modules, list):
                keep_in_fp32_modules = [keep_in_fp32_modules]

            if low_cpu_mem_usage is None:
                low_cpu_mem_usage = True
                logger.info("Set `low_cpu_mem_usage` to True as `_keep_in_fp32_modules` is not None.")
            elif not low_cpu_mem_usage:
                raise ValueError("`low_cpu_mem_usage` cannot be False when `keep_in_fp32_modules` is True.")
        else:
            keep_in_fp32_modules = []

        # append modules in overwrite_state_dict to keep_in_fp32_modules
        for k in overwrite_state_dict.keys():
            module_name = k.rsplit('.', 1)[0]
            if module_name and module_name not in keep_in_fp32_modules:
                keep_in_fp32_modules.append(module_name)

        init_contexts = [no_init_weights()]

        if low_cpu_mem_usage:
            init_contexts.append(accelerate.init_empty_weights())

        with ContextManagers(init_contexts):
            piflow_module = model_cls.from_config(config).eval()

        torch.set_default_dtype(dtype_orig)

        if hf_quantizer is not None:
            hf_quantizer.preprocess_model(
                model=piflow_module, device_map=None, keep_in_fp32_modules=keep_in_fp32_modules
            )

        # 4. Load model weights

        base_state_dict = base_module.state_dict()
        base_state_dict.update(overwrite_state_dict)
        empty_state_dict = piflow_module.state_dict()
        for param_name, param in base_state_dict.items():
            if param_name not in empty_state_dict:
                continue
            if hf_quantizer is not None and (
                    hf_quantizer.check_if_quantized_param(
                        piflow_module, param, param_name, base_state_dict, param_device=device)):
                hf_quantizer.create_quantized_param(
                    piflow_module, param, param_name, device, base_state_dict, unexpected_keys=[], dtype=torch_dtype
                )
            else:
                assign_param(piflow_module, param_name, param)

        empty_device_cache()

        if hf_quantizer is not None:
            hf_quantizer.postprocess_model(piflow_module)
            piflow_module.hf_quantizer = hf_quantizer

        if len(lora_state_dict) == 0:
            adapter_name = None
        else:
            if adapter_name is None:
                adapter_name = f"{target_module_name}_piflow"
            piflow_module.load_lora_adapter(
                lora_state_dict, prefix=None, adapter_name=adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)
        if adapter_name is None:
            logger.warning(
                f"No LoRA weights were found in {pretrained_model_name_or_path}."
            )

        setattr(self, target_module_name, piflow_module)

        return adapter_name
