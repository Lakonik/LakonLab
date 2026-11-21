import sys
import re
from safetensors.torch import load_file, save_file

# Key remaps for common Comfy/XLabs FLUX LoRA naming -> Diffusers FLUX naming.
# This covers typical UNet blocks: single/double blocks, img/txt attn proj, and MLP proj.
RULES = [
    # e.g. lora_unet_double_blocks_0_img_attn_proj.lora_up.weight
    (r"^lora_unet_double_blocks_(\d+)_img_attn_proj\.", r"lora_unet_double_blocks_\1_img_attn_proj."),
    (r"^lora_unet_double_blocks_(\d+)_txt_attn_proj\.", r"lora_unet_double_blocks_\1_txt_attn_proj."),
    (r"^lora_unet_double_blocks_(\d+)_mlp\.", r"lora_unet_double_blocks_\1_mlp."),
    (r"^lora_unet_single_blocks_(\d+)_img_attn_proj\.", r"lora_unet_single_blocks_\1_img_attn_proj."),
    (r"^lora_unet_single_blocks_(\d+)_txt_attn_proj\.", r"lora_unet_single_blocks_\1_txt_attn_proj."),
    (r"^lora_unet_single_blocks_(\d+)_mlp\.", r"lora_unet_single_blocks_\1_mlp."),
    # Some trainers save alpha as scalar; keep the same suffix
    (r"\.alpha$", r".alpha"),
    # Make sure the LoRA weight names keep the .lora_down/.lora_up suffixes
    (r"\.lora_down\.weight$", r".lora_down.weight"),
    (r"\.lora_up\.weight$", r".lora_up.weight"),
]


def remap(k):
    new = k
    for pat, rep in RULES:
        new = re.sub(pat, rep, new)
    return new


def main(src, dst):
    sd = load_file(src)
    out = {}
    for k, v in sd.items():
        out[remap(k)] = v
    save_file(out, dst)
    print(f"wrote {dst} ({len(out)} tensors)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_comfy_flux_lora_to_diffusers.py in.safetensors out.safetensors")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
