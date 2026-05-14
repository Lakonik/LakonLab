"""ComfyUI custom-node package for Lakonik AsymFLUX.2-klein-9B.

Drop this folder into `ComfyUI/custom_nodes/`. The package:

  1. registers a `models/asymflux2/` directory (so the safetensors file can
     be placed alongside other model weights and shows up in the dropdown);
  2. exposes 5 nodes under the "AsymFlux2" category (see nodes.py for the
     full graph diagram).
"""

import os

import folder_paths

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Register a new model-type folder so the adapter safetensors is discoverable
# from anywhere in ComfyUI's models tree.
_models_dir = os.path.join(folder_paths.models_dir, 'asymflux2')
os.makedirs(_models_dir, exist_ok=True)
folder_paths.folder_names_and_paths['asymflux2'] = (
    [_models_dir],
    folder_paths.supported_pt_extensions,
)


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
