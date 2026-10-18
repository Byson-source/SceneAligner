# Third-party components

The code in this repository is released under the MIT License (`LICENSE`). It installs or downloads the following components, which keep their own licenses.

| Component | Use | License |
|---|---|---|
| [π³](https://github.com/yyfz/Pi3) code | 3D reconstruction | [BSD 3-Clause](https://github.com/yyfz/Pi3/blob/main/LICENSE) |
| π³ weights fine-tuned by [MegaDepth-X](https://megadepth-x.github.io/) | 3D reconstruction | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), the license of the π³ weights (non-commercial use only) |
| Original [π³ weights](https://huggingface.co/yyfz233/Pi3) | 3D reconstruction on Structured3D | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (non-commercial use only) |
| [VGGT](https://github.com/facebookresearch/vggt) code, and weights fine-tuned by MegaDepth-X | 3D reconstruction for the training data of C3 (optional) | [VGGT License](https://github.com/facebookresearch/vggt/blob/main/LICENSE.txt) (code), [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), the license of the [VGGT-1B](https://huggingface.co/facebook/VGGT-1B) weights (non-commercial use only) |
| [GeoCalib](https://github.com/cvg/GeoCalib) code and weights | Gravity estimation | [Apache-2.0](https://github.com/cvg/GeoCalib/blob/main/LICENSE) (code), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (weights) |
| [DINOv3](https://github.com/facebookresearch/dinov3) code and weights | Image encoder | [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md) |
| [C3](https://c3po-correspondence.github.io/) | `data/c3` and `examples/S221` | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| [Structured3D](https://structured3d-dataset.org/) | Dataset of the single-view experiments (not included) | [Structured3D Terms of Use](https://structured3d-dataset.org/) |
| [Structured3D toolbox](https://github.com/bertjiazheng/Structured3D) and [SemRayLoc](https://github.com/TAU-VAILab/SemRayLoc) | The floorplan drawing in `scenealigner/structured3d.py` is adapted from them | [MIT](https://github.com/bertjiazheng/Structured3D/blob/master/LICENSE) (toolbox); SemRayLoc states no license |

Our LoRA weights ([jhcho99/SceneAligner](https://huggingface.co/jhcho99/SceneAligner)) are derived from the DINOv3 weights and are released under the DINOv3 License. Those for Structured3D were trained on Structured3D and are subject to its Terms of Use as well.
