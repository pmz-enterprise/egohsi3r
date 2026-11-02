<h1 align="center">EgoHSI3R: Online 4D Hand-Scene Interaction Reconstruction from Egocentric Video</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2610.40832"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b" alt="Paper"></a>
  <a href="https://www.modelscope.cn/models/QwenMaxPreview/egohsi3r"><img src="https://img.shields.io/badge/Weights-ModelScope-624aff" alt="Model weights"></a>
</p>

## Setup

```bash
conda create -n egohsi3r python=3.11 -y
conda activate egohsi3r
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

<details>
<summary>Optional: build RoPE for faster inference</summary>

Requires a matching CUDA toolkit and C++ compiler.

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
(cd src/croco/models/curope && python setup.py build_ext --inplace)
```

</details>

## Weights

| Resource | Location | Source |
| --- | --- | --- |
| EgoHSI3R | `weights/egohsi3r.pth` | [ModelScope](https://www.modelscope.cn/models/QwenMaxPreview/egohsi3r) |
| Hand detector | `weights/detector.pt` | [WiLoR](https://github.com/rolpotamias/WiLoR) |
| MANO models | `weights/mano/` | [MPI MANO](https://mano.is.tue.mpg.de/) |
| MANO mean parameters | `weights/mano/mano_mean_params.npz` | [WiLoR](https://github.com/rolpotamias/WiLoR/tree/main/mano_data) |

Download the EgoHSI3R checkpoint from ModelScope and save it at the path above.
For MANO, register on the official website, accept the license, and extract
`MANO_RIGHT.pkl` and `MANO_LEFT.pkl` into `weights/mano/`.

Download the WiLoR detector and mean parameters:

```bash
mkdir -p weights/mano
wget -O weights/detector.pt https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt
wget -O weights/mano/mano_mean_params.npz https://raw.githubusercontent.com/rolpotamias/WiLoR/main/mano_data/mano_mean_params.npz
```

## Inference

Run the included demo video. Results are saved to `outputs/demos/demo`;
the viewer runs at `http://127.0.0.1:8081`.

```bash
python infer.py --seq_path demos/demo.mp4 --save --output_dir outputs/demos/demo --port 8081
```

Replay saved results:

```bash
python infer.py --results_dir outputs/demos/demo --port 8081
```

Use `--no_viewer` for headless execution. See `python infer.py --help` for options.
Hand meshes use bright cyan (right) and bright magenta (left). Contact defaults
to `binary`: vertices at or above the threshold (default `0.5`) turn neon green.
`probability` blends the side color toward neon green as probability increases;
`off` shows solid side colors and hides contact markers.

In **Hands / Contact**, neon green **Contact markers** emphasize current-frame contact
vertices. **Contact on RGB** projects the
same vertices onto the input image, including the occluded side of the hand.
The readout shows counts and maximum probabilities for each hand. **Next contact**
jumps to the next frame with a vertex above the selected threshold. These are
model predictions on hand vertices, not measured object contact or contact force.
Use `--bbox_padding` to adjust the margin around detected hands (default: `0.25`
per side, giving 1.5× width and height before aspect-ratio and boundary adjustment).

## Acknowledgements

Built on [CUT3R](https://github.com/CUT3R/CUT3R),
[Human3R](https://github.com/fanegg/Human3R),
[WiLoR](https://github.com/rolpotamias/WiLoR), and
[TTT3R](https://github.com/Inception3D/TTT3R).
Third-party code and assets retain their original licenses.

## Citation

If you find this work useful, please cite:

```bibtex
@misc{egohsi3r,
  title={EgoHSI3R: Online 4D Hand-Scene Interaction Reconstruction from Egocentric Video},
  author={Mingzhi Pei and Zhangchen Ye and Ruoqu Chen and Liu Cao and Mengdi Xu},
  year={2026},
  eprint={2610.40832},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2610.40832}
}
```
