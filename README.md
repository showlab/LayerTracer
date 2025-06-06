# LayerTracer
**LayerTracer: Cognitive-Aligned Layered SVG Synthesis via  Diffusion Transformer**
<br>
[Yiren Song](https://scholar.google.com.hk/citations?user=L2YS0jgAAAAJ), 
[Danze Chen](https://scholar.google.com/citations?hl=en&user=7XRxZr0AAAAJ), 
and 
[Mike Zheng Shou](https://sites.google.com/view/showlab)
<br>
[Show Lab](https://sites.google.com/view/showlab), National University of Singapore
<br>

<a href="https://arxiv.org/abs/2502.01105"><img src="https://img.shields.io/badge/ariXv-2502.01105-A42C25.svg" alt="arXiv"></a>

<br>

<img src='./img/teaser.png' width='100%' />

## Installation
### 1. **Environment setup**
```bash
git clone https://github.com/showlab/LayerTracer.git
cd LayerTracer

conda create -n layertracer python=3.11.10
conda activate layertracer
```
### 2. **Requirements installation**
```bash
pip install --upgrade -r requirements.txt
```

## LoRa models
You can download the trained checkpoints of LoRA for `text2sequence inference`. Below are the details of available models:

| **Model**  |  **Description**  |  **Resolution** |
|:-:|:-:|:-:|
| [flux_lora_icon_blackline](https://drive.google.com/file/d/1i2_mlGy-LwcZ0ief7b2SZyaaHu5nWf38/view?usp=drive_link) | This model is used to generate **nine grid** icon images with **black lines**. | 768,768 |
| [flux_lora_emoji](https://drive.google.com/file/d/1Be4UJHzaIoM_KTXTkZnaEFLLbdGfhe62/view?usp=drive_link) | This model is used to generate **four grid** emoji and icon images. | 1024,1024 |

## Inference
### 1. **Text 2 Sequence**
According to the actual situation, replace all model paths, file paths, and parameters in `scripts/text2sequence.sh`.

```bash
chmod +x scripts/text2sequence.sh
scripts/text2sequence.sh
```

`NOTICE`

When generating 9-grid images using `prompts`, please modify your prompt according to the `examples` provided below to avoid generating irrelevant content. Add this prefix to your prompt:
```bash
# nine-grid icon
sks, This is a nine square grid image that describes the process of creating SVG icon images, ...
# or
sks, a set of nine icons, each representing ...
```
```bash
# four-grid emoji
sks, This is a four square grid image that describes the process of creating SVG emoji images, ...
```

### 2. **Image 2 Sequence**
#### 2.1 Merge LoRA to flux.1
Use our `scripts/lora_merge.sh` template script to merge the LoRA （[flux_lora_icon_blackline](https://drive.google.com/file/d/1i2_mlGy-LwcZ0ief7b2SZyaaHu5nWf38/view?usp=drive_link) and [flux_lora_emoji](https://drive.google.com/file/d/1Be4UJHzaIoM_KTXTkZnaEFLLbdGfhe62/view?usp=drive_link)) to flux.1 checkpoints for further recraft training. Note that the merged model may take up **around 50GB** of your memory space.

```bash
chmod +x scripts/lora_merge.sh
scripts/lora_merge.sh
```
#### 2.2 Recraft model
According to the actual situation, replace all model paths, file paths, and parameters in `scripts/image2sequence.sh`.

The [icon_lora_weights](https://drive.google.com/file/d/1LZPvEnsCDvrVbGPvtcdZJ-bC9Q64oJE_/view?usp=sharing) is a Recraft model used for generating 9-grid black line icon images. Simply replace the corresponding paths in `scripts/image2sequence.sh` to use it.


```bash
chmod +x scripts/image2sequence.sh
scripts/image2sequence.sh
```

## QuickStart
### 9grid Sequence(blackline) to SVG
```bash
python layertracer_icon_9grid.py --input input/icon.png --output output/
```
- `--input` : Input image path.
- `--output` : Output directory.
- `--colormode`: choices=['color', 'binary'], Color mode for SVG conversion.
- `--mode` : choices=['spline', 'polygon', 'none'], Tracing mode for SVG conversion.
- `--filter_speckle` : Speckle filter threshold.
- `--color_precision` : Color precision for SVG conversion.
- `--corner_threshold`: Corner detection threshold.
- `--length_threshold` : Length threshold for path simplification, in [3.5, 10].
- `--splice_threshold` : Splice threshold for path merging.
- `--path_precision` : Path precision for SVG conversion.

### 4grid Sequence to SVG
You can customize the parameters, just like the above process:
```bash
python layertracer_emoji_4grid.py --input input/emoji.png --output output/
```

## Dataset
- [4grid_emoji_icon](https://drive.google.com/file/d/1xaFUCP90XlHog8dBZw9YY1XEMSxTuWLX/view?usp=sharing)
- [9grid_icon_blackline](https://drive.google.com/file/d/1cAq7IukCGcLdsT0AxWPXTutU5WQjLF75/view?usp=sharing)

## Citation
```
@inproceedings{Song2025LayerTracerCL,
  title={LayerTracer: Cognitive-Aligned Layered SVG Synthesis via Diffusion Transformer},
  author={Yiren Song and Danze Chen and Mike Zheng Shou},
  year={2025},
  url={https://api.semanticscholar.org/CorpusID:276094351}
}
```
