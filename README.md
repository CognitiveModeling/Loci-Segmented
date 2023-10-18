# Loci-Segmented: Improving Scene Segmentation Learning

<b>TL;DR:</b> Introducing Loci-Segmented, an extension to Loci, with a dynamic background module. Demonstrates over 32% relative IoU improvement to SOTA on the MOVi dataset.


https://github.com/CognitiveModeling/Loci-Segmented/assets/28415607/0efa332c-a2c5-40f2-801d-66f79640cbe0

---
## Requirements
A suitable [conda](https://conda.io/) environment named `loci-s` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate loci-s
```

## Dataset and trained models

Preprocessed datasets together with model checkpoints can be found [here](https://unitc-my.sharepoint.com/:f:/g/personal/iiimt01_cloud_uni-tuebingen_de/El2HRkcvN0BAh2J4nddwFmABCgtALSfObFYhzTHJPMBJFw?e=8nPkld)

## Reproducing the results from the paper
Make sure you download all necessary datasets and model checkpoints.
To reproduce the MOVi results run:
```
run-movi-evalulation.sh
python eval-movi.py
```

To reproduce the evaluation on the datasets presented in the review paper on "Compositional scene representation learning via reconstruction: A survey" run:
```
run-review.sh
process-review.sh
python eval-review.py
```

## Training
We extensively utilize pretraining to accelerate the learning process. The training can be broadly divided into five distinct stages:

### Decoder Pretraining:
Pretrain individual decoders using the commands below:
```
python -m model.main -cfg configs/pretrain-mask-decoder.json --pretrain-objects --single-gpu
python -m model.main -cfg configs/pretrain-depth-decoder.json --pretrain-objects --single-gpu
python -m model.main -cfg configs/pretrain-rgb-decoder.json --pretrain-objects --single-gpu
```

### Encoder-Decoder Pretraining:
For pretraining the loci encoder with the pretrained mask, depth, and RGB decoders (excluding hyper-networks) in a single encoder-decoder pass, use either:
```
python -m model.main -cfg configs/pretrain-encoder-decoder-stage1-depth.json --pretrain-objects --single-gpu --load-mask <mask-decoder>.ckpt --load-depth <depth-decoder>.ckpt --load-rgb <rgb-decoder>.ckpt
python -m model.main -cfg configs/pretrain-encoder-decoder-stage1.json --pretrain-objects --single-gpu --load-mask <mask-decoder>.ckpt --load-depth <depth-decoder>.ckpt --load-rgb <rgb-decoder>.ckpt
```
where *-depth names the version that uses depth as input.

### Hyper-Network Pretraining:
Train the hyper-networks inside the encoder with three passes through the encoder-decoder using either:
```
python -m model.main -cfg configs/pretrain-encoder-decoder-stage2-depth.json --pretrain-objects --single-gpu --load-stage1 <encoder-decoder>.ckpt
python -m model.main -cfg configs/pretrain-encoder-decoder-stage2.json --pretrain-objects --single-gpu --load-stage1 <encoder-decoder>.ckpt
```

### Background Pretraining
Train the backgound module using either:
```
python -m model.main -cfg configs/pretrain-background-depth.json --pretrain-bg --single-gpu
python -m model.main -cfg configs/pretrain-background.json --pretrain-bg --single-gpu
```

# Loci Training

TODO

## Visualizations
Visualizations to inspect the various pretraining states can be generated using:
```
python -m model.main -cfg <config> --save-<mask|depth|rgb|objects|bg> --single-gpu --add-text --load <checkpoint>.ckpt
```
Visualizing the final trained loci-s model can be done using
```
python -m model.main -cfg <config> --save --single-gpu --add-text --load <checkpoint>.ckpt
```
