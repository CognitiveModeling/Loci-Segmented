# Loci-Segmented: Improving Scene Segmentation Learning

<b>TL;DR:</b> Introducing Loci-Segmented, an extension to Loci, with a dynamic background module. Demonstrates over 32% relative IoU improvement to SOTA on the MOVi dataset.

https://github.com/CognitiveModeling/Loci-Segmented/assets/28415607/62661702-11b3-41eb-a713-95acb840e76d


---
## Requirements
A suitable [conda](https://conda.io/) environment named `loci-s` can be created
and activated with:

```
conda env create -f environment.yml
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

## Use your own data

We provide a example dataset creating script in [data/datasets/create-dataset-example.py](data/datasets/create-dataset-example.py) that you can adjust to your needs.

---


## Training Guide

Our training pipeline employs multi-GPU configurations and extensive pretraining to accelerate model convergence. Specifically, we use a single node with 8 x GTX1080 GPUs for the pretraining phase, and a single node with 8 x A100 GPUs for the final Loci-s training. Below are the details for each stage of the training pipeline.

> **Note:** The following examples use a single GPU setup, which is suboptimal for performance. Multi-GPU configurations are highly recommended.

### Pretraining Phases

1. **Decoder Pretraining**

    Pretrain individual decoders for mask, depth, and RGB using the following commands:

    ```bash
    python -m model.main -cfg configs/pretrain-mask-decoder.json --pretrain-objects --single-gpu
    python -m model.main -cfg configs/pretrain-depth-decoder.json --pretrain-objects --single-gpu
    python -m model.main -cfg configs/pretrain-rgb-decoder.json --pretrain-objects --single-gpu
    ```

2. **Encoder-Decoder Pretraining**

    Pretrain the Loci encoder with already pretrained mask, depth, and RGB decoders:

    ```bash
    python -m model.main -cfg configs/pretrain-encoder-decoder-stage1.json --pretrain-objects --single-gpu --load-mask <mask-decoder>.ckpt --load-depth <depth-decoder>.ckpt --load-rgb <rgb-decoder>.ckpt
    ```

    > For a version that utilizes depth as an input feature, append `-depth` to the config name.

3. **Hyper-Network Pretraining**

    Execute three passes through the encoder-decoder architecture to train the internal hyper-networks:

    ```bash
    python -m model.main -cfg configs/pretrain-encoder-decoder-stage2.json --pretrain-objects --single-gpu --load-stage1 <encoder-decoder>.ckpt
    ```

4. **Background Module Pretraining**

    Train the background module:

    ```bash
    python -m model.main -cfg configs/pretrain-background.json --pretrain-bg --single-gpu
    ```

### Final Training: Loci-s

Execute full-scale training for Loci-s:

```bash
python -m model.main -cfg configs/loci-s.json --train --single-gpu --load-objects <encoder-decoder>.ckpt --load-bg <background>.ckpt
```
---

## Visualization Guide

Generate visualizations to inspect the model at various stages of pretraining and during the final phase.

### Pretraining Visualizations

To visualize individual components like mask, depth, RGB, objects, or background during pretraining:

```bash
python -m model.main -cfg <config> --save-<mask|depth|rgb|objects|bg> --single-gpu --add-text --load <checkpoint>.ckpt
```

### Final Model Visualizations

For visualizing the fully trained Loci-s model:

```bash
python -m model.main -cfg <config> --save --single-gpu --add-text --load <checkpoint>.ckpt
```

> **Note:** To visualize using the segmentation pretraining network, append the `--load-proposal` flag followed by the corresponding checkpoint:

```bash
--load-proposal <proposal>.ckpt
```
