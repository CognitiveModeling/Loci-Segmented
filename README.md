# Loci-Segmented: Improving Scene Segmentation Learning

<b>TL;DR:</b> Introducing Loci-Segmented, an extension to Loci, with a dynamic background module. Demonstrates over 28% relative IoU improvement to SOTA on the MOVi dataset.


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

TODO

## Reproducing the results from the paper
Make sure you download all necessary datasets and model checkpoints and placed them in the right folders. 
The evaluation scripts for the MOVi datasets can then be started with:
```
run-movi-evalulation.sh
```

To run the evaluation on the datasets presented in "Compositional scene representation learning via reconstruction: A survey" run:
```
run-review.sh
```
