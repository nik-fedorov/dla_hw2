# Speaker separation pipeline

## Report

All details about this homework can be found in 
[wandb report](https://wandb.ai/nik-fedorov/dla_hw2/reports/DLA-HW2-Speaker-separation--Vmlldzo1OTU3NDYw). 

## Description

This is a repository containing a convenient pipeline for training speaker separation models. 

Advantages of this repo:
- possibility of changing experimental configuration by only tuning one json file
- good and clean code structure (see `ss` folder with all elements of pipeline)
- prepared scripts for training and evaluation of models
- prepared downloadable checkpoint

## Installation guide

To set up the environment for this repository run the following command in your terminal (with your virtual environment activated):

```shell
pip install -r ./requirements.txt
```

## Evaluate model

To download my best checkpoint run the following:
```shell
python default_test_model/download_best_ckpt.py
```
if you are interested how I got this checkpoint, you can read about that in 
[wandb report](https://wandb.ai/nik-fedorov/dla_hw2/reports/DLA-HW2-Speaker-separation--Vmlldzo1OTU3NDYw).

You can evaluate model using `test.py` script. Here is an example of command to run my best checkpoint with default test config:

```shell
python test.py \
  -c default_test_model/config.json \
  -r default_test_model/checkpoint.pth \
  -t test_data \
  -o output_dir
```

After that command audio files with separated speech and file `metrics.json` with metrics will be in `output_dir`.

## Training
Use `train.py` for training. Example of command to launch training from scratch:
```shell
python train.py -c hw_asr/configs/config_librispeech.json
```

To fine-tune your checkpoint you can use option `-r` to pass path to the checkpoint file:
```shell
python train.py \
  -c hw_asr/configs/config_librispeech.json \
  -r saved/models/<exp name>/<run name>/checkpoint.pth
```
