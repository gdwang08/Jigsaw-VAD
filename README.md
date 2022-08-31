# Jigsaw-VAD
Official pytorch implementation for the paper entitled "Video Anomaly Detection by Solving Decoupled Spatio-Temporal Jigsaw Puzzles" (ECCV 2022)

![plot](./figs/arch.png)


# Data Preparation
Please make sure that you have sufficient storage.
```
python gen_patches.py --dataset shanghaitech --phase test --filter_ratio 0.8 --sample_num 9
```

|    Dataset    | # Patch (train) |  # Patch (test) |  filter ratio  |  sample num  |  storage  |
|:-------------:|:---------------:|:---------------:|:--------------:|:------------:|:---------:|
|      Ped2     |       27660     |       31925     |       0.5      |       7      |     20G   |
|     Avenue    |       96000     |       79988     |       0.8      |       7      |     58G   |
|  Shanghaitech |      145766     |      130361     |       0.8      |       9      |    119G   |


# Training
Train a model:

```
python main.py --dataset shanghaitech --val_step 100 --print_interval 20 --batch_size 192 --sample_num 9 --epochs 100 --static_threshold 0.2
```


# Testing
```
python main.py --dataset shanghaitech --sample_num 9 --checkpoint xxx.pth
```
