We are planning to release all training and evaluation code for Ped2, Avenue and STC datasets with a detailed instruction to reproduce our results in paper in few days.

# Jigsaw-VAD
Official pytorch implementation for the paper entitled "Video Anomaly Detection by Solving Decoupled Spatio-Temporal Jigsaw Puzzles" (ECCV 2022)

![plot](./figs/arch.png)


# Training
Train a model:

```
python main.py --dataset shanghaitech --val_step 100 --print_interval 20 --batch_size 192 --sample_num 9 --epochs 100 --static_threshold 0.2
```


# Testing
```
python main.py --dataset shanghaitech --sample_num 9 --checkpoint xxx.pth
```
