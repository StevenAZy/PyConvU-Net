# PyConvU-Net
A lightweight and multiscale network for biomedical image segmentation
# Introduction
dataset.py is used to load dataset and preprocess the data 
flops_counter.py is to calculate the number of parameters and computational complexity
metrics.py is to calculate the metrics: MIoU and Dice
plot.py is to plot the loss, MIoU and Dice curve
# Environment
Pytorch
# Run
python main.py --arch xxx --dataset xxx --epoch xxx --batch_size xxx
# Reference
https://github.com/Andy-zhujunwen/UNET-ZOO
https://github.com/iduta/pyconv
