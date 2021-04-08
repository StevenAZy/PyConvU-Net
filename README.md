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

# Citations

```bibtex
@article{li2021pyconvu,
  title={PyConvU-Net: a lightweight and multiscale network for biomedical image segmentation},
  author={Li, Changyong and Fan, Yongxian and Cai, Xiaodong},
  journal={BMC bioinformatics},
  volume={22},
  number={1},
  pages={1--11},
  year={2021},
  publisher={Springer}
}
```
