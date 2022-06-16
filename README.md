# LaneTransorfmer
## 王志博 复旦大学

# 特别说明

由于modifiedLaneTransformer中的MultiheadAttention函数的mask使用bool型，因此需要将

```bash
.../python3.8/site-packages/torch/nn/functional.py
```
该文件中第5091行
```bash
new_attn_mask.masked_fill_(attn_mask, float("-inf"))
```
中的-inf修改为0方可正常运行

其他部分和之前的train.py基本相同

# Requirements

- matplotlib==3.2.2
- numpy==1.18.1
- pandas==1.0.0
- torch==1.10.2
- tqdm==4.42.0

# 文件结构
- log：存储训练过程中产生的记录文件
- Dataset：默认用来存放预处理后的.pkl文件
- pic：每次val的过程都会存储一个batch的可视化结果
- saved：存储模型
- model：模型文件

# 模型训练

首先，修改train.py文件中的train_path, val_path两个超参数为Argoverse数据集所在路径，例如：

```bash
train_path = '/home/wzb/Datasets/Argoverse/train/data/'
val_path = '/home/wzb/Datasets/Argoverse/val/data/'       
```

接下来，根据需求修改parser文件中的pkl_save_dir参数，确定预处理结束后的pkl文件的存储地址（默认为当前目录下的Dataset文件夹）。

随后, 将parser文件中的mode参数设置为train, 执行下列命令即可开始模型训练：

```bash
python train.py           
```

如果是第一次进行训练，会进行数据预处理，此后只要不再改变pkl_save_dir，预处理结果就可以复用。

# 模型验证
在train.py中给出欲验证的模型文件的路径，例如：

```bash
load_checkpoint('./saved/baseline/baseline.pth', model, optimizer)
```

将parser中的mode参数改为val，执行下列命令即可开始模型验证：

```bash
python train.py 
```


