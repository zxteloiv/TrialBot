# TrialBot 多GPU训练使用指南

本文档介绍如何在 TrialBot 框架中使用多GPU进行训练，支持以下三种模式：
1. **DataParallel** - 最简单的多GPU训练方式
2. **DistributedDataParallel (DDP)** - PyTorch官方推荐的分布式训练
3. **DeepSpeed** - 微软的深度学习优化库

## 1. 安装依赖

确保已安装必要的依赖：

```bash
# PyTorch (已包含在TrialBot依赖中)
pip install torch

# DeepSpeed (可选，如需使用DeepSpeed功能)
pip install deepspeed
```

## 2. 命令行参数

TrialBot 新增了以下多GPU相关参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--gpus` | str | `''` | 使用的GPU ID，逗号分隔，如 "0,1,2,3"。设为 "all" 使用所有可用GPU |
| `--dist-backend` | str | `'nccl'` | 分布式后端：nccl, gloo, mpi |
| `--dist-url` | str | `'tcp://localhost:23456'` | 分布式训练初始化URL |
| `--world-size` | int | `1` | 分布式训练的节点数 |
| `--rank` | int | `0` | 节点排名 |
| `--multiprocessing-distributed` | flag | `False` | 启用多进程分布式训练 (DDP) |
| `--deepspeed` | flag | `False` | 启用DeepSpeed训练 |
| `--deepspeed-config` | str | `None` | DeepSpeed配置文件路径 |
| `--local_rank` | int | `-1` | 本地排名（由torch.distributed.launch自动设置） |

## 3. 使用示例

### 3.1 使用 DataParallel (最简单)

```bash
# 使用GPU 0和1进行训练
python your_training_script.py --gpus 0,1

# 使用所有可用GPU
python your_training_script.py --gpus all
```

### 3.2 使用 DistributedDataParallel (DDP)

```bash
# 单节点多GPU DDP训练
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=23456 \
    your_training_script.py \
    --gpus 0,1 \
    --multiprocessing-distributed \
    --world-size 1 \
    --rank 0
```

### 3.3 使用 DeepSpeed

```bash
# 使用DeepSpeed进行训练
python your_training_script.py --gpus 0,1 --deepspeed

# 使用自定义DeepSpeed配置
python your_training_script.py --gpus 0,1 --deepspeed --deepspeed-config ds_config.json
```

## 4. 代码示例

### 4.1 基本使用

```python
from trialbot.training.trial_bot import TrialBot

def get_model(hparams, vocab):
    # 创建你的模型
    return YourModel()

# 创建TrialBot实例
bot = TrialBot(
    args=args,  # 包含多GPU参数
    trial_name="multi_gpu_experiment",
    get_model_func=get_model
)

# 运行训练
bot.run()
```

### 4.2 完整示例

参考 `examples/multi_gpu_example.py` 查看完整示例。

## 5. 配置说明

### 5.1 批量大小调整

在多GPU训练中，批量大小会自动调整：
- **DataParallel**: 总批量大小 = 每GPU批量大小 × GPU数量
- **DDP**: 总批量大小 = 每GPU批量大小 × 世界大小
- **DeepSpeed**: 根据配置文件中的 `train_batch_size` 和 `gradient_accumulation_steps` 确定

### 5.2 DeepSpeed 配置

默认DeepSpeed配置：
```json
{
  "train_batch_size": 16,
  "gradient_accumulation_steps": 1,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.001,
      "betas": [0.9, 0.999],
      "eps": 1e-8
    }
  },
  "fp16": {
    "enabled": false
  },
  "zero_optimization": {
    "stage": 1,
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "contiguous_gradients": true
  }
}
```

## 6. 注意事项

1. **模型保存**: 在DDP模式下，只需在rank 0进程保存模型
2. **数据加载**: 确保数据加载器支持分布式采样
3. **随机种子**: 在多进程中设置相同的随机种子以保证可重复性
4. **内存使用**: 多GPU训练会增加显存使用，注意调整批量大小
5. **性能优化**: 使用 `nccl` 后端通常能获得最佳性能

## 7. 故障排除

### 7.1 CUDA内存不足
- 减少批量大小
- 使用梯度累积
- 启用DeepSpeed的ZeRO优化

### 7.2 分布式训练连接失败
- 检查防火墙设置
- 确保所有节点可以互相访问
- 使用正确的 `--dist-url`

### 7.3 性能不佳
- 使用 `nccl` 后端
- 调整数据加载器的工作进程数
- 使用混合精度训练

## 8. 高级功能

### 8.1 混合精度训练
```bash
# 使用DeepSpeed的FP16训练
python your_training_script.py --gpus 0,1 --deepspeed --deepspeed-config ds_config_fp16.json
```

### 8.2 梯度累积
在DeepSpeed配置中设置 `gradient_accumulation_steps` 以实现梯度累积。

### 8.3 模型并行
对于超大模型，可以结合使用模型并行和数据并行。

## 9. 参考资源

- [PyTorch Distributed Training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [TrialBot GitHub](https://github.com/zxteloiv/TrialBot)

---

通过以上指南，您可以在 TrialBot 框架中轻松使用多GPU加速训练过程。
