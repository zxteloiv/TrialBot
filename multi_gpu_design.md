# 多GPU训练支持设计方案

## 当前架构分析

1. **TrialBot**：主控制器，管理实验组件
2. **TrainingUpdater**：训练循环管理器
3. **TestingUpdater**：测试循环管理器
4. **模型初始化**：在`_init_models`方法中，通过`model.cuda(args.device)`移动到单GPU
5. **数据移动**：通过`move_to_device`函数将batch数据移动到指定设备

## 多GPU训练方案

### 1. 支持模式

#### 1.1 DataParallel (最简单)
- 包装模型：`torch.nn.DataParallel(model)`
- 优点：简单易用，无需修改训练循环
- 缺点：效率较低，不支持模型并行

#### 1.2 DistributedDataParallel (DDP) (推荐)
- 使用`torch.distributed`包
- 需要初始化进程组
- 每个进程处理不同数据
- 优点：效率高，支持多节点

#### 1.3 DeepSpeed (高级功能)
- 微软的深度学习优化库
- 支持ZeRO优化、混合精度、梯度检查点等
- 需要额外的配置文件

### 2. 实现计划

#### 2.1 命令行参数扩展
```python
# 多GPU相关参数
parser.add_argument('--gpus', type=str, default='', 
                    help='GPU IDs to use (comma-separated), e.g., "0,1,2,3"')
parser.add_argument('--dist-backend', type=str, default='nccl',
                    help='distributed backend (nccl, gloo, mpi)')
parser.add_argument('--dist-url', type=str, default='tcp://localhost:23456',
                    help='URL used to set up distributed training')
parser.add_argument('--world-size', type=int, default=1,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', type=int, default=0,
                    help='node rank for distributed training')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training')
parser.add_argument('--deepspeed', action='store_true',
                    help='Enable DeepSpeed training')
parser.add_argument('--deepspeed-config', type=str, default=None,
                    help='Path to DeepSpeed configuration file')
```

#### 2.2 模型初始化修改 (`_init_models`方法)
- 根据GPU数量决定是否使用DataParallel/DDP
- 添加DeepSpeed引擎初始化
- 处理模型分布到多个GPU

#### 2.3 数据移动修改
- 修改`move_to_device`函数以支持多GPU
- 在DDP模式下，数据会自动分配到对应GPU

#### 2.4 训练循环修改
- 在DDP模式下，需要同步梯度
- 在DeepSpeed模式下，使用DeepSpeed的优化器

#### 2.5 工具函数添加
- 添加`setup_distributed`函数初始化进程组
- 添加`cleanup_distributed`函数清理资源
- 添加多GPU相关的工具函数

### 3. 具体实现步骤

#### 步骤1：修改命令行参数解析
在`TrialBot.get_default_parser()`中添加多GPU相关参数

#### 步骤2：创建多GPU工具模块
创建`trialbot/utils/multi_gpu.py`包含：
- `setup_distributed_training`
- `cleanup_distributed_training`
- `get_device_count`
- `prepare_model_for_multigpu`

#### 步骤3：修改模型初始化
在`_init_models`方法中添加多GPU支持逻辑

#### 步骤4：修改数据移动
更新`move_to_device`函数以正确处理多GPU场景

#### 步骤5：修改TrainingUpdater
- 支持DDP的数据并行
- 支持DeepSpeed的优化步骤

#### 步骤6：添加示例和文档
- 创建多GPU使用示例
- 更新README文档

### 4. 兼容性考虑

1. **向后兼容**：保持单GPU训练不变
2. **渐进式启用**：用户可以选择启用多GPU功能
3. **错误处理**：提供清晰的错误信息
4. **性能优化**：确保多GPU训练效率

### 5. 测试计划

1. 单GPU训练（确保向后兼容）
2. DataParallel多GPU训练
3. DDP多GPU训练
4. DeepSpeed训练（如果环境支持）

## 实施优先级

1. DataParallel支持（最简单）
2. DDP支持（最实用）
3. DeepSpeed支持（高级功能）

这个设计方案将逐步实现，确保每个步骤都经过充分测试。
