# Prompt
system_prompt_0312.txt 包含一个完整的示例，而 system_prompt_0312_zero.txt 没有示例。
system_prompt_0312.txt 是一个 Few-Shot (实际上是 One-Shot) 提示，因为它提供了一个样本来指导模型。
system_prompt_0312_zero.txt 是一个 Zero-Shot 提示，它只给出指令，不提供样本。

这两个文件都定义了与 LLM 交互的格式规范，特别是针对需要模型在推理过程中生成并利用代码执行结果的任务。它们在 hjy_grpo_program.py 脚本中被用作 系统提示，在向模型发起请求时放在最前面，用以设定模型的行为模式。
指导模型生成: 它们告诉模型应该如何构建其响应，包括思考过程、代码嵌入、结果声明和最终答案的格式。
支持代码执行流程: <program> 标签和结果声明的格式是 hjy_grpo_program.py 中 get_completions 和 run 函数能够正确提取代码、执行代码并将结果反馈给模型的关键。


# RL-GRPO

## 主要组件

### 配置文件 (config.py)
- 定义所有训练参数和配置
- 不包含执行逻辑，而是定义了两组重要的配置：
  - `train_config`: 训练相关的超参数和设置
  - 路径配置: `model_path`, `save_path`, `data_path`
  - Weights & Biases 日志配置: `wandb_*`
  - 生成设备: `gen_device` (指定用于文本生成的 GPU 设备 ID)
  - 算法参数: 
    - `beta`: GRPO/PPO 中 KL 散度惩罚项的系数
    - `clip_param`: PPO/GRPO 损失函数中的裁剪参数
  - 批次设置:
    - `Q_batch_size`, `num_pre_Q`: 每次采样的问题数量和每个问题生成的候选答案数量
    - `train_batch_size`: 实际用于训练更新的批次大小
  - 更新控制: `gen_update_steps` (每隔多少训练步数更新生成模型)
  - 算法开关: `compute_gen_logps` (是否计算生成模型的 logprobs)
  - 服务器配置: `ref_server`, `port`
  - DeepSpeed 配置: `ds_config` (包含 ZeRO 优化、梯度累积、优化器等设置)

### Reference Server (ref_server.py)
- 作为独立的 Reference Server 运行
- 负责计算参考模型的对数概率并作为数据中转站
- 核心功能:
  - 加载参考模型: 固定的 LLM 用于计算参考 logprobs
  - 接收数据: 通过 `/upload` POST 端点接收生成数据
  - 计算参考 Logprobs: 使用固定模型计算每个 token 的对数概率
  - 缓存与分发: 将数据与 ref_logps 打包放入队列
  - 提供数据: 通过 `/get` 端点向训练进程提供完整数据包
- 数据流: gen_worker → /upload → raw_queue → 参考模型计算 → result_queue → /get → 主训练进程

### 主训练与生成脚本 (hjy_grpo_program.py)
包含主要的训练逻辑（模型更新）和独立的生成逻辑（数据收集）

#### 生成工作进程 (gen_worker)
- 启动: 在 rank 0 进程中使用 `torch.multiprocessing` 启动
- 模型加载: 使用 vLLM 加载模型以高效生成文本
- 核心循环:
  - 从数据集采样问题
  - 周期性更新模型权重
  - 生成答案:
    - 使用 system_prompt 格式化输入
    - 生成多个候选答案
    - 递归代码执行: 生成代码 → 执行 → 将结果反馈给模型 → 继续生成
  - 计算奖励:
    - `reward_correct`: 验证答案正确性
    - `reward_format`: 检查输出格式
    - `call_python`: 根据代码执行成功次数给予奖励
  - 计算生成 Logprobs (如果启用)
  - 发送数据到 ref_server

#### 主训练进程 (Main Process)
- 分布式设置: 使用 DeepSpeed 初始化分布式环境
- 模型加载: 加载基础模型并用 DeepSpeed 包装
- 训练循环:
  - 获取批次数据: 从 ref_server 获取处理好的数据
  - 计算 GRPO 损失:
    - 对输入进行前向传播获取当前策略的 logits 和 logprobs
    - 计算 KL 散度惩罚: 当前策略与参考策略之间
    - 计算策略比率: 当前策略与生成策略之间
    - 应用 PPO 裁剪目标函数
    - 最终损失: 结合裁剪目标和 KL 散度惩罚
  - 反向传播与优化
  - 周期性更新生成模型
  - 日志记录与保存

## 整体训练流程

1. **启动**:
   - 启动 ref_server.py
   - 启动 hjy_grpo_program.py (初始化分布式环境并启动 gen_worker)

2. **收集数据** (gen_worker):
   - 使用 vLLM 生成带有代码执行的答案
   - 计算奖励和生成 logprobs
   - 将数据发送到 ref_server

3. **参考 Logprobs 计算** (ref_server):
   - 接收数据并计算参考 logprobs
   - 将完整数据包放入队列

4. **模型训练** (主进程):
   - 获取完整数据包
   - 计算 GRPO 损失
   - 使用 DeepSpeed 更新模型参数

5. **模型同步**:
   - 主进程定期将更新后的模型权重发送给 gen_worker

6. **循环**:
   - 步骤 2-5 持续进行指定的训练步数

这个系统通过分离生成和训练、使用参考模型稳定训练、利用 vLLM 加速生成以及 DeepSpeed 支持分布式训练，实现了一个高效且复杂的 RL 训练系统，专门用于优化需要代码执行能力的 LLM。

