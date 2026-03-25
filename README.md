# gym_super_mario

基于 `gym-super-mario-bros` 和 PyTorch 的 DDQN 训练示例。这个仓库目前聚焦一件事：把原本集中在单文件里的训练脚本拆成清晰的模块，便于继续迭代环境封装、模型结构和训练策略。

## 重构后的目录

```text
gym_super_mario/
├── env_test.py               # 环境冒烟测试
├── mario_rl/
│   ├── __init__.py
│   ├── actions.py            # 动作集映射
│   ├── agent.py              # DDQN Agent 与经验回放
│   ├── config.py             # 配置数据类
│   ├── env.py                # 环境构建与观测包装器
│   ├── model.py              # 在线网络 / 目标网络
│   ├── trainer.py            # 训练循环与 checkpoint 策略
│   └── utils.py              # 设备、seed、Gym API 兼容工具
├── requirements.txt
├── train.py                  # 训练入口
└── weights/                  # 训练权重
```

## 为什么要这样拆

原始版本的核心问题不是功能少，而是职责混在一起：

- 环境包装、网络定义、Agent、训练循环全部塞在 `train.py`
- 类方法依赖外部全局变量，例如 `device`、`episode`、`checkpoint_period`
- checkpoint、日志、超参数都没有明确边界，后续改动成本高
- README 只有简介，没有把运行方式和代码结构解释清楚

重构后，每个模块只负责一层职责，便于你后续单独替换：

- 改环境预处理：看 `mario_rl/env.py`
- 调 DDQN 超参数：看 `mario_rl/config.py`
- 改网络结构：看 `mario_rl/model.py`
- 改训练策略或保存策略：看 `mario_rl/trainer.py`

## 环境依赖

建议使用 Python 3.9 到 3.11。

安装依赖：

```bash
pip install -r requirements.txt
```

如果你的系统没有配置好 NES / 渲染相关依赖，`gym-super-mario-bros` 可能无法正常启动；这属于环境层问题，不是本仓库代码问题。

## 训练

最简单的训练方式：

```bash
python train.py
```

常用参数：

```bash
python train.py --episodes 500 --action-set simple --save-dir weights
python train.py --checkpoint weights/checkpoint_662.pth
python train.py --render
python train.py --device cpu
```

参数说明：

- `--episodes`：训练回合数
- `--action-set`：动作集，可选 `right_only`、`simple`、`complex`
- `--checkpoint`：从已有权重继续训练
- `--save-dir`：checkpoint 输出目录
- `--render`：训练时显示游戏画面，速度会明显变慢
- `--device`：手动指定 `cpu` 或 `cuda`

## 环境测试

如果你只是想确认环境包装器是否正常：

```bash
python env_test.py --steps 200
python env_test.py --steps 200 --render
```

## 当前实现说明

当前版本仍然保留原项目的主要训练思路：

- 算法：Double DQN
- 经验回放：`deque`
- 目标网络：按固定步数同步
- 图像预处理：跳帧、灰度化、缩放、帧堆叠
- checkpoint：保存模型参数和当前探索率

同时修复了原始代码里的几个结构性问题：

- 去掉了类内部对全局变量的隐式依赖
- 把 checkpoint 保存逻辑收敛到训练器与 Agent 内部
- 统一处理 Gym 不同版本的 `reset/step` 返回值
- 让训练入口只负责参数解析和对象组装

## 后续可继续做的方向

- 增加训练指标持久化，例如 CSV / TensorBoard
- 引入评估模式，区分训练和验证
- 把 reward shaping、早停策略、课程学习做成可选配置
- 补单元测试和最小集成测试

## 说明

仓库中现有的 `weights/checkpoint_662.pth` 会继续兼容当前模型结构；使用 `--checkpoint` 即可加载。
