# DreamZero 文档导航

## 通用 DreamZero

- [网络结构](./DREAMZERO_ARCHITECTURE.md)
- [Wan causal action DiT 代码解析](./WAN_VIDEO_DIT_ACTION_CAUSAL_CHUNK_CODE.md)
- [Wan2.2-TI2V-5B backbone](./WAN22_BACKBONE.md)
- [新增 embodiment：数据到训练](./DATASET_TO_GEAR_AND_TRAIN.md)
- [DROID 转换](./DROID_CONVERSION.md)

## MobileManiBench / VGGT-3D WAM

从 [MM 当前状态与文档入口](./MM/README.md) 开始阅读。该页面区分：

- 当前已经实现并测试的代码；
- 已实现的 multiblock dual-plan teacher forcing 与离线 evaluator；
- 尚未实现的研究阶段；
- 数据、训练和验证命令；
- 只用于追溯的历史实验/工作树记录。

当前 multiblock 合同与执行入口见
[多 Block Teacher Forcing 实现规范](./MM/MOBILEMANIBENCH_MULTIBLOCK_TEACHER_FORCING_EXECUTION_PLAN.md)；
VGGT v3.x 优化状态见
[VGGT Tokenizer 优化计划](./MM/MOBILEMANIBENCH_VGGT_TOKENIZER_OPTIMIZATION_PLAN.md)。

## 文档状态约定

- **当前实现指南**：必须与代码、YAML 和 shell 默认值同步；
- **proposal / implementation plan**：描述目标，必须显式标出未实现项；
- **实验报告 / development log**：保留当时状态，不用于覆盖当前实现；
- **代码与 resolved config**：发生冲突时优先级最高。
