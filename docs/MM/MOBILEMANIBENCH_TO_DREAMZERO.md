# MobileManiBench 到 DreamZero / VGGT-3D-WAM 的数据转换方案

> 状态：转换脚本已实现；G1/XHand 小批量转换、语义回算和 DreamZero 原生 loader smoke test 已通过  
> 数据源：`/mnt/yihao/datasets/MobileManiBench/MobileManipVLA_opensource`（2026-07-22 远端抽样）  
> 参考：[DATASET_TO_GEAR_AND_TRAIN.md](./DATASET_TO_GEAR_AND_TRAIN.md)、[vggt_3d_wam_proposal.md](./vggt_3d_wam_proposal.md)、[MobileManiBench 项目页](https://dexhand.github.io/MobileManiBench_Website/)、[官方代码](https://github.com/DexHand/MobileManiBench)、[论文](https://arxiv.org/abs/2602.05233)

## 1. 目标与结论

本方案要同时满足两个不同层次的需求：

1. 生成能被现有 DreamZero / GEAR 数据链路读取的 LeRobot v2 兼容数据，用于 RGB、语言、机器人状态和动作的基线训练。
2. 无损保留 VGGT-3D-WAM 需要、但 DreamZero 当前标准模态没有覆盖的标定、深度、分割、底盘运动、完整机器人状态、场景和几何监督，并通过扩展加载器送入 3D tokenizer 与 WAM。

核心设计决定如下：

- G1 与 XHand 必须生成两个独立数据集/embodiment。它们的动作维度、关节数、body 数和末端执行器不同，不能不带 mask 地混入同一固定维度向量。
- 标准 `meta/modality.json` 只登记现有 DreamZero 能理解的 `state/action/video/annotation`；新增字段登记在 `meta/extensions.json`、`meta/calibration.json`、`meta/robot_schema.json` 和几何 sidecar 中。这样基线可直接运行，3D 分支也不会丢数据。
- 训练 3D 的主参考系采用“每个 episode 初始底盘坐标系” `B0`。同时保留原始局部世界系 `W` 数据以及 `T_W_B0`，保证可逆和便于排错。
- 研究版 action 采用两路结构化移动操作计划：`base plan + manipulator plan`。base plan 输出未来底盘 waypoints；manipulator plan 在同一路向量中输出未来 EEF pose 与夹爪/灵巧手构型。源 7/18 维 action 和 IK joint target 作为原始控制信息完整保留。
- 源数据没有单独记录上层规划器下发的底盘 waypoint，但这不妨碍使用未来实际底盘轨迹作为 imitation-learning waypoint 标签。必须在元数据中注明其 provenance 为 `derived_from_future_robot_base`，不能声称它是数据集原生记录的 commanded waypoint。
- 源 `action` 已经是末端相对位姿增量加手爪/灵巧手目标，不能再使用 DreamZero 通用的 `action - state` 相对动作转换。该数据集设置 `relative_action: false`，waypoint/EEF target 在专用预处理器内直接构造。
- 发布版深度和分割是 H.264 可视化视频，不是无损真值。第一版正式采用 MP4 作为 coarse pseudo-range/segmentation supervision，配合质量标签和置信 mask 跑通 2D/3D tokenizer 与 WAM；无损 NPZ 重放作为后续高精度几何增强，而不是第一版链路的前置门槛。
- 远端抽样视频容器标记为 25 FPS，而官方采集/控制语义为 30 Hz。帧与 state 应按索引对齐；物理控制时间和媒体时间必须分开保存，不能直接把 MP4 时长当作 episode 的物理时长。

## 2. 远端数据审计

### 2.1 目录结构

远端根目录包含两个 embodiment：

```text
MobileManipVLA_opensource/
├── G1_Robot/
└── XHand_Robot/
```

其下按照任务、资产来源、物体类别、实例和训练批次组织：

```text
<robot>/<Close|Open>/<partnet|unidoor|ycb>/
  <category>/<instance...>/train_0/
  ├── trajectory_info.txt
  ├── success_test.txt
  └── trajectories/
      └── traj_000/
          ├── scene_infos.json
          ├── log.txt
          └── episode_000/
              ├── rgb_image_head.mp4
              ├── rgb_image_arm.mp4
              ├── depth_image_head.mp4
              ├── depth_image_arm.mp4
              ├── segment_image_head.mp4
              ├── segment_image_arm.mp4
              └── state_infos.pkl
```

抽查到的物体组包括：

- PartNet：box、cart、dishwasher、faucet、laptop、microwave、oven、refrigerator、table、toilet、trashcan、washingmachine。
- UniDoor：cabinet、car、fridge、lever_door、round_door、safe、window。
- YCB：在 `Open/ycb/ycb` 下，任务语义实际为 pick。

源层级中的 `Open/Close` 不能直接等同于自然语言任务。官方 prompt 逻辑对 cart/chair 映射为 pull/push，对 YCB 映射为 pick；转换时应同时保存原始任务标签和规范化 skill。

### 2.2 episode 文件和同步关系

G1 抽样 episode 有 171 帧，XHand 抽样 episode 有 139 帧；每条样本的 6 个视频帧数和 `state_infos.pkl` 第一维一致。视频抽样参数为：

- 编码：H.264；
- 分辨率：520 × 520；
- 容器帧率：25 FPS；
- 图像视角：头部相机 `head` 和腕部相机 `arm`；
- 每个视角均有 RGB、深度可视化和分割可视化。

同步应以 `frame_index` 为权威，而不是用现有 MP4 的 PTS 推导物理采样时间。

### 2.3 `state_infos.pkl` 实际字段

共同字段如下；`T` 为 episode 帧数：

| 源字段 | G1 shape | XHand shape | 已核对语义 |
|---|---:|---:|---|
| `time` | `(T, 1)` | `(T, 1)` | 从 1 开始的源 step 计数，不是秒 |
| `success` | `(T, 1)` | `(T, 1)` | 成功标记/成功状态 |
| `action` | `(T, 7)` | `(T, 18)` | 6D 末端位姿增量 + D 维末端关节目标 |
| `camera_head_pose` | `(T, 6)` | `(T, 6)` | 相机世界位置 xyz + roll/pitch/yaw |
| `camera_arm_pose` | `(T, 6)` | `(T, 6)` | 同上，腕部相机动态外参 |
| `object` | `(T, 9)` | `(T, 9)` | 抓取位姿 xyz/rpy + 目标位置 xyz |
| `robot_base` | `(T, 12)` | `(T, 12)` | 位置、rpy、线速度、角速度 |
| `robot_hand` | `(T, 12)` | `(T, 12)` | 末端位置、rpy、线速度、角速度 |
| `robot_body` | `(T, 48, 12)` | `(T, 47, 12)` | 每个刚体的位置、rpy、线/角速度 |
| `robot_joint` | `(T, 36, 3)` | `(T, 40, 3)` | 每个关节的 q、qdot、qddot |
| `robot_joint_target` | `(T, 36)` | `(T, 40)` | IK/控制器完整关节目标 |
| `init` | dict | dict | robot、object、room 初始状态 |

`init` 中至少包括：

- robot：`root_link_state_w (1,13)`、完整 `joint_pos`、`joint_vel`；
- object：`root_link_state_w (1,13)`、物体关节位置和速度；
- room：`room_pos (3)`、`room_rot (3)`。

`scene_infos.json` 还保存 `task`、`object` 及 `room_infos`。后者包含房间 scale、height、translation、orientation、类型、split、资产 group/name/place 和源 USD 路径。源绝对资产路径只作为 provenance 保存，不能假定换机器后仍可访问。

## 3. 源字段的关键语义

### 3.1 原始控制动作与研究版规划动作

官方控制流程中，`action` 的前 6 维是相对上一帧的右手腕位姿增量，后 `D` 维是末端执行器目标：

- G1：`6 + 1 = 7`，1 维平行夹爪目标；
- XHand：`6 + 12 = 18`，12 维灵巧手关节目标。

动作先被裁剪到 `[-1, 1]`；位置和旋转前三/后三维分别乘以 `0.025 m` 和 `0.025 rad`，再与当前手腕位姿合成。IK 根据末端目标生成移动底盘和 7-DoF 右臂的 joint target。末端执行器部分由归一化值映射到关节上下限。

因此，源 7/18 维 action 适合作为兼容 DreamZero 的控制基线，但不是研究方案最终要预测的完整移动操作计划。转换后至少保留三种源控制表达：

1. `action.raw_world_normalized`：源数组，7/18 维，绝不覆盖；
2. `action.eef_delta_base`：换到当前底盘系、缩放到物理单位的 6D baseline policy target；
3. `action.end_effector_target`：末端关节物理目标；同时保留对应 normalized 值。

`robot_joint_target` 是控制器/IK 的结果，建议拆出：

- `supervision.base_joint_target`；
- `supervision.arm_joint_target`；
- `supervision.end_effector_joint_target`；
- `supervision.full_joint_target`。

这些字段可用于 baseline、auxiliary head、动力学一致性或控制蒸馏，但不应与研究版 waypoint/EEF plan target 混为一谈。

研究版 action chunk 以锚点帧 `t` 为条件，按配置的未来 offset 集合 `K = {k1, ..., kH}` 构造：

```text
action.plan.base_waypoints[h] = pose2d(inv(T_W_B(t)) @ T_W_B(t + kh))
eef_pose[h]                   = pose(inv(T_W_B(t)) @ T_W_EEF(t + kh))
hand_config[h]                = robot_joint[t + kh, end_effector_indices, position]
action.plan.manipulator[h]    = concat(eef_pose[h], hand_config[h])
action.plan.valid[h]          = (t + kh < episode_length)
```

- `base_waypoints` 推荐编码为 `[x, y, sin(yaw), cos(yaw)]`，单位为 m；它是相对当前底盘锚点 `B(t)` 的未来路径点序列，而不是相邻帧速度/增量。
- `manipulator` 的 EEF slice 推荐编码为 `[x, y, z, rotation_6d]`，同样表达在固定锚点 `B(t)` 中；hand slice 为 G1 1 维夹爪或 XHand 12 维手指构型。因此 manipulator plan 的维度分别为 G1 10、XHand 21。
- 另存 `supervision.eef_pose_in_future_base[h] = inv(T_W_B(t+kh)) @ T_W_EEF(t+kh)`，供执行器解耦或一致性损失使用，但不与主 EEF plan 表达混用。
- hand slice 与 EEF pose 使用一致的“未来实际轨迹”语义：从 `robot_joint[t+kh, end_effector_indices, 0]` 读取未来实际夹爪/手指关节位置。经过时间对齐并按 joint limit 反归一化的源 hand action 另存为 `supervision.commanded_hand_target`，`robot_joint_target` 的末端 slice 另存为 `supervision.controller_hand_joint_target`；两者只用于 baseline、控制蒸馏和执行误差分析，不替代主 plan 标签。
- EEF pose 和 hand configuration 共用一个 manipulator prediction branch，但仍使用独立 slice、normalization、mask 和 loss 权重。
- offset 可以逐控制帧，也可以是稀疏 waypoint；必须在 `extensions.json` 中固定 `waypoint_offsets`、单位和对应的 control time，训练/推理使用同一组 offset。

最终是两路预测，而不是三路：

```text
Action plan
├── base_plan[h]        = [x, y, sin(yaw), cos(yaw)]                   # 4
└── manipulator_plan[h] = [eef_x, eef_y, eef_z, eef_rotation_6d,
                           hand_configuration...]                     # 9 + D
```

推荐 manipulator branch 的输出 tensor 为 `[H, 9 + D]`，但损失仍按语义拆分：

```text
L_action = λ_base     L_base_waypoint
         + λ_eef_pos  L_eef_position
         + λ_eef_rot  L_eef_rotation
         + λ_hand     L_hand_configuration
```

这里“合成一路”指共享 manipulator token/decoder/projection，不代表把 m、rotation-6D 和 joint angle 用同一个统计量或同一个未加权 MSE 处理。EEF position、rotation 和 hand configuration 分别归一化，rotation-6D 在解码时重新正交化；G1/XHand hand slice 使用 embodiment mask。主 plan 的三个组成部分——base waypoint、EEF pose、hand configuration——均来自同一未来时刻的实际状态，从而避免 realized trajectory 与 commanded target 混用。

执行第 `h` 个计划点时，两个分支还能组合得到相对未来底盘的手腕目标：

```text
T_B(t+kh)_EEF(t+kh)
  = inverse(T_B(t)_B(t+kh)) @ T_B(t)_EEF(t+kh)
```

导航控制器跟踪 `base_plan[h]`，机械臂 IK 跟踪上式得到的 EEF target，夹爪/灵巧手直接执行同一个 `manipulator_plan[h]` 中的 hand slice。这样模型接口只有底盘与机械臂两路，同时保持整机规划的几何一致性。

因此，“底盘数据是 derived/realized”描述的是**监督标签的来源**；模型的输出接口仍然可以且应该定义为未来 waypoint plan。

### 3.2 动作时间偏移

官方 VLA dataset 代码在读取时使用：

```python
episode_action = concatenate([action[1:], action[-1:]], axis=0)
```

即 observation `t` 的监督 target 来自记录数组 `t+1`，末帧重复最后一个 action。转换器应复现并显式记录该规则：

- `action.raw_recorded[t] = source_action[t]`；
- `action.target[t] = source_action[min(t + 1, T - 1)]`；
- `action.source_index[t] = min(t + 1, T - 1)`；
- `meta/extensions.json` 中写入 `action_alignment: "next_recorded_step"`。

该 shift 仅用于原始 command baseline 和 commanded auxiliary supervision。研究版 `action.plan.base_waypoints` 与 `action.plan.manipulator` 直接按 `t + kh` 读取未来实际状态，不对状态轨迹额外执行 `t+1` shift。在全量转换前必须用“动作积分后的期望手腕位姿”和下一帧 `robot_hand` 做误差统计，验证 command shift，而不能只依据代码静态推断。

### 3.3 Action Plan 标签的可生成性与前置验证

当前 `state_infos.pkl` 已包含生成主 Action Plan 所需的全部状态字段，且这些标签来自数值状态数组，不受 depth/segmentation H.264 有损压缩影响：

| 计划标签 | 数据来源 | 可生成性 | 转换前置条件 |
|---|---|---|---|
| Base waypoint | `robot_base[:, 0:6]` | 可直接生成 | 验证移动平面轴和 yaw convention |
| EEF pose | `robot_hand[:, 0:6]` | 可直接生成 | 确认源 hand link，并定义到目标 TCP/EEF frame 的固定变换 |
| Hand configuration | `robot_joint[:, end_effector_indices, 0]` | 数据具备 | 必须按官方 joint-name 顺序冻结准确 indices |

因此数据在结构上足以生成 `base_plan [H,4]` 和 `manipulator_plan [H,9+D]`。但只有以下四项全部通过，标签才能视为 training-ready：

1. **Hand joint indices**：G1 的 `robot_joint` 为 36 个 joint，XHand 为 40 个 joint。禁止假定数组最后 1/12 维就是夹爪/手指；必须从官方 robot schema 按名称解析主动末端关节并写入 `meta/robot_schema.json`。转换时同时检查名称、index、joint limit、源维度和 schema hash。
2. **EEF/TCP 定义**：`robot_hand` 保存的是仿真环境指定 hand link 的位姿，不必然等于研究或部署控制器使用的 TCP、指尖中心或抓取中心。必须显式登记 `eef_source_link`、`eef_target_frame` 和固定 `T_sourceLink_TCP`。若目标是 TCP，则使用 `T_W_TCP = T_W_robot_hand @ T_robotHand_TCP` 生成 EEF plan；不得仅把字段重命名为 TCP。
3. **坐标 convention**：验证 `robot_base` 与 `robot_hand` 是否共享同一局部世界系、移动平面对应的世界轴、roll/pitch/yaw 顺序和 yaw 正方向。应可视化 base trajectory、EEF trajectory 及 `T_B(t+kh)_EEF(t+kh)`，并检查相对 EEF 始终位于合理工作空间。
4. **时间与 horizon**：主 plan 严格按 state frame index 读取 `t+kh`，物理 offset 使用 30 Hz control clock；25 FPS 只服务于现有视频媒体解码。对 `t+kh >= T` 的位置设置 horizon invalid，不重复末帧制造静止伪标签。

主 plan 标签的性质是 future realized trajectory，而不是原生 commanded waypoint。该语义适用于行为克隆、flow matching 和 WAM 轨迹规划；原始 action、IK target 和实际轨迹之间的 tracking error 应单独统计，不能混入主标签。

### 3.4 坐标系和旋转

源状态/动作声明在每个仿真环境的局部世界坐标系 `W` 中；camera pose 为相机在 `W` 中的位置和 roll-X / pitch-Y / yaw-Z 弧度。为避免欧拉角跳变，训练字段统一使用 quaternion 或 rotation-6D，原始 rpy 仍原样保存。

建议定义：

- `W`：源 episode 的局部世界系；
- `B(t)`：第 `t` 帧底盘系；
- `B0 = B(0)`：episode 初始底盘系，3D 主参考系；
- `C_v(t)`：视角 `v ∈ {head, wrist}` 的相机机体系；
- `O_v(t)`：OpenCV optical frame，`+x` 右、`+y` 下、`+z` 前。

若源 pose 生成的是 camera-to-world：

```text
T_B0_Cv(t) = inverse(T_W_B0) @ T_W_Cv(t)
T_Cv_B0(t) = inverse(T_B0_Cv(t))
T_B0_B(t)  = inverse(T_W_B0) @ T_W_B(t)
```

投影使用 `T_Ov_B0`。Isaac/仿真 camera frame 到 OpenCV optical frame 的固定旋转不能靠符号猜测，必须通过已知点或机器人 mesh 的重投影验证后写入 `meta/calibration.json`。

选择 `B0` 而不是当前 `B(t)` 的原因是：它在一个 episode 内稳定，能直接表达移动底盘的 ego-motion，适合跨帧点云、BEV、occupancy 和 point-flow；同时相较全局房间系数值范围更小。保留 `T_W_B0` 后仍可恢复世界系。

### 3.5 相机内参与外参

相机外参是逐帧动态字段：头部相机随机器人运动，腕部相机还随手臂运动。禁止只保存一份静态外参。

远端 pkl 没有直接保存内参。官方机器人配置给出两种 embodiment 的两个相机均为：

```text
model = pinhole
width = height = 520
focal_length = 18.0
horizontal_aperture = 36.0
```

这给出名义值 `fx = 18 / 36 × 520 = 260 px`。`fy`、主点、像素中心约定、畸变和 optical-frame 变换仍必须从 Isaac Lab 的实际相机标定 API 导出或用投影实验校验，不能把 `(260, 260)` 当作未经验证的最终答案。

每个相机在 `meta/calibration.json` 中至少保存：

- camera model、width、height；
- `K`、`K^-1`、distortion model/coefficients；
- 原始 focal length、aperture、focus distance；
- camera prim、robot link、配置中的 link-to-camera offset；
- `T_cameraBody_optical` 及其 convention；
- 标定来源、版本、验证误差和 valid 标志。

逐帧 parquet/geometry index 保存 `T_B0_CameraBody(t)`、`T_B0_Optical(t)` 和 transform valid mask。

## 4. 目标数据组织

建议每个 embodiment 一个标准 LeRobot/GEAR 根目录：

```text
MobileManiBench_DreamZero/
├── g1/
│   ├── data/chunk-000/episode_000000.parquet
│   ├── videos/chunk-000/observation.images.head/episode_000000.mp4
│   ├── videos/chunk-000/observation.images.wrist/episode_000000.mp4
│   ├── geometry/chunk-000/episode_000000/
│   │   ├── depth_head.npz
│   │   ├── depth_wrist.npz
│   │   ├── segmentation_head.npz
│   │   ├── segmentation_wrist.npz
│   │   └── index.json
│   └── meta/
│       ├── info.json
│       ├── modality.json
│       ├── embodiment.json
│       ├── stats.json
│       ├── relative_stats_dreamzero.json
│       ├── tasks.jsonl
│       ├── episodes.jsonl
│       ├── extensions.json
│       ├── calibration.json
│       ├── robot_schema.json
│       ├── geometry.json
│       ├── source_manifest.jsonl
│       └── splits.json
└── xhand/
    └── ...
```

RGB 可以使用 symlink/hardlink 避免重复拷贝，但生产/迁移数据应允许 `--link-mode copy`。深度和分割不建议继续作为普通 H.264 视频流保存；无损数组使用分块 NPZ、Zarr 或 Arrow sidecar，按 episode 和 frame index 索引。

### 4.1 为什么不把所有字段塞进标准 `modality.json`

DreamZero 当前 `modality.json` 的 schema 和 `LeRobotSingleDataset.get_data_by_modality()` 只理解 `state/action/video/annotation` 及少数已有特殊分支。随意新增 `geometry`、`calibration` 会在加载时被拒绝；把 K、4×4 变换、深度和 point cloud 拼进普通 state 又会破坏归一化与网络语义。

因此采用双层契约：

- 兼容层：现有 DreamZero 可直接消费的核心 state/action/RGB/language；
- 扩展层：专用 MobileManiBench loader 读取的标定、几何和 privileged supervision。

所有扩展字段必须在 `extensions.json` 中声明 dtype、shape、frame、unit、时间对齐、role、quality 和存储位置；禁止只有数组没有语义元数据。

## 5. 完整字段映射

下表中的 role 分为：`policy_input`、`policy_target`、`geometry_input`、`geometry_target`、`aux_target`、`eval_only` 和 `provenance`。

### 5.1 核心观测与动作

| 源数据 | 目标字段 | frame / unit | role | 处理 |
|---|---|---|---|---|
| `rgb_image_head.mp4` | `observation.images.head` | camera / uint8 | policy_input, geometry_input | 按帧索引读取 |
| `rgb_image_arm.mp4` | `observation.images.wrist` | camera / uint8 | policy_input, geometry_input | 将 `arm` 规范命名为 `wrist`，保留 source name |
| `robot_base[:,0:6]` | `observation.state.base_pose_b0` | B0 / m, rot6d | policy_input | W→B0，rpy→rot6d |
| `robot_base[:,6:12]` | `observation.state.base_twist_b0` | B0 / m/s, rad/s | policy_input | 旋转速度向量到 B0 |
| `robot_hand[:,0:6]` | `observation.state.eef_pose_base` | B(t) / m, rot6d | policy_input | 与官方 VLA 语义一致 |
| `robot_hand[:,6:12]` | `observation.state.eef_twist_base` | B(t) / m/s, rad/s | policy_input | 向量旋转 |
| `robot_joint` active arm | `observation.state.arm_joint_{position,velocity}` | joint / rad, rad/s | policy_input | 按 robot schema 索引 |
| `robot_joint` end effector | `observation.state.end_effector_joint_{position,velocity}` | joint | policy_input | G1 1 维，XHand 12 维 |
| future `robot_base` | `action.plan.base_waypoints` | anchor B(t) / m, sin/cos yaw | policy_target | 未来 pose2d 序列，按 waypoint offsets 采样 |
| future `robot_hand` + future end-effector joint position | `action.plan.manipulator` | anchor B(t) / m, rot6d + joint | policy_target | 全部为未来实际状态；G1 10 维、XHand 21 维 |
| shifted `action[:,0:6]` | `action.baseline.eef_delta_base` | B(t) / normalized | baseline_target | 与官方 VLA loader 一致：next-step shift 后 W→B(t)，兼容列不乘 0.025；物理诊断量另行派生 |
| shifted `action[:,6:]` | `action.baseline.end_effector_target` | joint / rad | baseline_target | normalized→joint limit；normalized 原值另存 |
| 原始 `action` | `action.raw_world_normalized` | W / normalized | provenance | 不修改的 7/18 维源数组 |

现有 DreamZero 兼容基线可把 7/18 维源控制 target 打包为固定向量。研究版使用 base/manipulator 两个逻辑分支；其中 manipulator tensor 内部拼接 EEF pose 和 hand configuration，并携带 horizon valid mask。在 `robot_schema.json` 记录每个 slice 的名字、起止位置、单位和归一化方式。G1/XHand 分开统计和训练。

### 5.2 底盘、完整机器人和接触相关字段

| 源/派生 | 目标字段 | role | 说明 |
|---|---|---|---|
| future `robot_base` | `action.plan.base_waypoints` | policy_target | 相对锚点 B(t) 的未来 `[x,y,sin(yaw),cos(yaw)]` 序列 |
| 相邻 `robot_base` | `supervision.base_motion_se2` | aux_target | 局部一步 `log_SE2(inv(T_W_B(t)) @ T_W_B(t+1))`，用于动力学/速度一致性，不是主 waypoint 输出 |
| `robot_base` twist | `supervision.base_twist` | aux_target | 实现运动，不是 action command |
| `robot_joint_target` base slice | `supervision.base_joint_target` | aux_target | IK 结果，需精确 joint-name 映射 |
| `robot_joint_target` arm slice | `supervision.arm_joint_target` | aux_target | 7 维右臂目标 |
| shifted source hand action | `supervision.commanded_hand_target` | aux_target | 原始夹爪/手指控制目标，与主 plan 的未来实际 hand configuration 分开 |
| `robot_joint_target` hand slice | `supervision.controller_hand_joint_target` | aux_target | 控制器目标，用于 tracking error/控制蒸馏 |
| 完整 `robot_joint` | `observation.aux.full_joint_state` | geometry_input/aux | q/qdot/qddot，保留所有固定/非策略关节 |
| 完整 `robot_body` | `observation.aux.body_kinematics` | geometry_input/aux | robot mesh/动态占据和自遮挡可用 |
| 完整 `robot_joint_target` | `supervision.full_joint_target` | aux_target | 控制器重建、调试 |

源数据未发现接触力、碰撞对、触觉或抓取接触标签。不得从 success 或距离阈值伪造“真值接触”。如果后续通过仿真重放获得 contact sensor，可新增：

- `supervision.contact_pairs`；
- `supervision.contact_force`；
- `supervision.grasp_contact_mask`；
- 对应 `available`/`valid` mask 和生成版本。

若只能按距离派生，应命名为 `pseudo_contact` 并记录阈值与置信度。

### 5.3 相机、深度、分割与 3D 字段

| 源/派生 | 目标字段 | role | 说明 |
|---|---|---|---|
| `camera_*_pose` | `observation.camera.<view>.T_b0_camera` | geometry_input | 逐帧 4×4；同时保存 raw xyz/rpy |
| 标定恢复 | `observation.camera.<view>.K` | geometry_input | 静态元数据可在 batch 时展开 |
| 标定恢复 | `T_camera_optical` | geometry_input | 固定 convention transform |
| depth MP4 / future lossless distance | `geometry.depth.<view>` | geometry_target | 第一版为 pseudo camera-ray range；未来 lossless 版本单独登记 |
| 深度有效性 | `geometry.depth_valid.<view>` | geometry_target | finite、范围、边界/压缩置信 mask |
| segment MP4 / future integer labels | `geometry.segmentation.<view>` | geometry_target | 第一版为带 unknown/confidence 的近似标签 |
| RGB-D+K+外参 | `geometry.points_b0.<view>` | geometry_target/cache | 可按需在线生成，通常不必永久重复存储 |
| 多视角深度 | `geometry.occupancy_b0` | geometry_target | voxel 原点、尺寸、分辨率、unknown mask 必须登记 |
| 相邻点云/对象 | `geometry.scene_flow_b0` | geometry_target | dynamic/static/occluded mask 分开 |
| camera ray | `geometry.rays_b0.<view>` | geometry_input/cache | 可由 K 和 pose 精确重算，优先在线生成 |

可重算的 point cloud、ray、voxel 不应成为唯一真值；应保存生成它们所需的源 depth、quality/valid mask、K、pose、坐标 convention 和生成版本，使 MP4 pseudo-depth 与未来 lossless depth 始终可区分。

### 5.4 场景、物体、任务和评估字段

| 源数据 | 目标字段 | role | 说明 |
|---|---|---|---|
| `object[:,0:6]` | `privileged.object_grasp_pose_b0` | geometry_target/eval_only | 训练策略时默认不作为输入，避免 privileged leakage |
| `object[:,6:9]` | `privileged.object_goal_position_b0` | eval_only | 目标监督/评估 |
| `init.robot` | `episode.init.robot` | provenance/geometry | 初始 root、joint pos/vel |
| `init.object` | `episode.init.object` | provenance/geometry | 初始 root、关节状态 |
| `init.room` | `episode.init.room` | provenance | 房间位姿 |
| `scene_infos.json` | `episode.scene` | provenance/eval | task/object/room/asset IDs |
| `success` | `episode.success`, `frame.success` | eval_only | 不能作为部署时观测输入 |
| 路径层级 | `episode.source_taxonomy` | provenance | robot、Open/Close、asset group、category、instance、train/traj/episode |
| 规范化 prompt | `annotation.language.task` | policy_input | open/close/push/pull/pick 等统一规则 |

`scene_infos.json` 的源 USD 绝对路径可能包含不同大小写根目录，只应保存为 `source_asset_uri`；另生成稳定的 `scene_id`、`object_instance_id` 和 `asset_relpath`。

### 5.5 每一个扩展字段都必须携带的元数据

`extensions.json` 的 field registry 至少包含：

```json
{
  "name": "geometry.depth.head",
  "dtype": "float32",
  "shape": [520, 520],
  "unit": "m",
  "frame": "head_optical",
  "value_semantics": "camera_ray_range",
  "time_alignment": "same_frame_index",
  "role": "geometry_target",
  "storage": "geometry/{chunk}/{episode}/depth_head.npz",
  "quality": "lossy_h264_pseudo_range_depth",
  "valid_mask": "geometry.depth_valid.head",
  "source": "depth_image_head.mp4",
  "version": 1
}
```

这里的 `.npz` 是转换器从 MP4 解码后生成的 float32 sidecar/cache，不代表源 Hugging Face 包含 `distance_image_*.npz`。若未来获得 lossless simulator distance，应使用新的 source/quality/version 登记，不能覆盖或冒充同一数据版本。

通用必备项为：坐标系、单位、旋转/四元数顺序、shape、dtype、时间对齐、输入/目标角色、可用性 mask、质量/来源和生成版本。

## 6. 深度和分割质量等级与分阶段路径

[官方 `env_model.py`](https://github.com/DexHand/MobileManiBench/blob/main/unimanip/utils/env_model.py#L1299-L1307) 当前的 episode 保存路径会写出：

- `distance_image_head.npz`、`distance_image_arm.npz`：毫米级 `uint16` 距离图；
- `segment_label_head.npz`、`segment_label_arm.npz`：整数分割标签。

但官方 Hugging Face 发布包及本次抽查的解压目录只看到相应 MP4，未发现 NPZ，也未发现 NPY/Parquet/HDF5/Zarr 等替代无损格式。Hugging Face 的 `unpack.py` 只原样解 tar，没有执行格式转换。因此更准确的结论是：**源码具备 NPZ 保存逻辑，但当前公开发布包未包含这些文件；这是发布裁剪还是采集版本差异，官方没有说明。**

当前 MP4 的性质为：

- depth 被裁剪到 0–5 m、映射到 0–255 后再 H.264 编码；仅量化步长就约为 `5/255 ≈ 1.96 cm`，还叠加有损压缩误差；
- segmentation 是彩色可视化再 H.264 编码，物体/机器人/背景的整数 label 已不可精确恢复，尤其是边界。

[官方 VLA `dataset_model.py`](https://github.com/DexHand/MobileManiBench/blob/main/unimanip/utils/dataset_model.py#L127-L132) 本身也读取 `depth_image_*.mp4`。因此 MP4 并非不可用：它适合作为大规模 coarse spatial supervision，把 2D/3D tokenizer 与 WAM 全链路跑通；限制在于不能据此声称高精度 metric geometry。

因此分两条路径：

### 路径 A：第一版正式路径——MP4 粗几何监督

- RGB、state、action、language 可用于 DreamZero 基线。
- 从 depth MP4 解码近似 range depth，记录 `quality=lossy_h264_pseudo_range_depth`；确认解码已恢复原始 full-range 0--255 后使用 `D = 5.0 * Y / 255`，其中 `Y` 取 luma/灰度或三通道中位数。若视频/解码器使用 limited-range YUV，必须先按 color-range 元数据恢复或用已知灰度标定。
- 生成置信 mask：接近 0/255、深度强边缘、遮挡边界、图像边界和明显 codec artifact 降权或设为 invalid；255 可能表示远于 5 m、无穷远或截断值，不能当成精确 5 m 表面。
- 源语义为相机 `distance_to_camera`，即沿单位 camera ray 的 range；ray decoder 使用 `P(d)=O+d*normalize(r)`。禁止默认当作 optical-axis z-depth。
- depth loss 使用 valid-mask Huber/Charbonnier 或粗 depth-bin loss，不使用全像素等权 L1；bin/voxel 有效尺度不得细于压缩数据精度。第一版可从约 4--5 cm 粗尺度开始，再用实测误差调整。
- 分割按期望颜色做最近邻分类并设置颜色距离阈值；不确定像素为 unknown，不能做 RGB 精确相等判断。
- occupancy 只生成带 uncertainty band 的 soft/free/unknown 弱监督；不生成硬接触、精细碰撞或毫米级表面标签。
- 该路径可以用于第一版完整 tokenizer/WAM 训练和效果对比，但论文中应称为 coarse/pseudo range supervision，不称为 simulator metric ground truth。

### 路径 B：高精度几何增强（非第一版跑通硬门槛）

- 使用官方环境和相同 scene/object/seed/state 重放，导出原始 distance 与 integer segment NPZ；
- 同时从渲染器/相机 API 导出最终 K、projection matrix 和 frame convention；
- 对重放轨迹执行 RGB/机器人状态一致性校验；若不能逐帧复现，标注 regenerated domain，不与原 episode 假装严格像素同步；
- lossless 数据用于校准 MP4 的 MAE/RMSE、边缘误差、截断比例和有效尺度，并用于高精度 occupancy、collision/contact 与 metric reconstruction 实验。

对于当前“2D tokens 为主、3D tokens 提供空间理解”的研究目标，路径 A 足以启动和评估第一版；路径 B 仅是提出高精度 metric geometry、精细 occupancy/collision/contact 结论时的硬门槛。

## 7. 两套时间轴的处理

建议保留以下列：

```text
frame_index                 int64   权威离散同步索引
source_step                 int64   源 time 字段（通常从 1 开始）
control_timestamp           float64 (source_step - source_step[0]) / 30
timestamp                   float64 供当前 LeRobot 视频解码使用的媒体时间
media_timestamp             float64 frame_index / probed_media_fps
control_dt                  float32 1 / 30
media_fps                   float32 实测值，本次为 25
```

推荐实现有两种模式：

1. `--video-clock preserve`：不改源视频，LeRobot 的 `timestamp = frame_index / 25`，保证现有 loader 解码正确；模型 horizon、速度和物理损失使用 `control_timestamp`/30 Hz。
2. `--video-clock repair-30fps`：逐帧解码再以明确 30 FPS PTS 重编码，断言输出帧数和像素顺序不变，然后令 `timestamp = frame_index / 30`。不能使用可能静默丢帧/补帧的简单 ffmpeg `-r` 流程。

默认先采用 mode 1，最少改变原始数据；所有基于秒的采样逻辑必须显式选择 control clock。3D tokenizer 和 WAM 的跨帧对应按 frame index 构造。

## 8. 转换实现步骤

当前已新增且本阶段唯一需要的代码文件：

```text
scripts/data/convert_mobilemanibench_to_gear.py
```

它只生成数据，不修改 DreamZero loader、模型或训练代码。G1 和 XHand 写入同一输出父目录下的两个独立 dataset root。实际 CLI：

```bash
cd /mnt/yihao/codes/dreamzero
/mnt/yihao/envs/dreamzero/bin/python \
  scripts/data/convert_mobilemanibench_to_gear.py convert \
  --input-root /mnt/yihao/datasets/MobileManiBench/MobileManipVLA_opensource \
  --output-root /mnt/yihao/datasets/MobileManiBench/MobileManipVLA_dreamzero_smoke_v2 \
  --embodiments g1 xhand \
  --max-episodes-per-embodiment 2 \
  --waypoint-offsets 1,4,8,12,16,24 \
  --link-mode hardlink \
  --validate
```

已有转换结果会被拒绝覆盖；完整转换时去掉 `--max-episodes-per-embodiment` 或设为 `0`，并使用新的输出目录。`hardlink` 不复制视频内容且不修改源文件；跨文件系统时改用 `symlink` 或 `copy`。

小批量验证同时执行：Parquet schema/shape/NaN/Inf、六路视频精确帧数、media/control 双时钟、由保留的 world-frame base/EEF/joint 状态独立回算 core state、next-step action、base waypoint、manipulator plan 与 valid mask，以及首/中/末帧六路拼图和 action-plan XY 图。最后还应使用 DreamZero 自己的 `LeRobotSingleDataset` 读取 state/action/language，并实际解码 head/wrist 帧。

### 2026-07-22 实际 smoke conversion 结果

最终可用于小样本过拟合的父目录：

```text
/mnt/yihao/datasets/MobileManiBench/MobileManipVLA_dreamzero_smoke_v2/
├── g1/
└── xhand/
```

| dataset root | episodes | frames | core action | manipulator/waypoint | validation |
|---|---:|---:|---:|---:|---|
| `g1` | 2 | 378 | 7 | 10 / 4 | passed |
| `xhand` | 2 | 334 | 18 | 21 / 4 | passed |

六路视频均为 `520×520`、25 FPS，帧数与 Parquet/state 严格一致；control clock 为 30 Hz。两条 G1 和两条 XHand episode 的 core state、shifted action、base plan、manipulator plan 独立回算最大绝对误差均为 `0`。DreamZero 原生 `LeRobotSingleDataset` 实测结果为：G1/XHand 分别加载 378/334 个 step；G1 hand action chunk shape `(4,1)`，XHand 为 `(4,12)`；head 视频成功解码为 `(1,520,520,3) uint8`，语言返回 `close box`。

每个 dataset root 的完整机器可读结果位于 `meta/validation_report.json`，可视样例位于 `validation_samples/`。目前仍有两个预期 warning：相机 K/optical convention 是 `nominal_unverified`；depth MP4 是 lossy pseudo-range。因此该 smoke 数据适合 RGB-action 链路与 coarse 3D 实验，不应直接支持高精度投影损失或厘米级几何结论。

### Step 0：冻结 schema 和 robot name map

- 从官方 robot 配置提取 G1/XHand 完整 body/joint 有序名称、主动关节、上下限和 source index。
- 写入 `robot_schema.json`，包含 base、arm、end-effector slice；end-effector indices 必须按 joint name 解析，禁止按“数组末尾 1/12 维”猜测。
- 明确 `robot_hand` 对应的源 link，并登记研究/部署使用的 TCP frame 与 `T_robotHand_TCP`；若两者不同，转换标签前先应用该固定变换。
- 为每个 schema 计算 hash；转换器发现维度/名称不一致立即失败。

### Step 1：建立只读清单

- 遍历所有 `state_infos.pkl`，记录相对源路径、robot、任务层级、traj、episode。
- 解析 `scene_infos.json`、`trajectory_info.txt` 和必要的 log；不依赖 log 作为唯一标签。
- 对每个被转换 episode 的 pickle 计算 size/hash，写 `source_manifest.jsonl`。
- 生成稳定 episode ID，禁止用并行处理完成顺序编号。
- pickle 只能读取可信数据源；转换工具不得对任意上传 pickle 直接反序列化。

### Step 2：读取并做结构验证

- 所有 state 数组第一维必须相同；6 路视频 `nb_frames == T`。
- 验证 G1 action/joint/body 为 7/36/48，XHand 为 18/40/47。
- 检查 NaN、Inf、越界动作、非单调 source step、缺失相机或空视频。
- 失败 episode 写入 quarantine manifest，不静默截断到最短长度。

### Step 3：动作对齐和单位还原

- 保存 unmodified raw action。
- 生成 next-step shifted target 和 source index。
- 兼容 DreamZero/官方 VLA loader 的 `action` 保持 normalized 表达：next-step shift 后从 W 旋转到当前 B(t)，不乘 `0.025`。
- 如果生成用于物理重建/诊断的派生 delta，再对前三/后三维分别乘 `0.025 m`/`0.025 rad`；该派生量不得冒充训练兼容列。
- 末端 normalized 值依据 robot joint limits 映射为物理 joint target。
- 用下一帧手腕位姿、full joint target 做统计验证。
- 上述 7/18 维 target 作为兼容基线和控制 provenance 保留，不作为研究版 action plan 的唯一标签。

### Step 4：坐标标准化和标定

- raw xyz/rpy → quaternion/SE(3)，检查 `R^T R` 和 `det(R)`。
- 计算并保存 W、B0、B(t)、camera body、optical frame 间变换。
- 若研究 EEF 与源 `robot_hand` link 不同，先计算 future `T_W_TCP = T_W_robot_hand @ T_robotHand_TCP`。
- 对每个锚点按 `waypoint_offsets` 从 future `robot_base`、future EEF/TCP 和 future end-effector joint position 构造语义一致的 realized trajectory：相对 B(t) 的 base waypoints，以及拼接 EEF pose 与实际 hand configuration 的 manipulator plan；同时生成 horizon valid mask。
- 计算一步 realized base SE(2) delta 作为辅助一致性监督；最后一帧标记 invalid，不盲目重复成真实运动。
- 从仿真相机导出/验证 K 和 camera convention。

### Step 5：写核心 parquet 和视频

- parquet 写 timestamp、frame/source/control 索引、episode/task index、核心 packed state/action、逐帧动态外参和小型 auxiliary 数组。
- RGB 写到 LeRobot 规范目录；按 link mode 处理。
- 过大的 body state、depth、segmentation、point/voxel 放 sidecar，parquet 只存索引/offset，避免单行过宽。

### Step 6：写扩展几何数据

- `released` 模式解码 lossy MP4，并写 quality、valid/confidence mask。
- `regenerated` 模式写 lossless depth/segment 和相机标定。
- 所有 sidecar 原子写入，完成后计算 hash；支持断点续跑。
- point/ray/occupancy 默认在线派生；若离线缓存，必须记录算法配置和输入 hash。

### Step 7：生成 GEAR 元数据和统计量

- 生成标准 `info.json`、`tasks.jsonl`、`episodes.jsonl`、`embodiment.json`、`modality.json`。
- 统计仅使用训练 split；G1/XHand 分开。
- policy state/action 使用各字段合理 normalization；SE(3)、K、ID、mask 不进入普通 z-score。
- 现有 `convert_lerobot_to_gear.py` 主要生成标准 meta，且对扩展模态和多列 state/action 自动发现不足。建议由新转换器直接生成完整 meta，或先扩展该脚本再调用，不能期待它把原始 MobileManiBench 转成 LeRobot。

### Step 8：实现专用 loader

`MobileManiBenchSingleDataset` 应在现有 LeRobot loader 上增加：

- 根据 frame index 读取 depth/seg/geometry sidecar；
- 展开静态 K、robot schema，并读取动态外参；
- 返回 `geometry_inputs`、`geometry_targets`、`aux_targets`、`valid_masks`，不把它们全部拼进 policy state；
- 保持视频、深度、分割、K 的增强严格同步；
- resize/crop 后更新 K；segmentation 只用 nearest；RGB color jitter 不作用于 depth；
- 默认禁用 horizontal flip，除非同时正确变换 K、外参、动作和左右语义；
- 以 role 阻止 `privileged/eval_only` 字段进入部署策略输入；
- delta index 始终先作用于 frame index，再映射 sidecar，防止未来泄漏。

## 9. DreamZero 配置与分阶段训练

### Stage 0：DreamZero RGB-action 基线

先用源数据已有的 `eef_delta_base + end_effector_target` 跑通 head/wrist RGB、语言、base/EEF/active-joint state 的兼容基线。该阶段用于验证转换、时间对齐和现有 DreamZero 训练链路，不代表研究方案的最终动作接口。配置要点：

```yaml
relative_action: false
relative_action_keys: []
fps: 30         # 物理/控制语义；媒体读取仍遵循 timestamp
max_action_dim: 18  # 仅兼容基线：共享 head 时覆盖 XHand；有效维 G1=7、XHand=18
```

如果为两个 embodiment 使用独立 action head，模型有效维可分别设为 7 和 18；如果共用 18/32 维 head，则必须携带 embodiment-specific action mask，padding 不参与 loss。注意：实际 target 中前 6 维是物理 delta、末端维是 joint target，二者量纲不同；应按 slice 统计/归一化，不做整向量统一缩放。

研究版有两个输出分支：base 为 4 维；manipulator 为 `9 (EEF xyz+rot6d) + D (hand)`，即 G1 10 维、XHand 21 维。如果因现有 DreamZero 接口需要把两路临时拼成一个 action token，总有效维为 G1 14、XHand 25，现有 `max_action_dim: 32` 可以容纳。无论一个还是两个 projection head，都必须按 base/EEF position/EEF rotation/hand slice 使用不同 normalization 和 loss，并提供 embodiment mask 与 horizon valid mask。

### Stage 1：VGGT 2D/3D tokenizer

输入/监督：

- 双目 RGB；
- verified K 和逐帧 `T_B0_Optical`；
- 第一版使用 MP4 解码的 coarse pseudo-range depth + confidence/valid mask；
- 第一版使用 MP4 近似 segmentation/动态 mask，并将边界不确定区域标为 unknown；
- 可选 robot body kinematics 用于自遮挡、机器人占据；
- 由这些字段生成高置信区域的 cross-view range depth、masked-view、feature rendering 和 soft occupancy 等目标。

这一阶段采用“2D video tokens 为主、3D tokens 提供粗粒度空间理解”的目标优先级：先保证 video reconstruction/latent statistics 不弱于 2D-only baseline，再逐步增加 pseudo-depth 等几何损失。geometry loss 使用较小权重 warm up，并监控共享 VGGT backbone 上的 gradient norm；若 2D 指标明显下降，降低几何权重或使用 3D adapter/部分 stop-gradient。该阶段不需要把 future action 暴露给视觉 tokenizer；所有目标按窗口末端和 mask 生成，明确区分 condition/target。

第一版验收不是要求厘米级重建，而是要求：2D 质量不显著退化、3D tokens 能解码 coarse range、相机运动下静态区域基本一致，并且 2D+3D WAM 相比 2D-only WAM 在 action/video prediction 或强视角变化场景有可测增益。

### Stage 2：3D-WAM 联合训练

条件：历史 video/3D tokens、机器人 state、语言。目标：未来 video、future 3D geometry/depth/occupancy/flow，以及由 `base plan + manipulator plan` 组成的两路结构化 action plan。manipulator plan 内部联合预测 EEF pose 和 hand configuration。关键约束：

- 所有 future geometry 只作 target，不进入 condition；
- base waypoint 与 manipulator plan 使用同一组 `waypoint_offsets` 与 horizon valid mask；
- action plan 统一表达在锚点 B(t)，避免每个未来步使用变化坐标系造成语义漂移；
- 源 action 采用已验证的 next-step alignment，作为辅助控制监督或 baseline；
- 一步 realized base motion 用作 waypoint/ego-motion 一致性 auxiliary target，不替代 waypoint plan；
- 损失按 `depth_valid`、visibility、dynamic、unknown、field availability mask 加权；
- released lossy depth 和 regenerated GT 不混用同一个无区别的质量权重。

### Stage 3：控制/评估

- 策略主输出只有两路：未来底盘 waypoint 序列，以及联合包含未来 EEF pose 与夹爪/灵巧手构型的 manipulator plan。
- 导航/底盘控制器跟踪 base waypoints；移动操作控制器或 IK 根据 base waypoint 与 EEF plan 联合求解底盘和右臂关节目标。
- 可保留 EEF-delta baseline head 或一步 base-motion auxiliary head，但二者都不替代主规划输出。
- success、object goal、scene asset 和未来状态仅用于评估，不能泄漏到 observation。

## 10. 切分策略

不能随机按 episode 切分，因为同一物体、场景和几乎相同轨迹会泄漏到验证集。推荐：

1. 优先复用官方 `room_infos.split` 和资产 seen/unseen 定义；
2. 若开源子集只有 train，则以 `object_instance_id + scene_id` 为 group 做 train/val/test；
3. 单独报告：seen-object/seen-scene、unseen-object、unseen-scene、unseen-category；
4. G1 与 XHand 可各自评估，跨 embodiment 迁移另设 split，不把它当普通 IID 验证；
5. stats 只由 train group 计算。

## 11. 验收与 QA 清单

### 11.1 文件和时间同步

- [ ] 每 episode 的 state 长度等于 6 路视频帧数和全部 sidecar 帧数。
- [ ] `frame_index` 连续，`source_step` 和两个 timestamp 单调。
- [ ] 随机抽取首/中/末帧，通过 DreamZero loader 解码到正确源帧。
- [ ] repaired 30 FPS 模式没有复制、丢失或重排帧。

### 11.2 动作和机器人状态

- [ ] G1/XHand 的 action、joint、body shape 与 robot schema 一致。
- [ ] G1/XHand 的 end-effector indices 由 joint names 解析，维度分别为 1/12；未使用“最后若干维”的隐式假设。
- [ ] `robot_hand` 的 source link、目标 EEF/TCP frame 和 `T_robotHand_TCP` 已登记，并用前向运动学或可视化验证。
- [ ] action shift 的 `source_index` 可审计，末帧规则明确。
- [ ] EEF delta 积分与下一帧 hand pose 的误差分布合理。
- [ ] 每个锚点的 base waypoint 与 future `robot_base` 重建一致，episode 尾部 valid mask 正确。
- [ ] EEF plan 与 future 目标 EEF/TCP pose 重建一致，并与 base waypoint 使用同一锚点坐标系。
- [ ] manipulator plan 的 hand slice 与同一 `t+kh` 的实际 end-effector joint position 一致，不误取 commanded hand target。
- [ ] waypoint offsets 对应正确的 30 Hz control time，训练和部署配置一致。
- [ ] base/EEF 相对轨迹可视化已确认移动平面轴、yaw 正方向、rpy 顺序和共同世界系。
- [ ] base SE(2) delta 与 base twist 积分一致；末帧 invalid。
- [ ] auxiliary commanded hand target 反归一化后不越 joint limit，并与实际 hand configuration 分开统计 tracking error。
- [ ] waypoint 标签注明由 future realized trajectory 派生，没有被描述成数据集原生 commanded waypoint。

### 11.3 几何和相机

- [ ] K、图像尺寸、crop/resize 后 K 一致；`fx/fy > 0`。
- [ ] 所有 rotation 正交且 determinant 接近 1。
- [ ] 深度反投影再投影的像素误差达到阈值。
- [ ] 头/腕点云在 B0 合并后，静态背景和机器人 mesh 基本重合。
- [ ] segment 与 RGB/深度边界对齐；lossy 数据的不确定边缘被 mask。
- [ ] world/camera/OpenCV convention 通过数值实验而非目测确认。

### 11.4 数据泄漏和训练冒烟

- [ ] `privileged`、success、goal、future pose 不在 policy condition 中。
- [ ] split 中无相同 object-instance/scene group 交叉。
- [ ] normalization 只用 train split，mask/ID/K/SE(3) 不被错误归一化。
- [ ] 先用 10 个 episode 跑 dataset/dataloader 单测和单 batch forward。
- [ ] 再跑小规模 overfit，确认 RGB-action loss 和几何 loss 都能下降。
- [ ] 可视化同一 batch 的 RGB、depth、seg、ray、点云、动作和 base trajectory。

建议额外生成 `conversion_report.json`，汇总 episode 数、失败清单、shape/fps 分布、缺失字段、动作重建误差、相机重投影误差、depth quality 分布和各 split 统计。报告分别维护 `coarse_3d_readiness` 与 `precision_3d_readiness`：MP4、K、convention 和 mask 通过验收后可将前者标为 true；只有 lossless depth 与高精度几何 QA 通过后才可将后者标为 true。

## 12. 实施优先级

建议按以下顺序落地：

1. **P0：schema 与审计器**——冻结 G1/XHand joint/body map，完成全量 inventory、视频帧数和 action alignment 报告。
2. **P0：RGB-action 基线转换**——生成两个标准 GEAR 数据集，跑通现有 DreamZero 小规模训练。
3. **P0：相机标定验证**——从仿真 API 导出 K 和 optical convention，完成重投影 QA。
4. **P0：MP4 coarse geometry 路径**——解码 pseudo-range/segmentation，生成 confidence mask，并验证相机 ray convention。
5. **P1：扩展 loader 与同步增强**——读取动态外参、depth、segment、mask 和几何 sidecar。
6. **P1：VGGT tokenizer 数据目标**——按 B0 生成高置信 ray/cross-view depth、masked-view 和 soft occupancy target。
7. **P1：3D-WAM 联合训练**——以 2D 为主、3D 为空间辅助，完成 2D-only 与 2D+3D ablation。
8. **P2：lossless 几何增强**——重放或获取官方 NPZ，用于校准 MP4 误差和高精度 metric/occupancy 实验。
9. **P2：接触/可达性扩展**——仅在能从仿真导出可信 contact/collision 后加入。

完成 P0 的判定标准不是“文件已转完”，而是：现有 DreamZero 能训练、动作对齐已数值验证、相机坐标 convention 已重投影验证、MP4 pseudo-range 的 mask/误差特征已记录，并且 `coarse_3d_readiness=true`。这样既保留原 DreamZero 训练兼容性，也能先验证 geometry-aware 3D tokens 是否为 WAM 带来空间理解增益；高精度 metric 结论留到 P2 lossless 增强后评估。
