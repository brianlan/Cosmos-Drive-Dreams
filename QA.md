These models are shown on the README.md, which should I donwload?
---

• For style transfer with your HDMap controls:

  - Single-view generation (step 4): Use Cosmos-Transfer1-7B-Sample-AV. That’s the model you need to
    download and point --checkpoint_dir to.
  - Multiview expansion (step 5, optional): Use Cosmos-7B-Single2Multiview-Sample-AV.

  You do not need the base_model, multiview base, annotate, or LiDAR-GEN for this pipeline.

  Suggested layout after download:

  checkpoints/
    cosmos_transfer1_7b_sample_av/    # weights from Cosmos-Transfer1-7B-Sample-AV
    cosmos_single2multiview_7b/       # weights from Cosmos-7B-Single2Multiview-Sample-AV (only if
  you’ll do multiview)

  Once Cosmos-Transfer1-7B-Sample-AV is in place and cosmos-transfer1 is on PYTHONPATH, rerun the
  single-view command (it will now find the package and the checkpoint). If you want, tell me where
  you’ll place the checkpoint folders and I can wire the exact paths into the command.


› Does Multiview expansion require the original multiview images (like back, left_back, etc.)? Or it
  only depend on the front view?


Does Multiview expansion require the original multiview images (like back, left_back, etc.)? Or it only depend on the front view?
---

• Multiview expansion does not need the original RGB multiview shots. The pipeline is:

  - Step 4: generate a single front-view RGB with Cosmos-Transfer1-7B-Sample-AV.
  - Step 5: Cosmos-Single2Multiview takes that front-view output plus your multiview control signals
    (HDMap per camera) to synthesize the other views.

  So you only need:

  - The front-view single_view output.
  - The rendered control videos for all cameras (HDMap/world-scenario). No original multiview RGB is
    required.

帮我研究一下这个codebase中的代码和README，我想知道，要启动这一整套流程最终生成某种指定style的多视图的4D数据(连续帧的3D目标和3D map、自车轨迹、各种sensor内外参等)，我需要准备什么样的数据作为输入？我给了输入之后，我的数据被它的处理流程是怎样的？是否我的输入数据完整性不同的时候，有不同的流程？
---
● 根据我对代码库和README的深入研究，我来为你详细解答：

  一、需要准备的输入数据

  核心必需数据（RDS-HQ格式）

  <INPUT_ROOT>/
  ├── all_object_info/          # 3D边界框
  ├── 3d_lanelines/             # 车道线
  ├── 3d_road_boundaries/       # 道路边界
  ├── pose/                     # 相机位姿 (4x4矩阵, 30fps)
  ├── ftheta_intrinsic/         # 相机内参
  └── captions/                 # 文本描述

  可选但推荐的数据

  ├── 3d_road_markings/         # 道路标记
  ├── 3d_crosswalks/            # 人行横道
  ├── 3d_poles/                 # 杆状物
  ├── 3d_traffic_signs/         # 交通标志
  ├── lidar_raw/                # LiDAR点云
  └── vehicle_pose/             # 车辆位姿

  关键文件格式：
  - .tar 文件：使用WebDataset格式存储，内部是.npy（位姿）或.json（bbox）
  - 位姿：shape (T, 4, 4)，相机到世界坐标变换矩阵
  - Bbox JSON：包含 centroid、size、heading、velocity 等

  ---
  二、完整数据处理流程

  ┌─────────────────────────────────────────────────────────────┐
  │  步骤0: 数据格式转换（如需要）                                 │
  │  convert_waymo_to_rds_hq.py 或 convert_ruqi_to_rds.py       │
  └─────────────────────────────────────────────────────────────┘
                             ↓
  ┌─────────────────────────────────────────────────────────────┐
  │  步骤1: 渲染条件视频（HDMap/LiDAR/World Scenario）            │
  │  render_from_rds_hq.py                                      │
  │  输出: outputs/hdmap/<camera>/*.mp4                         │
  └─────────────────────────────────────────────────────────────┘
                             ↓
  ┌─────────────────────────────────────────────────────────────┐
  │  步骤2: Prompt重写（可选，风格转换）                           │
  │  rewrite_caption.py                                         │
  │  支持: Golden hour, Rainy, Snowy, Night, Foggy等            │
  └─────────────────────────────────────────────────────────────┘
                             ↓
  ┌─────────────────────────────────────────────────────────────┐
  │  步骤3: 单视图视频生成（前视）                                 │
  │  generate_video_single_view.py                              │
  │  模型: Cosmos-Transfer1-7B-Sample-AV                        │
  │  输出: 121帧RGB视频 (24fps)                                  │
  └─────────────────────────────────────────────────────────────┘
                             ↓
  ┌─────────────────────────────────────────────────────────────┐
  │  步骤4: 多视图扩展（可选）                                    │
  │  generate_video_multi_view.py                               │
  │  模型: Cosmos-7B-Single2Multiview-Sample-AV                │
  │  输出: 6个视图的RGB视频 + 网格拼接                           │
  └─────────────────────────────────────────────────────────────┘

  ---
  三、不同输入数据完整性下的流程差异
  场景: A: 完整数据
  数据要求: HDMap全层 + Bbox + Pose + LiDAR
  支持的功能: 全部功能
  处理方式: 所有渲染模式都可用
  ────────────────────────────────────────
  场景: B: 无LiDAR
  数据要求: HDMap + Bbox + Pose
  支持的功能: ✓ HDMap渲染✗ LiDAR渲染✓ 单/多视图生成
  处理方式: controlnet_specs中只包含hdmap
  ────────────────────────────────────────
  场景: C: 最小化数据
  数据要求: 部分HDMap层 + Bbox + Pose
  支持的功能: ✓ 部分HDMap渲染✓ 单/多视图生成
  处理方式: 缺失的HDMap层被跳过，代码会打印警告
  ────────────────────────────────────────
  场景: D: 只有RGB
  数据要求: 仅RGB视频
  支持的功能: ✗ 不可用
  处理方式: 需要先提取标注转换为RDS-HQ格式
  数据转换工具

  公开数据集 → RDS-HQ：
  - convert_waymo_to_rds_hq.py：Waymo Open Dataset转换
  - 支持 pinhole → f-theta 相机模型转换
  - 自动插值位姿到30fps

  自有数据 → RDS-HQ：
  需要准备：
  1. 相机内参（f-theta或pinhole）
  2. 相机/车辆位姿（4x4变换矩阵）
  3. 3D bbox（每帧物体检测）
  4. HDMap 3D数据（多边形/多段线）

  ---
  四、Style和多视图4D数据生成

  Style转换方式

  1. Prompt重写：使用rewrite_caption.py配合Qwen3-VLM
    - 支持：时间、天气、车辆颜色变化
  2. ControlNet参数：调整控制信号强度和CFG引导

  多视图生成

  - 方式1：单视图→6视图扩展（Cosmos-Transfer1）
  - 方式2：直接多视图生成（Cosmos-Transfer2.5）

  4D连续性保证

  - 时间：121帧@24fps，模型保证帧间连续性
  - 空间：通过pose约束保证多视图几何一致性
  - 动态物体：Bbox插值 + 运动补偿
