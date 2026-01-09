# Cosmos-Drive-Dreams 模型部署和运行指南

本文档介绍如何从已有的 RDS 数据和 HDMap 条件视频开始，部署模型并生成多视图4D驾驶场景视频。

---

## 快速开始摘要

核心步骤：
1. **环境安装** → 运行 `conda env create` 和安装依赖
2. **模型下载** → 下载 Cosmos-Transfer1-7B-Sample-AV 和 Single2MultiView 两个模型
3. **准备Captions** → 为每个clip写简单的场景描述 (`.txt`文件)
4. **创建配置** → 修改 `multi_view_spec.json` 指向你的HDMap视频路径
5. **运行生成** → 先单视图，后多视图扩展

---

## 前提条件

### 数据准备状态
- ✅ 已有 RDS 数据 (通过 `convert_ruqi_to_rds.py` 转换完成)
- ✅ 已有 HDMap 条件视频 (通过 `render_from_rds_hq.py` 渲染完成)

---

## 第一步：环境准备

### 1.1 系统要求
- **操作系统**: Ubuntu 20.04.5 或 22.04.5
- **Python**: 3.10.x 或 3.12
- **GPU**: 推荐 80GB VRAM (A100/H100)，最低 32GB VRAM
- **存储**: 至少 300GB 可用空间
- **内存**: 至少 64GB RAM

### 1.2 安装环境
```bash
# 1. 克隆代码仓库（如果还没有）
git clone git@github.com:nv-tlabs/Cosmos-Drive-Dreams.git
cd Cosmos-Drive-Dreams
git submodule update --init --recursive

# 2. 创建conda环境
conda env create --file environment.yaml
conda activate cosmos-drive-dreams

# 3. 安装依赖
pip install -r requirements.txt

# 4. 安装 vllm
pip install https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.5%2Bcu128torch2.7-cp38-abi3-linux_x86_64.whl
export VLLM_ATTENTION_BACKEND=FLASHINFER
pip install vllm==0.9.0

# 5. 修复 Transformer engine 链接问题
ln -sf $CONDA_PREFIX/lib/python3.12/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/
ln -sf $CONDA_PREFIX/lib/python3.12/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/python3.12

# 6. 安装 Transformer engine
pip install transformer-engine[pytorch]==2.4.0
```

---

## 第二步：模型下载

### 2.1 需要下载的模型（多视图场景，两者都需要）

| 模型 | 用途 | 必需性 | Hugging Face 链接 |
|------|------|--------|------------------|
| Cosmos-Transfer1-7B-Sample-AV | 单视图视频生成 | ✅ 必需 | https://huggingface.co/nvidia/Cosmos-Transfer1-7B-Sample-AV |
| Cosmos-Transfer1-7B-Single2MultiView | 单视图→多视图扩展 | ✅ 必需 | https://huggingface.co/nvidia/Cosmos-Transfer1-7B-Sample-AV-Single2MultiView |

### 2.2 下载前准备

1. **生成 Hugging Face access token** (需要 'Read' 权限)
   - 访问: https://huggingface.co/settings/tokens

2. **登录 Hugging Face**
   ```bash
   huggingface-cli login
   ```

3. **接受模型许可条款** (必须手动在网页上操作)
   - Llama-Guard-3-8B: https://huggingface.co/meta-llama/Llama-Guard-3-8B
   - Cosmos-Tokenize1-CV8x8x8-720p: https://huggingface.co/nvidia/Cosmos-Tokenize1-CV8x8x8-720p
   - Cosmos-Guardrail1: https://huggingface.co/nvidia/Cosmos-Guardrail1
   - Cosmos-Transfer1-7B-Sample-AV: https://huggingface.co/nvidia/Cosmos-Transfer1-7B-Sample-AV

### 2.3 执行下载
```bash
cd cosmos-transfer1
PYTHONPATH=$(pwd) python scripts/download_checkpoints.py --output_dir checkpoints/ --model 7b_av
cd ..
```

下载后目录结构：
```
checkpoints/
├── nvidia/
│   ├── Cosmos-Guardrail1/
│   ├── Cosmos-Transfer1-7B-Sample-AV/
│   │   ├── base_model.pt
│   │   ├── hdmap_control.pt
│   │   └── lidar_control.pt
│   ├── Cosmos-Transfer1-7B-Sample-AV-Single2MultiView/
│   │   ├── t2w_base_model.pt
│   │   ├── t2w_hdmap_control.pt
│   │   ├── v2w_base_model.pt
│   │   └── v2w_hdmap_control.pt
│   └── Cosmos-Tokenize1-CV8x8x8-720p/
└── ...
```

---

## 第三步：准备 Captions

**重要**: 每个RDS clip需要一个对应的caption文件，描述场景内容。

### 3.1 创建原始caption文件

**方式A：手动编写** (推荐用于开始测试)

为每个clip创建一个简单的文本描述，保存为 `<clip_id>.txt`：
```
outputs/captions/
└── <clip_id>.txt
```

caption内容示例：
```
A car driving on a highway with multiple lanes. There are several other vehicles on the road,
including trucks and sedans. The weather is clear and the road surface is dry.
```

**编写建议**：
- 描述主要场景类型（highway, city street, rural road等）
- 描述其他车辆和物体
- 描述天气条件
- 描述道路状况
- 1-3句话即可

**方式B：使用示例caption格式参考**
```bash
# 查看示例caption格式
cat assets/example/captions/<sample_id>.txt
```

### 3.2 生成风格变体（可选，用于批量生成不同风格）

有了原始caption后，可以使用VLM自动生成多种风格变体：
```bash
python scripts/rewrite_caption.py \
    -i outputs/captions \
    -o outputs/captions
```

生成的JSON文件格式：
```json
{
    "Original": "原始场景描述...",
    "Rainy": "雨天场景描述...",
    "Night": "夜晚场景描述...",
    "Snowy": "雪天场景描述...",
    "Golden hour": "黄金时刻场景描述...",
    "Foggy": "雾天场景描述...",
    "Morning": "早晨场景描述...",
    "Sunny": "晴天场景描述..."
}
```

支持的转换类型：
- **时间**: Golden hour, Morning, Night
- **天气**: Rainy, Snowy, Sunny, Foggy

### 3.3 文件结构要求

最终caption目录结构：
```
outputs/captions/
├── <clip_id_1>.txt          # 原始文本描述
├── <clip_id_1>.json         # 风格变体（如果使用了rewrite_caption.py）
├── <clip_id_2>.txt
└── <clip_id_2>.json
```

**注意**: 如果只有`.txt`文件，脚本会使用原始描述。如果有`.json`文件，会为每种风格生成一个视频。

---

## 第四步：创建配置文件

### 4.1 单视图配置 (`single_view_spec.json`)
```json
{
    "hdmap": {
        "control_weight": 1,
        "input_control": "outputs/hdmap/ftheta_camera_front_wide_120fov/<sample_name>_0.mp4"
    }
}
```

### 4.2 多视图配置 (`multi_view_spec.json`)
```json
{
    "hdmap": {
        "control_weight": 1,
        "input_control": [
            "outputs/hdmap/ftheta_camera_front_wide_120fov/<sample_name>_0.mp4",
            "outputs/hdmap/ftheta_camera_cross_left_120fov/<sample_name>_0.mp4",
            "outputs/hdmap/ftheta_camera_cross_right_120fov/<sample_name>_0.mp4",
            "outputs/hdmap/ftheta_camera_rear_tele_30fov/<sample_name>_0.mp4",
            "outputs/hdmap/ftheta_camera_rear_left_70fov/<sample_name>_0.mp4",
            "outputs/hdmap/ftheta_camera_rear_right_70fov/<sample_name>_0.mp4"
        ],
        "ckpt_path": "nvidia/Cosmos-Transfer1-7B-Sample-AV-Single2MultiView/t2w_hdmap_control.pt"
    }
}
```

**参数说明**：
- `control_weight`: 控制信号强度 (0.3-1.0)，越大越严格遵循条件
- `input_control`: HDMap 条件视频路径

---

## 第五步：运行生成

### 5.1 单视图视频生成
```bash
PYTHONPATH="cosmos-transfer1" python scripts/generate_video_single_view.py \
    --caption_path outputs/captions \
    --input_path outputs \
    --video_save_folder outputs/single_view \
    --checkpoint_dir checkpoints/ \
    --is_av_sample \
    --controlnet_specs single_view_spec.json \
    --num_steps 35 \
    --guidance 5.0 \
    --num_gpus 1
```

**关键参数**：
- `--num_steps`: 扩散采样步数 (默认35，越多质量越好但越慢)
- `--guidance`: CFG引导强度 (默认5.0，越高越遵循prompt)
- `--num_gpus`: GPU数量 (80GB GPU默认1即可)

**输出**：
- `outputs/single_view/<sample_name>_<style>.mp4` (121帧, 24fps)

### 5.2 多视图扩展（必需）
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH="cosmos-transfer1" python scripts/generate_video_multi_view.py \
    --caption_path outputs/captions \
    --input_path outputs \
    --input_view_path outputs/single_view \
    --video_save_folder outputs/multi_view \
    --checkpoint_dir checkpoints \
    --is_av_sample \
    --controlnet_specs multi_view_spec.json \
    --num_steps 35 \
    --guidance 3.0 \
    --num_gpus 1
```

**输出**：
- `0.mp4` ~ `5.mp4`: 6个视图的视频
- `grid.mp4`: 网格拼接视频 (2×3布局)
- `prompt.json`: 使用的提示词记录

**视图顺序**：
- 0: 前视 (front_wide_120fov)
- 1: 左交叉 (cross_left_120fov)
- 2: 右交叉 (cross_right_120fov)
- 3: 后视 (rear_tele_30fov)
- 4: 左后 (rear_left_70fov)
- 5: 右后 (rear_right_70fov)

---

## 关键文件路径

| 文件/目录 | 路径 |
|----------|------|
| 生成脚本 | `scripts/generate_video_single_view.py` |
| 多视图脚本 | `scripts/generate_video_multi_view.py` |
| Caption重写 | `scripts/rewrite_caption.py` |
| 配置示例 | `assets/sample_av_hdmap_spec.json` |
| 模型下载 | `cosmos-transfer1/scripts/download_checkpoints.py` |
| 安装指南 | `INSTALL.md` |

---

## 验证测试

### 测试1：使用示例数据验证
```bash
cd cosmos-drive-dreams-toolkits
python render_from_rds_hq.py \
    -i ../assets/example \
    -o ../outputs \
    -d rds_hq_mv \
    --skip lidar
cd ..

python scripts/rewrite_caption.py \
    -i assets/example/captions \
    -o outputs/captions

PYTHONPATH="cosmos-transfer1" python scripts/generate_video_single_view.py \
    --caption_path outputs/captions \
    --input_path outputs \
    --video_save_folder outputs/single_view \
    --checkpoint_dir checkpoints/ \
    --is_av_sample \
    --controlnet_specs assets/sample_av_hdmap_spec.json
```

### 测试2：使用自己的数据
将配置文件中的 `<sample_name>` 替换为你的clip ID，路径指向你的数据。

---

## 故障排查

| 问题 | 解决方案 |
|------|---------|
| 生成速度慢 | 使用多GPU `--num_gpus 4` 或减少 `--num_steps 20` |
| 效果不理想 | 调整 `control_weight` (0.3-1.0) 或 `--guidance` (3.0-7.0) |
| 权限错误 | 确保已接受Hugging Face模型许可条款 |
| Caption格式错误 | 确保caption是.txt或.json格式，内容是有效的英文描述 |
| 找不到HDMap视频 | 检查`multi_view_spec.json`中的路径是否正确指向outputs/hdmap/ |

---

## 参考链接

- **Cosmos-Transfer1 文档**: https://github.com/nvidia-cosmos/cosmos-transfer1
- **Hugging Face**: https://huggingface.co/nvidia/Cosmos-Transfer1-7B-Sample-AV
- **安装指南**: `INSTALL.md`
- **项目README**: `README.md`
