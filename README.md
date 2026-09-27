# 双 SLM 光学实验硬件工具说明

本文对应 `tools` 中现有脚本，说明 DMD 输入、双 SLM 相位显示、DVP2 相机采集与 CCD 图像整理流程。命令参数按源码核对，未在本次文档编写中连接或测试硬件。

## 代码结构与开发约定

本次重构移除了根目录旧脚本，不提供兼容入口。请在项目根目录用 `python -m tools.<目录>.<模块>` 运行下文命令，不要直接执行包内文件。

```text
tools/
  apps/                     # 参数解析、预览及多设备运行流程
  devices/
    zkwx_slm/               # 中科微兴 SLM1
      sdk.py                # ctypes 绑定、设备枚举及底层控制
      controller.py         # 图像校验、显示选择等业务操作
    magicholo_slm/           # MagicHolo SLM2，同样分 sdk / controller
    dvp2_camera/            # DVP2 相机，同样分 sdk / controller
    dmd_display/
      controller.py         # Tkinter 全屏显示、显示器枚举
  processing/               # 棋盘格生成、CCD 视频校正、图像旋转
  utils/
    paths.py                # 项目、tools、SDK 根路径与路径解析
    images.py               # 数字文件名排序
  tests/                    # 不连接硬件的回归测试
  3rdparty/                 # 厂商原始库，禁止修改
```

新增硬件请放到 `devices/<硬件名称>/`，厂商接口放在 `sdk.py`，图像处理及设备操作放在 `controller.py`；命令行和多设备编排放在 `apps`。公共函数放在 `utils`。硬件库不依赖应用入口，导入 SDK 模块不会加载 DLL 或连接设备；实例化 SDK 时才加载 DLL。

复用示例（从项目根目录运行）：

```python
from tools.devices.zkwx_slm import ZhongkeTimeoutSDK
from tools.devices.magicholo_slm import HDSLM8BitSDK
from tools.devices.dvp2_camera import DvpApi, CameraSession
from tools.utils.paths import THIRD_PARTY_ROOT
```

SDK 默认位置、输入输出目录及命令参数保持原来的含义。旋转工具仍默认处理 `tools/input`，写入 `tools/input_rotated`。棋盘格工具改名为 `processing/generate_alignment_patterns.py`。

无硬件回归检查：

```powershell
python -m unittest discover -s tools/tests -v
```

## 1. 系统组成与职责

```text
电脑 A：DMD 显示输出 + DVP2 相机
电脑 B：中科微兴 SLM1 + MagicHolo SLM2

输入光场 → DMD 输入编码 → SLM1 → 传播 → SLM2 → 传播 → 相机
                                                         ↓
                                             配准后的单通道光强 → 数字检测器
```

这是软件工具对应的逻辑关系，具体折叠光路、中继倍率、偏振和相机安装以实际平台为准。

| 部件 | 项目配置/角色 | 显示或采集工具 |
| --- | --- | --- |
| DMD | 640×640 输入，硬件像元 5.4 μm | `apps/play_dmd_input_fullscreen.py` |
| SLM1 | 中科微兴，硬件像元 8.0 μm，有效区 432×432 | `apps/play_zkwx_slm1.py` |
| SLM2 | MagicHolo HDSLM45R，硬件像元 4.5 μm，有效区 768×768；脚本按 1920×1080 面板处理 | `apps/play_magicholo_slm2.py` |
| 双 SLM | 同名相位对联动播放 | `apps/play_dual_slms.py` |
| 相机 | DVP2 SDK 支持的工业相机，设备型号以枚举结果为准 | `apps/control_dvp2_camera.py` |

三块调制设备的有效宽度均为 3.456 mm。当前项目数值传播使用 13.0 μm / 6.4 μm 的有效采样和 0.20 m / 0.10 m 的传播距离，波长为 532 nm；这些是模型配置，不能替代实测标定。尤其不能用数值采样间隔计算硬件图案尺寸。

硬件几何来源为 `models/SLM/physical_defaults.py`，模型控制来源为 `models/SLM/config_optical.py`。配置改变后，需重新核对检查点和导出文件。

## 2. 脚本索引

| 脚本 | 用途 |
| --- | --- |
| `processing/generate_alignment_patterns.py` | 生成两台电脑使用的棋盘格、对准图案及物理几何元数据 |
| `apps/play_zkwx_slm1.py` | 通过中科微兴 SDK 显示单图或相位序列 |
| `apps/play_magicholo_slm2.py` | 通过 HDSLM SDK 显示 8 bit BMP/PNG，支持有效区平移 |
| `apps/play_dual_slms.py` | 按文件名主干配对两片 SLM，并按共同间隔推进 |
| `apps/play_dmd_input_fullscreen.py` | 将 DMD 图像按原始像素居中全屏显示 |
| `apps/control_dvp2_camera.py` | 枚举相机、预览、PNG 拍照、AVI 录像、曝光及 ROI 配置 |
| `apps/play_dmd_with_dvp2_camera.py` | DMD 播放后等待稳定，再采集对应相机帧并保存同名 PNG |
| `processing/capture_ccd_video_frames.py` | 从录像中交互选区、透视校正并逐样本保存 CCD 光强 |
| `processing/image_origin_rotate.py` | 旋转图像并保留通道与位深；不是相机配准或标签变换工具 |

主项目根目录的 `src` 中的脚本负责模型相位导出和检测评估；本目录主要负责硬件播放与数据采集。

## 3. 环境与 SDK

以下示例从主项目根目录执行，而不是从 `tools` 目录执行：

```powershell
Set-Location E:\Xiongwq\YOLOv3_SLM
conda activate xwq
python -c "import struct; print(struct.calcsize('P') * 8)"
```

SDK 控制脚本面向 Windows，使用与 DLL 匹配的 64 位 Python。实际环境名称可自行替换；需要 NumPy、Pillow、带 GUI 支持的 OpenCV，以及 DMD 窗口使用的 Tkinter。相位生成、导出和检测还需要主项目依赖。不要将 `opencv-python-headless` 当作交互预览所需的 GUI 版本。

| 设备 | 默认 SDK 位置（相对项目根目录） | 覆盖参数 |
| --- | --- | --- |
| 中科微兴 | `tools/3rdparty/python_VS2015_x64`，包含 `SecondDll.dll` 及依赖 | 单片 `--sdk-dir`；双片 `--zkwx-sdk-dir` |
| MagicHolo | `tools/3rdparty/HDSLM_SDK_0612/HDSLM_API/Bin`，包含 `HDSLMFunc.dll` | 单片 `--sdk-dir`；双片 `--magicholo-sdk-dir` |
| DVP2 | `tools/3rdparty/DVP2 SDK CN/library/Visual C++/bin/x64/DVPCamera64.dll` | `--dll` |

相机脚本用 `ctypes` 调用官方 DLL，没有使用随附的 Python 3.6 专用 `dvp.pyd`。安装设备驱动及 SDK 所需运行库，并保留 DLL 配套依赖；发现 DLL 文件不等于依赖已完整。

将设备作为扩展显示屏连接，先枚举索引再播放。中科微兴使用 Windows Monitor 索引，MagicHolo 使用 HDSLM SDK display ID，两个数字不能互换。DMD 工具也是显示输出程序，不是 DMD 厂商的高速二值图案/触发控制接口。

## 4. 首次连接与对准

### 4.1 枚举设备

在各设备所在电脑运行对应命令：

```powershell
python -m tools.apps.play_dmd_input_fullscreen --list-monitors
python -m tools.apps.control_dvp2_camera --list
python -m tools.apps.play_dual_slms --list-devices
```

也可单独枚举 SLM：

```powershell
python -m tools.apps.play_zkwx_slm1 --list-monitors
python -m tools.apps.play_magicholo_slm2 --list-displays
```

显示器编号会受接线和系统设置影响。下面示例的 DMD `--monitor 1` 只是示例，应替换为枚举得到的索引。纯 DMD 播放默认索引为 1，DMD+相机脚本默认索引为 0，因此实验时建议显式指定。

### 4.2 生成对准包

```powershell
python -m tools.processing.generate_alignment_patterns --output output/alignment --target all --blocks-y 10 --blocks-x 10
```

输出目录结构：

```text
output/alignment/
  alignment_geometry.json
  dmd_camera_pc/           # 复制到 DMD/相机电脑
  dual_slm_pc/
    slm1/                 # 与 slm2 中的同名文件配对
    slm2/
```

`--target dmd` 或 `--target slms` 可仅生成对应设备包。默认 SLM 棋盘格灰度为 0 和 128；128 只在理想线性 0～2π 响应下约对应 π，真实平台应根据实测灰度—相位 LUT 选择。

```powershell
python -m tools.apps.play_dual_slms --zkwx-input output/alignment/dual_slm_pc/slm1/checkerboard.png --magicholo-input output/alignment/dual_slm_pc/slm2/checkerboard.png --dry-run
python -m tools.apps.play_dual_slms --zkwx-input output/alignment/dual_slm_pc/slm1/checkerboard.png --magicholo-input output/alignment/dual_slm_pc/slm2/checkerboard.png
python -m tools.apps.play_dmd_input_fullscreen --input output/alignment/dmd_camera_pc --monitor 1
```

双片工具默认打开窗口后等待在终端按 Space 才发送首对相位。需要保持某个对准图案时使用单图输入；目录输入会按序播放多个图案。

### 4.3 对准与光度标定

先核对照明和偏振、逐片灰度—相位响应，再检查有效区居中、平移、方向与相机 ROI。保持曝光和增益固定，记录暗场、饱和比例及强度响应。必要时测量传播/中继关系与系统 PSF，而不是仅调整模拟采样值使图像看起来相似。

**相位图不能被拉伸到整个面板。** 432×432 和 768×768 是有效区，应按 1:1 器件像素显示；播放器居中放置，外围填充。显示链路本身的缩放、颜色处理或 gamma 仍需核查。播放器不会自动将任意灰度图转换为经过 LUT 标定的相位驱动图。

## 5. 双 SLM 相位播放

假设相位文件已分别放入 `tools/slm1` 和 `tools/slm2`：

```text
tools/slm1/0001.png    tools/slm2/0001.png
tools/slm1/0002.png    tools/slm2/0002.png
```

```powershell
python -m tools.apps.play_dual_slms --zkwx-input tools/slm1 --magicholo-input tools/slm2 --dry-run
python -m tools.apps.play_dual_slms --zkwx-input tools/slm1 --magicholo-input tools/slm2 --interval 5
```

| 参数 | 含义 |
| --- | --- |
| `--zkwx-monitor` | 手动指定中科微兴 Windows 显示索引 |
| `--magicholo-display` | 手动指定 MagicHolo SDK display ID |
| `--start-id 0021`、`--limit 10` | 选择相位序列的起点和数量 |
| `--auto-start` | 不等待 Space，直接发送相位 |
| `--loop` | 循环播放 |
| `--close-after` | 播放结束关闭窗口；默认保持末帧并等待 Enter |
| `--magicholo-offset-x/y` | 按 SLM2 像素平移，正值向右/向下 |
| `--magicholo-background` | SLM2 有效区外围的 8 bit 灰度值 |

播放过程中在终端按 Space 暂停/继续，Esc 停止。`--dry-run` 会进行设备识别和输入检查，可能仍需要 SDK 与显示设备，但不会打开相位显示窗口。

单片排查可用：

```powershell
python -m tools.apps.play_zkwx_slm1 --input tools/slm1 --interval 5
python -m tools.apps.play_magicholo_slm2 --input tools/slm2 --interval 5
```

MagicHolo 输入限制为 8 bit BMP/PNG。中科微兴支持更多格式，但实验相位建议用无损 PNG/BMP，避免 JPEG 改变灰度值。

**“双片同步”是同一进程顺序调用两个 SDK 并按共同间隔推进，不代表两片器件物理上同时刷新。** 它也没有与另一台电脑的 DMD 建立触发或网络握手。

## 6. DMD 与相机采集

### 6.1 单独预览与记录

```powershell
python -m tools.apps.control_dvp2_camera --camera 0 --output output/camera_check --manual-exposure
python -m tools.apps.control_dvp2_camera --camera 0 --output output/camera_check --manual-exposure --capture 10 --capture-interval 1
python -m tools.apps.control_dvp2_camera --camera 0 --output output/camera_record --record --duration 30
```

`--manual-exposure` 关闭自动曝光；没有给定 `--exposure-us` 时仍需确认当前曝光值。曝光单位为微秒，`--gain` 为设备模拟增益值，具体范围由设备决定。`--roi X Y W H` 是相机硬件 ROI，需符合设备对齐要求，不等于后处理透视配准。

相机预览快捷键：Space/S 保存全分辨率 PNG，R 开始/停止 AVI 录像，A 切换自动曝光，`[`/`]` 减少/增加手动曝光，Esc/Q 退出。`--window-scale` 只改变预览大小，不改变保存分辨率。可用 `--load-config`/`--save-config` 读写 SDK 配置。

### 6.2 DMD 播放并按同名保存相机帧

准备真实硬件导出目录，下例用 `output/hardware_run/input` 表示输入文件夹：

```powershell
python -m tools.apps.play_dmd_with_dvp2_camera --input output/hardware_run/input --output output/hardware_run/ccd_raw --monitor 1 --dry-run
python -m tools.apps.play_dmd_with_dvp2_camera --input output/hardware_run/input --output output/hardware_run/ccd_raw --monitor 1 --display-seconds 2 --settle-seconds 1 --manual-exposure --require-camera
```

默认每张 DMD 图像显示 2 秒，约在切换后 1 秒请求足够新的相机帧；启动时暂停，按窗口中的 Space 开始。保存文件沿用 DMD 文件主干，例如 `0001.png`，并写出 `dvp2_capture_manifest.json`，记录帧 ID、实际采集延迟、曝光、增益和路径。

默认相机初始化不可用时会降级为仅 DMD 播放，使用 `--require-camera` 可在初始化失败时停止。运行中仍可能因超时或缺帧跳过采集，因此播放结束后必须核查 manifest 与图像数量。`--overwrite` 明确允许替换已有同名图像及 manifest，重复试验优先使用新输出目录。

该脚本保存相机输出帧，不自动执行视频工具中的透视矫正、固定黑白电平转换和 640×640 光学区域裁切。因此这里显式使用 `ccd_raw` 保存原始采集；脚本默认目录名为输入旁的 `ccd`，目录名不代表已完成配准。

静态学生实验可先固定一对 SLM 相位，再运行 DMD+相机序列。动态教师实验每张输入对应不同相位，需要额外实现跨电脑同步或逐样本人工确认；只让两个播放程序使用相同间隔不足以保证样本对应。

### 6.3 仅播放 DMD

```powershell
python -m tools.apps.play_dmd_input_fullscreen --input output/hardware_run/input --monitor 1 --interval 5
```

原图按原生像素居中置于黑色画布，不缩放。Space 播放/暂停，左右方向键切图，Home 返回首图，F11 切换置顶，Esc/Q 退出。图像尺寸必须能容纳于目标显示区域。

## 7. 从录像整理配准后的 CCD 光强

```powershell
python -m tools.processing.capture_ccd_video_frames --video output/hardware_run/camera.avi --export-root output/hardware_run --output output/hardware_run/ccd --selection quad --channel green --roi-samples 3
```

`--export-root` 应包含导出的 `input/`，有 `manifest.json` 时优先按其样本记录建立顺序。工具在多个较清晰的调制帧上独立选区，汇总并确认 ROI 后重新从采集起点播放，再由用户逐帧确认与导出 ID 对应的光学图像。

| 参数 | 默认值与用途 |
| --- | --- |
| `--selection` | `quad`：四角透视矫正；`rect`：矩形 ROI |
| `--roi-start-time` | 5 秒，开始选区的位置；别名 `--start-time` |
| `--roi-samples` / `--roi-sample-gap` | 3 次选区，建议间隔 30 秒 |
| `--capture-start-time` | 0 秒，确认 ROI 后返回的采集起点 |
| `--output-size` | 640，矫正后的正方形大小，需匹配检测器配置 |
| `--channel` | 默认 green，可选 gray/red/blue |
| `--black-level` / `--white-level` | 0 / 255，固定 8 bit 光度标定范围 |
| `--start-id` / `--limit` | 选择样本范围 |
| `--reselect-roi` | 忽略已保存 ROI，重新选择 |
| `--overwrite` | 允许覆盖已有确认帧 |

选区阶段按窗口提示选择角点并用 Enter 确认；采集阶段 Enter/C 保存当前帧，Space/P 暂停或继续，Esc/Q 退出。输出为同名 PNG 和 `ccd_manifest.json`，保存配准与采集记录。默认首次成功采集后启用 7 秒自动暂停，可用 `--auto-pause-seconds 0` 关闭。

工具不做逐图 min/max 对比度拉伸；黑白电平应来自固定标定。否则同一模型会看到随图像改变的光度变换。录像采用压缩编码时可能引入强度误差，定量实验应与原始 PNG 采集对照。

## 8. 与模型导出、检测的衔接

1. 用 `src/export_teacher_v2_dataset_hardware.py` 或 `src/export_slm_light_dataset_hardware.py` 从匹配配置的检查点导出输入及相位。详细参数查看 `src/README.md` 和对应脚本 `--help`。
2. 保留导出 `manifest.json`、输入 ID 与相位对应关系。静态相位保持固定，动态相位必须按 ID 配对。
3. 完成设备播放、采集、几何配准与固定光度转换。
4. 用 `src/detect_ccd_light_dataset.py` 对整理后的真实光强执行检测；该步骤与运行光学学生仿真的 `src/evaluate_slm_light_dataset.py` 不同，不能将仿真评估结果写成实物性能。

部分工具默认路径仍指向历史 `Tv2_dmd640_scratch/hardware_export_100`。正式实验显式传入路径，不能因为默认目录能运行就认为检查点和相位匹配。

`processing/image_origin_rotate.py` 没有命令行参数，直接运行会将 `tools/input` 旋转 -90° 后写入 `tools/input_rotated`。90° 整数倍使用无插值旋转；其他角度使用最近邻并扩大画布。它不更新检测标签、配准矩阵或 manifest，因此不能把任意旋转当成已完成的坐标标定。

## 9. 常见问题

| 现象 | 检查方向 |
| --- | --- |
| DLL 存在但加载失败 | Python 位数、SDK 配套 DLL、运行库、设备驱动；路径含空格时加引号 |
| 相位显示到错误屏幕 | 重新枚举；区分 Windows monitor 与 HDSLM display ID |
| MagicHolo 不接受图像 | 检查 BMP/PNG、8 bit 灰度、图像尺寸和 SDK 返回状态 |
| 双片文件无法配对 | 两目录的文件名主干、重复 ID、`--start-id` 与 `--limit` |
| DMD 正常但无相机 PNG | 检查初始化降级、运行时超时、旧文件冲突和 manifest；不要仅看播放窗口 |
| 光强尺寸正确但检测异常 | 核对 ROI、透视、方向、通道、黑白电平、曝光、输入 ID 和训练配置 |
| 相位图被拉伸 | 检查导出有效区、播放器、桌面缩放和设备显示链路 |
| 同名文件但目标对不上 | 同名只保证文件命名；动态相位跨电脑同步仍需独立验证 |

每次实验建议记录：代码版本、检查点哈希、相位/输入文件及清单、设备与 SDK 版本、显示索引、LUT、传播/中继配置、相机曝光和增益、ROI/变换、每帧采集时间以及缺帧情况。

## 10. 文档与版本管理

当前 `tools` 内有独立 `.git`，是嵌套仓库。其 `.gitignore` 忽略 `*.md`，所以本文创建后默认不会出现在该仓库的未跟踪列表中；如需随硬件工具提交，应在该仓库为 `README.md` 添加忽略例外或显式强制添加。本文未修改忽略规则、SDK 或设备脚本。
