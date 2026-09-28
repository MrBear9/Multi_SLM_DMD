# MainWindow 光学控制台

界面文件为 `MainWindow.ui`，窗口类及 objectName 均为 `MainWindow`。运行控制器位于 `tools/gui/main_window.py`；Qt Designer 仍可直接编辑界面。

## 启动

从项目根目录运行：

```powershell
python -m pip install -r tools/requirements-gui.txt
python -m tools.apps.optical_control
```

不连接设备的演示模式：

```powershell
python -m tools.apps.optical_control --demo
```

也支持直接运行入口文件：

```powershell
python .\tools\apps\optical_control.py --demo
```

以上两种方式使用同一个入口；直接运行文件时会自动定位项目根目录。

载入保存的配置：

```powershell
python -m tools.apps.optical_control --config output/optical_config.json
```

本机的 `xwq` 环境及 `D:\software\python31011\python.exe` 均已检查过 PyQt5、NumPy、Pillow、OpenCV 可用。其他环境缺少依赖时，使用上面的安装命令。

## 使用流程

1. 初始页面填写实验名称、保存目录、各设备输入及 SDK 路径。只勾选这台电脑实际连接的设备。
2. 通过“浏览…”选择文件或目录。DMD、SLM1、SLM2 输入载入后即可在第二页本地预览；未连接时的播放按钮只更新本地预览。
3. 点击“枚举设备”查看 Windows 显示器编号、MagicHolo SDK display ID 和相机编号，再点击“检查参数”。
4. 点击“连接设备”。界面会锁定连接参数并切换到预览页；CCD 在后台开始连续取帧。连接后读取相机实际曝光及增益，不自动写入界面初值。
5. 点击“开始运行”发送首帧并播放。也可单独控制某一设备。启用“双 SLM 同步播放”时，两路相位文件必须具有相同主干名称及顺序，共用 SLM1 播放间隔和帧索引。
6. 需要随 DMD 输入逐帧采集时，勾选位于“双 SLM 同步播放”左侧的“DMD 同步采集”，再播放 DMD。默认不勾选；未勾选时只播放，不自动保存相机帧。播放中锁定该开关，暂停后可切换。
7. CCD 连接后可“应用”曝光 / 增益、“保存单帧”“开始录像”。ROI 在配置页设置，修改后需重新连接。
8. “停止全部”停止图案播放、待处理的自动保存及录像，CCD 预览继续；“断开设备”或关闭窗口停止采集并释放设备。DMD 输出窗口按 Esc 也会断开设备。

## 画面与采集行为

- 第一页为参数配置；第二页按 2×2 显示 DMD、SLM1、SLM2、CCD，每张卡片左侧为大预览，右侧为参数和操作。正方形、横向、竖向图像均保持宽高比。
- DMD、SLM1、SLM2 面板标称均为 1920×1080；640×640、432×432、768×768 是当前常用的有效输入区域，不是面板分辨率。界面根据实际输入显示有效区尺寸，连接后根据设备枚举值显示面板尺寸。
- 默认“有效区域”只放大输入图案，便于检查细节；切换“完整面板”查看黑底面板中的位置与偏移。两种模式均支持“适应窗口”或“1:1 像素”，只影响预览，不改变硬件输出。CCD 直接显示实际采集图像。
- DMD 使用独立 Qt 全屏窗口按原始像素居中输出，不再启动 Tkinter。SLM1 和 SLM2 调用原有 ctypes SDK 封装。两片 SLM 均提供 X/Y 偏移和应用按钮，偏移单位为面板像素，相对居中位置计算，越界会拒绝发送。中科 SLM1 用完整黑底画布实现内部偏移，显示窗口保持在原面板位置；SLM2 使用原有全尺寸灰度画布。
- 同时启用 DMD 和 CCD 并勾选“DMD 同步采集”后，DMD 播放每帧时会在稳定时间之后请求一张新相机帧，保存为 DMD 同名 PNG。采集超出该图案的显示窗口则报告超时，不把过期帧保存到下一张图案名下。
- 这是基于软件计时的采集，不是硬件触发。相机内部缓冲与曝光时序仍需实机标定。双 SLM 与 DMD 各按设置间隔运行；双 SLM 同步选项只同步两片 SLM。
- 手动保存使用时间戳和随机后缀。默认拒绝覆盖文件；循环播放时，同名自动保存也遵循“覆盖同名文件”选项。
- PNG 保存相机原始通道与位深。AVI 使用 MJPG、20 FPS 和 8 bit 预览图像，适合回看；实际采集帧率不足时，AVI 时间长度不代表精确实验时长。
- 勾选“保存采集记录”会在输出目录追加 `captures.jsonl`，记录文件名、相机帧号、曝光、增益、图像格式及当时的输入图案。
- DMD 和 SLM 画面显示软件发送的图案，不是从调制器读取的光学反馈。CCD 显示相机采集帧。

## 参数和设备约定

- 相对路径统一以项目根目录解析；厂商原始文件仍在 `tools/3rdparty`，未改动。
- SLM1 的显示器编号与 SLM2 的 SDK display ID 不同。`-1 / 自动选择` 按原封装规则选择设备；选择到相同显示区域会拒绝连接。
- 仅在勾选“DMD 同步采集”时要求 DMD 稳定时间小于播放间隔，并等待相机第一帧后才允许开始同步播放。SDK 拒绝的曝光、增益、ROI 或偏移会显示错误；ROI 对齐和参数范围以实际设备为准。
- Windows 显示输出使用物理像素；建议将对应扩展显示器设置为原生分辨率和 100% 缩放。
- 保存/载入配置使用带版本号的 JSON。连接状态下禁止载入配置或修改连接参数，需先断开。
- 演示模式生成临时示例图案，CCD 帧带 `DEMO - NO CAMERA` 字样。它不会调用厂商 DLL，也不会在真实设备连接失败时自动启用。保存的记录中包含 `demo: true`。演示输入位于临时目录，不应作为长期实验输入路径。

## 代码结构

```text
tools/
  apps/optical_control.py       # 启动入口、--demo、--config
  gui/main_window.py           # MainWindow 控制器与播放调度
  gui/configuration.py         # 配置序列化、图像与 Qt 转换
  gui/displays.py              # Qt DMD 窗口及两种 SLM 适配器
  gui/camera_worker.py         # 单独线程中的 CCD 取帧、参数、保存、录像
  ui/MainWindow.ui            # Qt Designer 布局
  requirements-gui.txt         # GUI 可选依赖
```

相机 SDK 调用集中在同一个工作线程。GUI 以约 30 Hz 读取最新一帧，避免慢渲染造成消息积压。断开和关闭窗口会等待线程释放相机，不强制终止 SDK 线程。

## 验证

```powershell
python -m unittest discover -s tools/tests -v
```

GUI 测试在 offscreen 模式下运行，覆盖配置、预览、配对、模拟采集、同步开关、不同宽高比的预览、两片 SLM 偏移边界、PNG、AVI、停止、连接失败回滚和线程退出。未安装 GUI 依赖时会跳过 GUI 测试。实际 SDK 调用已接入，但真实设备连接、显示和取帧仍需在实验台验证。
