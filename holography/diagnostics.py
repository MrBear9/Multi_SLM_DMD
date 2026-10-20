"""Human-readable geometry diagnostics; original measurements remain in raw.npy."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def gray_channel(image):
    image = np.asarray(image)
    return image[..., 1] if image.ndim == 3 else image


def display_gray(image, scale):
    value = np.nan_to_num(np.asarray(image, np.float32), nan=0., posinf=scale, neginf=0.)
    return np.uint8(np.rint(np.clip(value / max(float(scale), 1e-12), 0, 1) * 255))


def save_capture_preview(folder, image, white):
    # Fixed white-level scaling makes background/probe PNG brightness comparable.
    Image.fromarray(display_gray(gray_channel(image), white)).save(Path(folder)/'raw_preview.png')


def save_spot_diagnostics(folder, raw, report, point, error=None):
    folder = Path(folder)
    arrays = {'delta', 'smooth', 'compact', 'mask'}
    info = {key:value for key,value in report.items() if key not in arrays}
    info.update(schema=1, target_point_um=list(point), error=error,
                raw_shape=list(raw.shape), raw_dtype=str(raw.dtype),
                coordinates='Original camera pixels: x right, y down; no preview rotation or flip.',
                preview_channel='Green channel for BGR input; otherwise grayscale.',
                raw_preview_scale='0..white_level maps to 0..255; raw.npy retains measured values.',
                difference_definition='max(probe - background, 0); positive difference used by detector.')
    if report.get('detector')=='radial_rings':
        info['difference_definition']='PNG: max(probe-background,0); ring detection uses signed probe-background.'
    scale = max(float(np.max(report['delta'])), 1e-12) if 'delta' in report else 1.
    info['difference_display_max'] = scale
    for key, filename in (('delta','difference.png'),('smooth','difference_smoothed.png')):
        if key in report:
            Image.fromarray(display_gray(report[key], scale)).save(folder/filename)
    if 'compact' in report:
        Image.fromarray(display_gray(report['compact'],max(float(report['compact'].max()),1e-12))).save(folder/'compact_signal.png')
    if 'mask' in report:
        Image.fromarray(report['mask']*255).save(folder/'candidate_mask.png')
    preview = display_gray(gray_channel(raw), report['white_level'])
    overlay = cv2.cvtColor(preview, cv2.COLOR_GRAY2RGB)
    # Limit labels for pathological noisy images; JSON retains every candidate.
    candidates = report.get('candidates', [])
    info['annotated_candidate_limit'] = 20
    for candidate in candidates[:20]:
        rank = candidate['rank']
        x,y,w,h = candidate['bbox_xywh']
        accepted = report.get('status') == 'accepted' and rank == report.get('selected_rank')
        color = (50,255,80) if accepted else (255,180,30) if rank==1 else (255,70,70)
        cv2.rectangle(overlay,(x,y),(x+w-1,y+h-1),color,2)
        cx,cy = candidate['centroid_xy']
        cv2.drawMarker(overlay,(round(cx),round(cy)),color,cv2.MARKER_CROSS,12,1)
        text = f"#{rank} {candidate.get('quality','candidate')} peak={candidate.get('peak_signal',0):.3g}"
        cv2.putText(overlay,text,(max(0,min(x,overlay.shape[1]-300)),max(14,y-5)),
                    cv2.FONT_HERSHEY_SIMPLEX,.45,color,1,cv2.LINE_AA)
    if report.get('search_roi_xywh'):
        x,y,w,h=report['search_roi_xywh']
        cv2.rectangle(overlay,(x,y),(x+w-1,y+h-1),(255,40,40),2)
    if report.get('detector')=='radial_rings' and report.get('status')=='accepted':
        center=tuple(round(x) for x in report['position_xy'])
        cv2.circle(overlay,center,round(report['analysis_radius_px']),(50,255,80),1)
    Image.fromarray(overlay).save(folder/'candidates.png')
    info['annotation_legend'] = {'green':'Accepted fitted center', 'orange':'Rank 1, NOT accepted',
                                 'red':'Other candidates or search ROI', 'rank':'Descending compact peak signal'}
    (folder/'diagnostics.json').write_text(json.dumps(info,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


GUIDE = """几何标定采集诊断

圆环模式：使用带正负号的背景差分、边缘方向与 180° 中心对称性定位，绿色圆表示分析范围。
diagnostics.json 包含对称性、角向覆盖、内外环中心差、重复采集质量。difference.png 仍仅展示正差分。
若启用五点移动预检，前五个 probe 是中心/左右/上下；之后九点拟合、五点独立验证。
movement_check.json 保存预检；exposure_preparation.json 保存自动曝光过程，曝光图也保留原始数组。

background/ 是平相位背景，probe_00/ 是第 1 个定位点，依次编号。
每个目录的 raw.npy 保留供算法使用的采集数组（可能是多帧平均）；raw.json 为采集元数据。
slm1.png / slm2.png 是发送给对应 SLM 的相位输入图。
raw_preview.png：按相同的 camera_white 缩放到 8 位灰度，便于比较背景和定位采集。
彩色输入使用绿色通道，与定位算法一致。PNG 不代替原始数值分析。

各 probe 目录还保存：
- difference.png：max(定位采集 - 背景, 0)，按本张差分最大值拉伸显示。
- difference_smoothed.png：检测使用的平滑差分，与 difference.png 共用显示范围。
- compact_signal.png：移除宽缓背景后的紧凑信号，按本张最大值显示；不用于光度训练。
- candidate_mask.png：紧凑信号检测阈值对应的掩膜，包含随后被面积条件排除的小区域。
- candidates.png：原始相机坐标下的候选框、拟合中心和峰值排名（最多标注 20 个）。
  绿色表示通过检查的定位点；橙色仅表示排名第一但未通过，红色表示其他候选。
- diagnostics.json：失败原因、阈值、最多 32 个强候选的拟合结果、半高全宽、外围残差、
  两次外围拟合的中心变化、饱和占比和候选拒绝原因。candidate_count 记录全部候选数。
  候选按去除宽缓背景后的峰值排序；total_signal 仅为该紧凑区域积分，不用于排名。
- repeatability.json：每个定位点所有成功重复采集的中心、最大两两距离及检查结果。
  probe_XX_repeat_01 等文件夹保存重复采集；每张通过定位不代表重复稳定性或整个标定通过。

当前歧义阈值为 0.45：第二个合格候选峰值大于第一个的 45% 时拒绝定位。
中心有限饱和时，使用未饱和外围与平面背景拟合；候选框内饱和超过 35%、外围不足、
残差过大、光斑过宽或位置不稳定仍拒绝。此规则只用于几何定位，不放宽回环过曝检查。
搜索 ROI 使用原始帧坐标，红色外框标出搜索范围；不修改硬件 ROI 或最终目标 ROI。
首次失败的定位点不会写入 points.json 的成功点列表；诊断文件独立保留。
"""
