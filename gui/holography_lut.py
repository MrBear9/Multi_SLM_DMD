"""Independent LUT controls; stays usable without a training Python runtime."""
import copy
import json
from pathlib import Path
import numpy as np
from PIL import Image
from PyQt5 import QtCore,QtGui,QtWidgets,uic
from ..holography.lut import validate_profile,phase_to_gray
from ..utils.paths import TOOLS_ROOT


class HolographyLutPanel(QtCore.QObject):
    def __init__(self,ui):
        super().__init__(ui)
        self.ui=ui;self.pending=None
        self.page=uic.loadUi(str(TOOLS_ROOT/'ui'/'HolographyLut.ui'))
        self.page.layout().setAlignment(QtCore.Qt.AlignTop)
        for label in self.page.findChildren(QtWidgets.QLabel):
            label.setSizePolicy(QtWidgets.QSizePolicy.Preferred,QtWidgets.QSizePolicy.Maximum)
        ui.settingsTabs.insertTab(ui.settingsTabs.indexOf(ui.feedbackPage),self.page,'LUT 标定')
        for name in ('slm1_lut_source','slm2_lut_source','lut_split','lut_negative',
                     'lut_gray_step','lut_carrier_x','lut_carrier_y'):
            setattr(ui,name,getattr(self.page,name))
        for name,action in (
            ('lutReadRoiButton',self.read_roi),('lutLoadButton',self.load),
            ('lutApplyButton',self.apply),('lutSaveButton',self.save),
            ('lutPatternButton',self.save_pattern),('lutConvertButton',self.convert),
            ('lutStartButton',lambda:ui.controller.start('lut'))):
            getattr(self.page,name).clicked.connect(lambda _,action=action:ui.guarded(action))
        self.page.lutClearRoiButton.clicked.connect(self.clear_roi)
        ui.mode.currentIndexChanged.connect(self.refresh)
        ui.wavelength_nm.valueChanged.connect(self.refresh)

    def read_roi(self):
        detail=self.ui.controller.window.ccd_detail
        if detail.displayed is None or detail.view.roi is None:
            raise ValueError('请先在第三页框选多条清晰直线干涉条纹；不要框普通圆环或整个亮面。')
        r=detail.view.roi
        document=detail._document(dict(name='LUT 干涉条纹',x=r.x(),y=r.y(),width=r.width(),height=r.height()))
        document=json.loads(json.dumps(document,ensure_ascii=False))
        from ..holography.spot_detection import search_rect
        _,_,w,h=search_rect(document,detail.displayed[0].shape,self.ui.controller.hardware_signature())
        if min(w,h)<32:raise ValueError('LUT 条纹 ROI 至少需要 32×32 原始像素。')
        self.ui.settings.lut_roi=document
        self.refresh()

    def clear_roi(self):
        self.ui.settings.lut_roi={};self.refresh()

    def load_path(self,path):
        profile=json.loads(Path(path).read_text(encoding='utf-8'))
        validate_profile(profile)
        self.pending=profile
        self.ui.log.appendPlainText('LUT 待应用：'+str(path)+'；请核对 SLM、波长和相位方向。')
        self.refresh()

    def load(self):
        path,_=QtWidgets.QFileDialog.getOpenFileName(self.ui,'导入独立实测 LUT','','LUT (lut.json);;JSON (*.json)')
        if path:self.load_path(path)

    def apply(self):
        if self.pending is None:raise ValueError('请先完成扫描或导入实测 LUT。')
        key=self.pending.get('slm')
        if key not in ('slm1','slm2'):raise ValueError('未知 SLM。')
        validate_profile(self.pending,self.ui.values(),key)
        self.ui.settings.lut_profiles[key]=copy.deepcopy(self.pending)
        self.pending=None
        getattr(self.ui,key+'_lut_source').setCurrentIndex(2)
        self.ui.log.appendPlainText(f'已应用 {key} 实测 LUT；后续训练及导出采用此曲线。已生成的旧相位不会自动重算，旧模型需匹配 LUT。')
        self.refresh()

    def selected(self):
        if self.pending is not None:return self.pending
        if not hasattr(self.ui,'settings'):return None
        key=('slm1','slm2','slm1')[self.ui.mode.currentIndex()]
        return self.ui.settings.lut_profiles.get(key)

    def save(self):
        profile=self.selected()
        if profile is None:raise ValueError('暂无实测 LUT。')
        path,_=QtWidgets.QFileDialog.getSaveFileName(self.ui,'保存独立 LUT','lut.json','JSON (*.json)')
        if path:Path(path).write_text(json.dumps(profile,ensure_ascii=False,indent=2),encoding='utf-8')

    def save_pattern(self):
        from ..holography.lut_worker import split_pattern
        config=self.ui.values()
        if config.mode=='dual':raise ValueError('请选择单片 SLM。')
        path,_=QtWidgets.QFileDialog.getSaveFileName(self.ui,'保存分区测试图（参考 0 / 测试 128）','lut_split_128.png','PNG (*.png)')
        if path:
            Image.fromarray(split_pattern(config,128)).save(path)
            self.ui.log.appendPlainText('可在主 UI 加载分区测试图，用于调整干涉光路。它不会自动产生适合测量的条纹。')

    def convert(self):
        config=self.ui.values();key=config.mode
        if key=='dual' or getattr(config,key+'_lut_source')!='measured':
            raise ValueError('请选择单片 SLM 并应用实测 LUT，再转换外部理想相位图。')
        validate_profile(config.lut_profiles.get(key),config,key)
        path,_=QtWidgets.QFileDialog.getOpenFileName(self.ui,'选择外部理想线性相位图（不能是已经校正的导出图）','','PNG (*.png);;BMP (*.bmp)')
        if not path:return
        with Image.open(path) as image:
            if image.mode!='L':raise ValueError('只支持 8 位单通道理想相位图。')
            values=np.asarray(image,dtype=float)/255
        # Ideal images encode an absolute phase; a global piston is immaterial.
        phase=(1-values if config.gray_inverted else values)*2*np.pi
        converted=phase_to_gray(phase,config,key)
        output,_=QtWidgets.QFileDialog.getSaveFileName(self.ui,'另存实测 LUT 灰度图',str(Path(path).with_name(Path(path).stem+'_lut.png')),'PNG (*.png)')
        if output:
            if Path(output).resolve()==Path(path).resolve():raise ValueError('请另存文件，保留原始理想相位。')
            Image.fromarray(converted).save(output)
            Path(output).with_suffix('.lut.json').write_text(json.dumps({'phase_encoding':'native_gray_8bit_already_lut_aware_do_not_remap',
                'source':str(path),'lut':config.lut_profiles[key]},ensure_ascii=False,indent=2),encoding='utf-8')
            self.ui.log.appendPlainText('已转换，主 UI 请直接加载此灰度图，不要再次校正：'+output)

    def refresh(self,*_):
        if not hasattr(self.ui,'settings'):return
        settings=self.ui.settings
        roi=settings.lut_roi.get('roi')
        self.page.lutRoiLabel.setText('条纹 ROI：未选择' if not roi else
            f"条纹 ROI：({roi['x']}, {roi['y']})，{roi['width']}×{roi['height']} px")
        lines=[]
        for key in ('slm1','slm2'):
            profile=settings.lut_profiles.get(key)
            text='无实测曲线'
            if profile:
                try:
                    curve=validate_profile(profile)
                    text=f"已存 {profile['wavelength_nm']:g} nm / 范围 {abs(curve[-1]):.3f} rad"
                except ValueError as exc:text=str(exc)
            lines.append(key.upper()+'：'+getattr(self.ui,key+'_lut_source').currentText()+'；'+text)
        self.page.lutStatusLabel.setText('\n'.join(lines))
        self.page.lutStartButton.setEnabled(self.ui.mode.currentIndex()!=2)
        self.page.lutApplyButton.setEnabled(self.pending is not None)
        profile=self.selected()
        self.page.lutSaveButton.setEnabled(profile is not None)
        self.draw(profile)

    def draw(self,profile):
        pix=QtGui.QPixmap(780,150);pix.fill(QtGui.QColor('#ffffff'))
        painter=QtGui.QPainter(pix);painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(QtGui.QColor('#40566c'))
        if profile is None:
            painter.drawText(pix.rect(),QtCore.Qt.AlignCenter,'未加载实测 LUT；默认无模型时使用理想线性曲线')
        else:
            try:
                curve=validate_profile(profile);values=np.abs(curve);span=max(values[-1],2*np.pi)
                prefix='待应用：' if self.pending is not None else '已保存：'
                painter.drawText(12,18,prefix+f"{profile['slm']} / {profile['wavelength_nm']:g} nm：|相位| vs 灰度；范围 {values[-1]:.3f} rad；方向 {'递减' if curve[-1]<0 else '递增'}")
                painter.drawLine(40,125,760,125);painter.drawLine(40,30,40,125)
                painter.setPen(QtGui.QPen(QtGui.QColor('#9cabb8'),1,QtCore.Qt.DashLine))
                painter.drawLine(QtCore.QPointF(40,125),QtCore.QPointF(750,125-90*2*np.pi/span))
                painter.setPen(QtGui.QPen(QtGui.QColor('#1269d3'),2))
                polygon=QtGui.QPolygonF([QtCore.QPointF(40+710*g/255,125-90*v/span) for g,v in enumerate(values)])
                painter.drawPolyline(polygon)
                painter.drawText(40,145,'0');painter.drawText(650,145,'255（虚线：理想）')
            except ValueError as exc:painter.drawText(pix.rect(),QtCore.Qt.AlignCenter,str(exc))
        painter.end()
        self.page.lutCurveLabel.setPixmap(pix)
        self.page.lutCurveLabel.setScaledContents(True)
