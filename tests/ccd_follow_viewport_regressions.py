"""Live overview follows a viewport, never a frozen reference frame."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import unittest

import numpy as np
from PyQt5 import QtCore,QtGui,QtWidgets

from tools.gui.main_window import MainWindow
from tools.gui.configuration import apply_values,read_values,camera_qimage
from tools.gui.ccd_detail_view import CcdDetailView
from tools.gui.ccd_viewport_preview import DetailViewport
from tools.utils.image_transforms import ImageTransform


class FollowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.w=MainWindow(demo=True);self.w.resize(1440,1000);self.w.show()
        self.d=self.w.ccd_detail;self.app.processEvents()

    def tearDown(self):self.w.close();self.app.processEvents()

    def frame(self,value,number=1,shape=(800,1200)):
        raw=np.full(shape,value,np.uint8);meta={'frame_id':number,'image_transform':{}}
        self.w.camera_preview_packet=(raw,meta)
        self.d.receive(raw,meta)
        return raw

    def setup_detail(self):
        self.w.mainTabWidget.setCurrentWidget(self.w.ccdDetailPage)
        self.frame(40);self.d.view.set_zoom(3);self.d.view.centerOn(400,300)
        self.app.processEvents()

    def test_frozen_detail_stays_frozen_but_following_overview_is_live(self):
        self.setup_detail();saved=self.d.overview_viewport
        self.w.ccdDetailFreezeButton.setChecked(True);frozen=self.d.displayed
        self.w.mainTabWidget.setCurrentWidget(self.w.previewPage)
        self.w.ccdZoomComboBox.setCurrentIndex(1)
        self.assertIs(self.w.mainTabWidget.currentWidget(),self.w.previewPage)
        for number,value in ((2,90),(3,180)):
            self.frame(value,number);self.w.refresh_preview('ccd')
            image=self.w.images['ccd']
            self.assertEqual(image.pixelColor(image.width()//2,image.height()//2).red(),value)
            self.assertIs(self.d.displayed,frozen)
        self.assertEqual(saved.transform,self.d.overview_viewport.transform)
        self.w.resize(1100,800);self.app.processEvents()
        self.assertEqual(saved.transform,self.d.overview_viewport.transform)

    def test_pan_updates_raw_region_and_geometry_change_falls_back(self):
        self.setup_detail()
        before=self.d.overview_viewport.transform
        self.d.view.centerOn(800,500);self.app.processEvents()
        self.assertNotEqual(before,self.d.overview_viewport.transform)
        self.w.mainTabWidget.setCurrentWidget(self.w.previewPage)
        self.w.ccdZoomComboBox.setCurrentIndex(1)
        self.frame(80,2,(600,800));self.w.refresh_preview('ccd')
        self.assertIn('暂显示全图',self.w.ccdPreviewLabel.toolTip())
        self.assertEqual(self.w.images['ccd'].size(),QtCore.QSize(800,600))
        self.d.clear();self.assertIsNone(self.d.overview_viewport)

    def test_old_config_migrates_without_jumping_or_resetting_view(self):
        self.setup_detail();zoom=self.d.view.zoom
        self.w.mainTabWidget.setCurrentWidget(self.w.previewPage)
        apply_values(self.w,{'ccdZoomComboBox':'1:1 像素','ccdDetailAutoOpenCheckBox':True})
        self.assertEqual(self.w.ccdZoomComboBox.currentText(),'跟随第三页视野')
        self.assertIs(self.w.mainTabWidget.currentWidget(),self.w.previewPage)
        self.assertEqual(self.d.view.zoom,zoom)
        self.assertNotIn('ccdDetailAutoOpenCheckBox',read_values(self.w))

    def test_rotation_flip_and_crop_match_full_raw_render(self):
        yy,xx=np.indices((700,900));raw=np.stack((xx%256,yy%256,(xx//8+yy//8)%256),axis=-1).astype(np.uint8)
        v=CcdDetailView();v.resize(420,320);v.show()
        try:
            for angle in (0,45,90,180):
                v.set_image(camera_qimage(raw),ImageTransform(angle=angle,horizontal=True,vertical=angle==45))
                v.set_zoom(2);v.centerOn(410,310);self.app.processEvents()
                packet=(raw,{'frame_id':1},None);view=DetailViewport.capture(v,packet)
                actual=view.render(packet,v.viewport().size())
                expected=QtGui.QImage(actual.size(),QtGui.QImage.Format_RGB32);expected.fill(QtGui.QColor('#101b2b'))
                painter=QtGui.QPainter(expected);painter.setTransform(view.transform)
                painter.drawImage(0,0,camera_qimage(raw));painter.end()
                for x,y in ((100,100),(210,160),(300,210)):
                    a,b=actual.pixelColor(x,y),expected.pixelColor(x,y)
                    self.assertEqual(a,b,(angle,x,y,a.getRgb(),b.getRgb()))
        finally:v.close()

    def test_high_bit_depth_crop_keeps_full_frame_brightness(self):
        v=CcdDetailView();v.resize(420,320);v.show()
        try:
            raw=np.full((700,900),1000,np.uint16);raw[0,0]=4000
            v.set_image(camera_qimage(raw),ImageTransform());v.set_zoom(2);v.centerOn(400,300)
            self.app.processEvents();packet=(raw,{},None)
            actual=DetailViewport.capture(v,packet).render(packet,QtCore.QSize(300,200))
            self.assertEqual(actual.pixelColor(actual.width()//2,actual.height()//2).red(),63)
        finally:v.close()


if __name__=='__main__':unittest.main()
