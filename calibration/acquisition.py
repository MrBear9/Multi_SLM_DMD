"""Thread-safe raw-camera batches, isolated from preview refresh and transforms."""
from dataclasses import dataclass, field
import threading

import numpy as np


@dataclass
class FrameBatch:
    after: float
    deadline: float
    count: int
    discard: int
    event: threading.Event = field(default_factory=threading.Event)
    frames: list = field(default_factory=list)
    metadata: list = field(default_factory=list)
    error: str = ''
    last_id: object = None
    cancelled: bool = False

    def fail(self, message):
        self.error = message
        self.event.set()

    def offer(self, image, metadata, read_started, now):
        if self.event.is_set():
            return
        if self.cancelled:
            self.fail('标定采集已取消。')
            return
        if now > self.deadline:
            self.fail('标定等待新相机帧超时。')
            return
        if read_started < self.after:
            return
        frame_id = metadata.get('frame_id')
        if frame_id is not None and frame_id == self.last_id:
            return
        self.last_id = frame_id
        if self.discard:
            self.discard -= 1
            return
        if self.frames and image.shape != self.frames[0].shape:
            self.fail('标定过程中相机图像尺寸发生变化。')
            return
        self.frames.append(np.array(image, copy=True))
        self.metadata.append(dict(metadata))
        if len(self.frames) >= self.count:
            self.event.set()

    def result(self):
        if self.error:
            raise RuntimeError(self.error)
        if not self.event.is_set() or len(self.frames) != self.count:
            raise RuntimeError('相机批次尚未采集完成。')
        dtype = self.frames[0].dtype
        return np.rint(np.mean(self.frames, axis=0)).astype(dtype), self.metadata

