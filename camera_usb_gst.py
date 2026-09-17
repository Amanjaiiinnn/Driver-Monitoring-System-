import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

import numpy as np
import cv2

Gst.init(None)

class Camera:
    def __init__(self, device="/dev/video7", width=640, height=480, fps=30):
        self.pipeline = Gst.Pipeline()

        # Source
        self.source = Gst.ElementFactory.make("v4l2src", "source")
        self.source.set_property("device", device)

        # Caps (IMPORTANT)
        caps_str = f"video/x-raw,width={width},height={height},framerate={fps}/1"
        self.caps = Gst.ElementFactory.make("capsfilter", "caps")
        self.caps.set_property("caps", Gst.Caps.from_string(caps_str))

        # Convert
        self.convert = Gst.ElementFactory.make("videoconvert", "convert")

        # Appsink
        self.appsink = Gst.ElementFactory.make("appsink", "sink")
        self.appsink.set_property("emit-signals", True)
        self.appsink.set_property("sync", False)
        self.appsink.set_property("max-buffers", 1)
        self.appsink.set_property("drop", True)

        # Force RGB for numpy
        self.appsink.set_property(
            "caps",
            Gst.Caps.from_string(f"video/x-raw,format=RGB,width={width},height={height}")
        )

        self.appsink.connect("new-sample", self.on_frame)

        # Add + link
        self.pipeline.add(self.source)
        self.pipeline.add(self.caps)
        self.pipeline.add(self.convert)
        self.pipeline.add(self.appsink)

        self.source.link(self.caps)
        self.caps.link(self.convert)
        self.convert.link(self.appsink)

    def on_frame(self, sink):
        sample = sink.emit("pull-sample")
        buf = sample.get_buffer()
        caps = sample.get_caps()

        width = caps.get_structure(0).get_value('width')
        height = caps.get_structure(0).get_value('height')

        # Convert to numpy
        frame = np.ndarray(
            (height, width, 3),
            buffer=buf.extract_dup(0, buf.get_size()),
            dtype=np.uint8
        )

        # FIX COLOR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Show (for testing)
        cv2.imshow("Camera", frame)
        cv2.waitKey(1)

        return Gst.FlowReturn.OK

    def start(self):
        self.pipeline.set_state(Gst.State.PLAYING)

    def stop(self):
        self.pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    cam = Camera(device="/dev/video7", width=640, height=480, fps=30)
    cam.start()

    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        cam.stop()
        cv2.destroyAllWindows()
