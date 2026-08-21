import sys
import os
import cv2
from DetectorGUI import WatermarkDetectorApp

def test():
    import faulthandler
    faulthandler.enable()
    app = None
    try:
        from PyQt5.QtWidgets import QApplication
        app = QApplication(sys.argv)
        ex = WatermarkDetectorApp()

        print("Testing Image with YOLOv8:")
        ex.drop_zone.file_path = "test_img.jpg"
        ex.model_selector.setCurrentText("YOLOv8")
        ex.scan_file()
        print("Result:", ex.result_label.text(), "| Confidence:", ex.conf_bar.value(), "| Loc:", ex.loc_value.text(), "| Frames:", ex.frames_value.text())

        print("\nTesting Video with YOLOv8:")
        ex.drop_zone.file_path = "test_vid.mp4"
        ex.model_selector.setCurrentText("YOLOv8")
        ex.scan_file()
        print("Result:", ex.result_label.text(), "| Confidence:", ex.conf_bar.value(), "| Loc:", ex.loc_value.text(), "| Frames:", ex.frames_value.text())
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == '__main__':
    test()
