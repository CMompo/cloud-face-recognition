import cv2
import face_detection
print(face_detection.available_detectors)
detector = face_detection.build_detector(
  "RetinaNetMobileNetV1", confidence_threshold=.5, nms_iou_threshold=.3)
# BGR to RGB conversion due to opencv color ordering
im = cv2.imread("test.jpg")[:, :, ::-1]

detections = detector.detect(im)
print(len(detections))