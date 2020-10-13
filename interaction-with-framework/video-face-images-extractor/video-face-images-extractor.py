import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx_tf.backend import prepare
import json
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(prog="Video Face Image Extraction",
    description="This program detects and extracts the faces in a video stream.")
    parser.add_argument('--video-source', required=True,
                        help='File path, web source of device source of the video stream.')
    parser.add_argument('--output-folder', required=True,
                        help='Output folder where the face images extracted are stored.')
    parser.add_argument('--break-at-eof', required=False, action='store_true',
                        help='In case the input file is finite (has an end), break once no frames are available. Do not use for real-time!')
    parser.add_argument('--skip-frames', required=False, type=int,
                        help='Number of frames to be skiped. For 30fps, if set to 5, the effective framerate is 5fps.')
    parser.add_argument('--target-fps', required=False, type=float,
                        help='Approximate target frames per second. Only for video files!')
    parser.add_argument('--min-size', required=False, type=int, default=50,
                        help='Minimum face image size on the long axis. Smaller images are discarded.')
    parser.add_argument('--max-size', required=False, type=int, default=150,
                        help='Maximum face image size on the long axis. Larger images are resized.')
    return parser.parse_args()
    
def area_of(left_top, right_bottom):
    """
    Compute the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]

def iou_of(boxes0, boxes1, eps=1e-5):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)

def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Perform hard non-maximum-supression to filter out boxes with iou greater
    than threshold
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
        picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]

def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.5, top_k=-1):
    """
    Select boxes that contain human faces
    Args:
        width: original image width
        height: original image height
        confidences (N, 2): confidence array
        boxes (N, 4): boxes array in corner-form
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
    Returns:
        boxes (k, 4): an array of boxes kept
        labels (k): an array of labels for each boxes kept
        probs (k): an array of probabilities for each boxes being in corresponding labels
    """
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(box_probs,
           iou_threshold=iou_threshold,
           top_k=top_k,
           )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

class JSONObj(object):
    def __init__(self, data):
	    self.__dict__ = json.loads(data)

def exception_handler(request, exception):
    print("Request failed.")

args = parse_args()
OUTPUT_FOLDER = str(args.output_folder)
if(not os.path.isdir(OUTPUT_FOLDER)):
    os.mkdir(OUTPUT_FOLDER)
# Preparing Video feed
video_handler = cv2.VideoCapture(args.video_source)

onnx_path = 'ultra_light_640.onnx'
onnx_model = onnx.load(onnx_path)
predictor = prepare(onnx_model)
ort_session = ort.InferenceSession(onnx_path)
input_name = ort_session.get_inputs()[0].name
frame_count = 0

# Frame skipping
loop=True
SKIP_FRAMES = 0
if(args.skip_frames!=None):
    SKIP_FRAMES = args.skip_frames
if(args.target_fps!=None):
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver)  < 3 :
        FPS = video_handler.get(cv2.cv.CV_CAP_PROP_FPS)
        if(video_handler.get(cv2.cv.CV_CAP_PROP_FPS)<args.target_fps):
            print('Warning: target-fps is higher than original FPS. Setting to {}.'.format(video_handler.get(cv2.cv.CV_CAP_PROP_FPS)))
            SKIP_FRAMES = 0
        else:
            SKIP_FRAMES = round(video_handler.get(cv2.cv.CV_CAP_PROP_FPS)/args.target_fps)-1
    else:
        FPS = video_handler.get(cv2.CAP_PROP_FPS)
        if(video_handler.get(cv2.CAP_PROP_FPS)<args.target_fps):
            print('Warning: target-fps is higher than original FPS. Setting to {}.'.format(video_handler.get(cv2.CAP_PROP_FPS)))
            SKIP_FRAMES = 0
        else:
            SKIP_FRAMES = round(video_handler.get(cv2.CAP_PROP_FPS)/args.target_fps)-1
            
assert SKIP_FRAMES >= 0
            
if(args.skip_frames!=None and args.target_fps!=None):
    print('Error: skip-frames and target-fps cannot be set simultaneously.')
    loop = False
if(args.max_size<=0):
    print ('Error: max-size must be set to a value of one or higher.')
    loop = False
print("1 out of {} frame(s) will be processed. Source FPS: {}. Target FPS: {}. Processed FPS: {}.".format(SKIP_FRAMES, FPS, args.target_fps, FPS/(SKIP_FRAMES+1)))
while loop: 
    # Frame skipping
    for _ in range(SKIP_FRAMES):
        _ = video_handler.grab()
    ret, frame = video_handler.read()
    if frame is not None:
        h, w, _ = frame.shape

        # preprocess img acquired
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert bgr to rgb
        img = cv2.resize(img, (640, 480)) # resize
        img_mean = np.array([127, 127, 127])
        img = (img - img_mean) / 128
        img = np.transpose(img, [2, 0, 1])
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)

        confidences, boxes = ort_session.run(None, {input_name: img})
        boxes, labels, probs = predict(w, h, confidences, boxes, 0.7)
        
        image_count = 0
        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            x1, y1, x2, y2 = box
            # Margins for the face box
            y_m = int((y2-y1)*0.25)
            x_m = int((x2-x1)*0.25)
            if(y1-y_m>=0 and y2+y_m<frame.shape[0] and x1-x_m>=0 and x2+x_m<frame.shape[1]):
                face_image = frame[y1-y_m:y2+y_m, x1-x_m:x2+x_m]
                if(face_image.size!=0):
                    cv2.imshow('Detected face', face_image)
                #print('y_m:{} x_m:{}'.format(y_m,x_m))
                #print('{}-{}-{}-{}'.format(y1+y_m,y2-y_m,x1+x_m,x2-x_m))
            else:
                if(y1>=0 and y2<frame.shape[0] and x1>=0 and x2<frame.shape[1]):
                    face_image = frame[y1:y2, x1:x2]
                    cv2.imshow('Detected face', face_image)
                    print("Extended size face image out-of-bounds. Using reduced version.")
                else:
                    print("The face is out of range.")
                
            if(face_image.shape[0]>args.min_size and face_image.shape[1]>args.min_size):
                file_name = "Image {}-{}.jpg".format(frame_count, image_count)
                file_path = os.path.join(OUTPUT_FOLDER, file_name)
                # Resize image
                height = face_image.shape[0]
                width = face_image.shape[1]
                if(height>=width):
                    if(height>args.max_size):
                        rescale_factor = height/args.max_size
                        face_image = cv2.resize(face_image, (round(width/rescale_factor), args.max_size))
                else:
                    if(width>args.max_size):
                        rescale_factor = width/args.max_size
                        face_image = cv2.resize(face_image, (args.max_size, round(height/rescale_factor)))
                        
                cv2.imwrite(file_path, face_image)
                image_count = image_count+1
        
        
        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,255), 2)

        cv2.imshow('Video', frame)
        frame_count = frame_count + 1
        # Hit 'q' on the keyboard to quit!
    else:
        if(args.break_at_eof):
            break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_handler.release()
cv2.destroyAllWindows()