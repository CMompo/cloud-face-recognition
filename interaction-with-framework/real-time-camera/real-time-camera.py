import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx_tf.backend import prepare
import requests
import json
import argparse
import grequests

def parse_args():
    parser = argparse.ArgumentParser(prog="Real-Time Face Recognition - Multithreaded",
    description="This program recognices the faces from the on-board webcam by using the Cloud API face recognition functions.")
    parser.add_argument('--ip-address', required=True,
                        help='IP address of the Cloud API server')
    parser.add_argument('--min-size', required=False, type=int, default=50,
                        help='Minimum face image size on the long axis. Smaller images are discarded.')
    parser.add_argument('--max-size', required=False, type=int, default=150,
                        help='Maximum face image size on the long axis. Larger images are resized.')
    parser.add_argument('--min-confidence', required=False, type=float, default=0.7,
                        help='Minimum confidence level to show result of face recognition.')
    parser.add_argument('--multithreading', action='store_true', required=False,
                        help='Send all face images in a frame simultasneously to the Cloud API (improves performance).')
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
API_BASE_URL = 'http://'+str(args.ip_address)
send_url = API_BASE_URL + '/face-recognition/get-results'
# Preparing Video feed
video_handler = cv2.VideoCapture(0)

onnx_path = 'ultra_light_640.onnx'
onnx_model = onnx.load(onnx_path)
predictor = prepare(onnx_model)
ort_session = ort.InferenceSession(onnx_path)
input_name = ort_session.get_inputs()[0].name

while True:
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
        
        if(args.multithreading):
            face_images = []
        else:
            face_data = []
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
                        
                _, face_image_enc = cv2.imencode(".jpg",face_image)
                if(args.multithreading):
                    face_images.append(face_image_enc)
                else:
                    myobj = {'image': face_image_enc}
                    try:
                        request = requests.post(send_url, files = myobj, timeout=10)
                        #print(request.text)
                        face_data.append(request.text)
                        print(request.text)
                    except requests.exceptions.ReadTimeout:
                        pass
        if(args.multithreading):
            prep_req = (grequests.post(send_url, files = {'image': face_img_send}, timeout=10) for face_img_send in face_images)
            result = grequests.map(prep_req, exception_handler)
        
        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (80,18,236), 2)
            #cv2.rectangle(frame, (x1, y2 - 20), (x2, y2), (80,18,236), cv2.FILLED)
            try:    
                if(args.multithreading):
                    det_data = JSONObj(result[i].text)
                else:
                    det_data = JSONObj(face_data[i])
                text = f"{det_data.faces[0]['top_prediction']['label']}"
                if(det_data.faces[0]['top_prediction']['confidence']<args.min_confidence):
                    # Not Recognized
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.rectangle(frame, (x1, y2 - 20), (x2, y2), (0,255,0), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, text, (x1 + 6, y2 - 6), font, 0.33, (0,0,0), 1)
            except:
                # Not a face
                pass

        cv2.imshow('Video', frame)
        # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_handler.release()
cv2.destroyAllWindows()