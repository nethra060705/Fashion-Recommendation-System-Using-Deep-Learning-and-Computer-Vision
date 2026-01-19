import cv2
import numpy as np
import os
import yaml
from yaml.loader import SafeLoader

class ObjDetection():
    def __init__(self, onnx_model, data_yaml):
        # load data yaml
        with open(data_yaml, mode='r') as f:
            data_yaml = yaml.load(f, Loader=SafeLoader)

        self.labels = data_yaml['names']
        self.nc = data_yaml['nc']

        # load object detection model
        self.yolo = cv2.dnn.readNetFromONNX(onnx_model)
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def crop_objects(self, image, conf_threshold=0.25, nms_threshold=0.4):
        """
        Improved object detection with better confidence thresholds and error handling
        """
        try:
            if len(image.shape) != 3 or image.shape[2] != 3:
                raise ValueError("Input image must be a 3-channel color image")
                
            row, col, d = image.shape

            # convert img into square array
            max_rc = max(row, col)
            input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
            input_image[0:row, 0:col] = image

            # get prediction from square array
            INPUT_WH_YOLO = 640
            blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
            self.yolo.setInput(blob)
            preds = self.yolo.forward() # prediction from model

            # NMS - improved confidence thresholds
            detections = preds[0]
            boxes = []
            confidences = []
            classes = []

            # width and height of the image (input_image)
            image_h, image_w = input_image.shape[:2]
            x_factor = image_w / INPUT_WH_YOLO
            y_factor = image_h / INPUT_WH_YOLO

            for i in range(len(detections)):
                detection = detections[i]
                confidence = detection[4] # confidence of obj detection
                
                if confidence > conf_threshold:
                    class_scores = detection[5:]
                    class_score = np.max(class_scores)
                    class_id = np.argmax(class_scores)

                    if class_score > conf_threshold:
                        cx, cy, w, h = detection[0:4]
                        
                        # construct bbox from 4 values
                        left = int((cx - 0.5 * w) * x_factor)
                        top = int((cy - 0.5 * h) * y_factor)
                        width = int(w * x_factor)
                        height = int(h * y_factor)
                        
                        # Ensure bounding box is within image bounds
                        left = max(0, left)
                        top = max(0, top)
                        width = min(width, image_w - left)
                        height = min(height, image_h - top)
                        
                        # Filter out very small detections
                        if width > 20 and height > 20:
                            box = np.array([left, top, width, height])
                            confidences.append(float(confidence))
                            boxes.append(box)
                            classes.append(class_id)

            if not boxes:
                return []

            # Apply NMS with better threshold
            boxes_np = np.array(boxes).tolist()
            confidences_np = np.array(confidences).tolist()
            
            indices = cv2.dnn.NMSBoxes(boxes_np, confidences_np, conf_threshold, nms_threshold)
            
            if len(indices) == 0:
                return []
                
            # Flatten indices array if needed
            if isinstance(indices, np.ndarray) and len(indices.shape) > 1:
                indices = indices.flatten()

            # Crop objects from original image (not padded image)
            cropped_objects = []
            for ind in indices:
                x, y, w, h = boxes_np[ind]
                
                # Map coordinates back to original image
                orig_x1 = max(0, min(x, col))
                orig_y1 = max(0, min(y, row))
                orig_x2 = max(0, min(x + w, col))
                orig_y2 = max(0, min(y + h, row))

                # Ensure we have a valid crop region
                if orig_x2 > orig_x1 and orig_y2 > orig_y1:
                    cropped_obj = image[orig_y1:orig_y2, orig_x1:orig_x2].copy()
                    if cropped_obj.size > 0:  # Ensure the cropped object is not empty
                        cropped_objects.append(cropped_obj)

            return cropped_objects
            
        except Exception as e:
            print(f"Error in object detection: {e}")
            return []
