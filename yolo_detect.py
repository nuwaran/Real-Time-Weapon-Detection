import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO

# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")',
                    required=True)
parser.add_argument('--source', help='Image source, can be image file ("test.jpg"), \
                    image folder ("test_dir"), video file ("testvid.mp4"), index of USB camera ("0" or "usb0"), or index of Picamera ("picamera0")', 
                    required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
                    default=0.5)
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480"), \
                    otherwise, match source resolution',
                    default=None)
parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.',
                    action='store_true')

args = parser.parse_args()

# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record

# Check if model file exists and is valid
if not os.path.exists(model_path):
    print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
    sys.exit(0)

# Load the model into memory and get label map
model = YOLO(model_path, task='detect')
labels = model.names

# Validate and parse source
def validate_source(source):
    print(f"Validating source: {source}")
    try:
        # Try as webcam index (plain number like "0" or integer 0)
        index = int(source)
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            print(f"Cannot open webcam at index {index}")
            return None, None
        cap.release()
        print(f"Webcam at index {index} is valid")
        return 'usb', index
    except ValueError:
        # Try as file or folder
        if os.path.isdir(source):
            print(f"Source {source} is a valid folder")
            return 'folder', source
        elif os.path.isfile(source):
            _, ext = os.path.splitext(source)
            img_ext_list = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP']
            vid_ext_list = ['.avi', '.mov', '.mp4', '.mkv', '.wmv']
            if ext in img_ext_list:
                print(f"Source {source} is a valid image")
                return 'image', source
            elif ext in vid_ext_list:
                print(f"Source {source} is a valid video")
                return 'video', source
            else:
                print(f'File extension {ext} is not supported.')
                return None, None
        # Try USB or PiCamera format (e.g., "usb0", "picamera0")
        elif 'usb' in source.lower():
            usb_idx = int(source[3:])
            cap = cv2.VideoCapture(usb_idx)
            if not cap.isOpened():
                print(f"Cannot open webcam at index {usb_idx}")
                return None, None
            cap.release()
            print(f"Webcam at index {usb_idx} is valid")
            return 'usb', usb_idx
        elif 'picamera' in source.lower():
            picam_idx = int(source[8:])
            # Note: PiCamera validation would require Picamera2; for simplicity, assume valid if format matches
            print(f"PiCamera at index {picam_idx} assumed valid (not tested)")
            return 'picamera', picam_idx
        else:
            print(f'Input {source} is invalid. Please try again.')
            return None, None

source_type, source_value = validate_source(img_source)
if source_type is None:
    sys.exit(0)

# Parse user-specified display resolution
resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

# Check if recording is valid and set up recording
if record:
    if source_type not in ['video', 'usb']:
        print('Recording only works for video and camera sources. Please try again.')
        sys.exit(0)
    if not user_res:
        print('Please specify resolution to record video at.')
        sys.exit(0)
    
    # Set up recording
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW, resH))

# Load or initialize image source
if source_type == 'image':
    imgs_list = [source_value]
elif source_type == 'folder':
    imgs_list = []
    filelist = glob.glob(source_value + '/*')
    for file in filelist:
        _, file_ext = os.path.splitext(file)
        if file_ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP']:
            imgs_list.append(file)
elif source_type == 'video' or source_type == 'usb':
    if source_type == 'video':
        cap_arg = source_value
    elif source_type == 'usb':
        cap_arg = source_value
    cap = cv2.VideoCapture(cap_arg)
    if user_res:
        ret = cap.set(3, resW)
        ret = cap.set(4, resH)
elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (resW, resH)}))
    cap.start()

# Set bounding box colors (using the Tableau 10 color scheme)
bbox_colors = [(164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133), (88, 159, 106),
               (96, 202, 231), (159, 124, 168), (169, 162, 241), (98, 118, 150), (172, 176, 184)]

# Initialize control and status variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# Begin inference loop
while True:
    t_start = time.perf_counter()

    # Load frame from image source
    if source_type == 'image' or source_type == 'folder':
        if img_count >= len(imgs_list):
            print('All images have been processed. Exiting program.')
            sys.exit(0)
        img_filename = imgs_list[img_count]
        frame = cv2.imread(img_filename)
        img_count += 1
    elif source_type == 'video' or source_type == 'usb':
        ret, frame = cap.read()
        if not ret:
            print('Unable to read frames from the source. Exiting program.')
            break
    elif source_type == 'picamera':
        frame = cap.capture_array()
        if frame is None:
            print('Unable to read frames from the PiCamera. Exiting program.')
            break

    # Resize frame to desired display resolution
    if resize:
        frame = cv2.resize(frame, (resW, resH))

    # Run inference on frame
    results = model(frame, verbose=False)

    # Extract results
    detections = results[0].boxes

    # Initialize variable for basic object counting example
    object_count = 0

    # Go through each detection and get bbox coords, confidence, and class
    for i in range(len(detections)):
        xyxy_tensor = detections[i].xyxy.cpu()
        xyxy = xyxy_tensor.numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        conf = detections[i].conf.item()

        if conf > float(min_thresh):
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            object_count += 1

    # Calculate and draw framerate (if using video, USB, or PiCamera source)
    if source_type in ['video', 'usb', 'picamera']:
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Display detection results
    cv2.putText(frame, f'Number of objects: {object_count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow('YOLO detection results', frame)
    if record:
        recorder.write(frame)

    # Handle keypresses
    if source_type == 'image' or source_type == 'folder':
        key = cv2.waitKey(0)
    else:
        key = cv2.waitKey(5)
    
    if key == ord('q') or key == ord('Q'):
        break
    elif key == ord('s') or key == ord('S'):
        cv2.waitKey(0)
    elif key == ord('p') or key == ord('P'):
        cv2.imwrite('capture.png', frame)

    # Calculate FPS
    t_stop = time.perf_counter()
    frame_rate_calc = float(1 / (t_stop - t_start))

    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)
    frame_rate_buffer.append(frame_rate_calc)
    avg_frame_rate = np.mean(frame_rate_buffer)

# Clean up
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
if source_type in ['video', 'usb']:
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record:
    recorder.release()
cv2.destroyAllWindows()