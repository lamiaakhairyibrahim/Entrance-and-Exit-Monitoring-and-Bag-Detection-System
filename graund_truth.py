import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *
import json
import os

# --- Part 1: Setup for Detection and Tracking ---
model = YOLO('yolov8m.pt')
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

area1 = np.array( [(312, 388), (250, 397) ,(423, 477) , (497, 462) ], np.int32)
area2 = np.array( [(312, 388), (250, 397) ,(423, 477) , (497, 462) ], np.int32)

tracker = Tracker()

# This list will store your ground truth data (1 for bag, 0 for no bag)
ground_truth_bags = []
# This set will keep track of IDs that have already been annotated to avoid duplicates
annotated_ids = set()

# --- Part 2: Video Processing and Automated Annotation ---
video_path = r'..\data\video3.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

print("Instructions:")
print("The video will play automatically.")
print("It will pause when a person enters or exits a region.")
print("When it pauses, enter '1' for a bag or '0' for no bag.")
print("Press 'q' at any time to quit and save.")
print("-" * 50)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_copy = cv2.resize(frame, (1020, 500))
    results = model.predict(frame_copy, verbose=False)
    
    person_list = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]
            if conf > 0.4 and label == 'person':
                person_list.append([x1, y1, x2, y2])
           
    bbox_id = tracker.update(person_list)
    
    # Check for people crossing the regions
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        
        # Check if the person's ID has already been annotated
        if id in annotated_ids:
            continue
        
        # Check if person is in either region
        is_in_area1 = cv2.pointPolygonTest(area1, ((x4, y4)), False) >= 0
        is_in_area2 = cv2.pointPolygonTest(area2, ((x4, y4)), False) >= 0
        
        if is_in_area1 or is_in_area2:
            # Pause the video and prompt for input
            cv2.rectangle(frame_copy, (x3, y3), (x4, y4), (0, 255, 255), 2)
            cv2.putText(frame_copy, f"ID: {id}", (x3, y3-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imshow("Automated Ground Truth Creator", frame_copy)
            
            print(f"Person with ID {id} detected. Paused video.")
            try:
                has_bag_input = int(input("Enter 1 for bag, 0 for no bag: "))
                if has_bag_input in [0, 1]:
                    ground_truth_bags.append(has_bag_input)
                    annotated_ids.add(id)
                    print(f"Recorded for ID {id}: {has_bag_input}")
                else:
                    print("Invalid input. Please enter 1 or 0.")
            except ValueError:
                print("Invalid input. Please enter a number.")
            
            # Resume video playback after input
            cv2.waitKey(1)
            
    # Draw ROIs
    cv2.polylines(frame_copy, [area1], True, (0, 0, 255), 2)
    cv2.polylines(frame_copy, [area2], True, (0, 255, 0), 2)

    cv2.imshow("Automated Ground Truth Creator", frame_copy)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Save Ground Truth Data ---
output_dir = r'../data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_file = os.path.join(output_dir, 'ground_truth_bags_auto.json')
with open(output_file, 'w') as f:
    json.dump(ground_truth_bags, f, indent=4)

print("-" * 50)
print(f"Ground truth data saved to: {output_file}")
print("Now, run the main system code and compare the results.")

cap.release()
cv2.destroyAllWindows()