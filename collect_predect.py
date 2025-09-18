import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *
import json
import os

# Load YOLO model and class names
model = YOLO('yolov8m.pt')
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

# Define areas of interest (ROIs)
area1 = [(312, 388), (289, 390), (474, 469), (497, 462)]
area2 = [(279, 392), (250, 397), (423, 477), (454, 469)]

# Define the combined bag check region as a bounding box
all_points = area1 + area2
min_x = min(p[0] for p in all_points)
min_y = min(p[1] for p in all_points)
max_x = max(p[0] for p in all_points)
max_y = max(p[1] for p in all_points)
bag_check_region = [(312, 388), (250, 397) ,(423, 477) , (497, 462) ]
# Define goods to detect
goods_classes = [
    'backpack', 'handbag', 'suitcase', 'shopping bag', 'bottle', 'cup',
    'book', 'laptop', 'cell phone', 'keyboard'
]

# Initialize tracker and dictionaries
tracker = Tracker()
people_entering_track = {}
people_exiting_track = {}
entering = set()
exiting = set()
people_with_bags = set()
# A list to store the predictions
predicted_bags = []
# A set to keep track of IDs that have already been predicted to avoid duplicates
predicted_ids = set()


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# Function to get mouse coordinates (for debugging ROIs)
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Open video file
cap = cv2.VideoCapture(r'../data\video3.mp4')

# --- Video Writer setup ---
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
output_path = r'../data\output_video3_final_bag_region_lamiaa_predect.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
# --------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (1020, 500))
    
    results = model.predict(frame, verbose=False)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    
    person_list = []
    goods_detections = []
    
    for index, row in px.iterrows():
        x1, y1, x2, y2 = map(int, row[:4])
        d = int(row[5])
        conf = float(row[4])
        c = class_list[d]
        
        if conf > 0.4:
            if 'person' in c:
                person_list.append([x1, y1, x2, y2])
            elif c in goods_classes:
                goods_detections.append({'box': (x1, y1, x2, y2), 'label': c})
           
    bbox_id = tracker.update(person_list)
    
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        
        has_goods = False
        
        # Check for bags only within the combined region
        if min_x <= x4 <= max_x and min_y <= y4 <= max_y:
            for goods_box in goods_detections:
                gx1, gy1, gx2, gy2 = goods_box['box']
                if calculate_iou((x3, y3, x4, y4), (gx1, gy1, gx2, gy2)) > 0.1:
                    has_goods = True
                    people_with_bags.add(id)
                    break
        
        # Check if the person's ID has already been predicted
        if id in predicted_ids:
            continue
            
        # Entering logic
        results_enter = cv2.pointPolygonTest(np.array(area2, np.int32), ((x4, y4)), False)
        if results_enter >= 0:
            people_entering_track[id] = (x4, y4)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
            
        if id in people_entering_track:
            results_enter_final = cv2.pointPolygonTest(np.array(area1, np.int32), ((x4, y4)), False)
            if results_enter_final >= 0:
                entering.add(id)
                # Predict and record the bag status
                predicted_bags.append(1 if id in people_with_bags else 0)
                predicted_ids.add(id)
                if id in people_entering_track:
                    del people_entering_track[id]

        # Exiting logic
        results_exit = cv2.pointPolygonTest(np.array(area1, np.int32), ((x4, y4)), False)
        if results_exit >= 0:
            people_exiting_track[id] = (x4, y4)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
            
        if id in people_exiting_track:
            results_exit_final = cv2.pointPolygonTest(np.array(area2, np.int32), ((x4, y4)), False)
            if results_exit_final >= 0:
                exiting.add(id)
                # Predict and record the bag status
                predicted_bags.append(1 if id in people_with_bags else 0)
                predicted_ids.add(id)
                if id in people_exiting_track:
                    del people_exiting_track[id]
        
        label = "with bag" if has_goods else "without bag"
        color = (0, 255, 0) if has_goods else (255, 0, 0)
        cv2.putText(frame, f"ID: {id} - {label}", (x3, y3 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)

    # Draw ROIs
    cv2.polylines(frame, [np.array(area1, np.int32)], True, (255, 0, 0), 2)
    cv2.putText(frame, 'Area 1 (Exit)', (497, 462), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
    cv2.polylines(frame, [np.array(area2, np.int32)], True, (255, 0, 0), 2)
    cv2.putText(frame, 'Area 2 (Enter)', (289, 390), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
    
    # Draw the combined bag check region
    cv2.polylines(frame, [np.array(bag_check_region, np.int32)], True, (255, 255, 0), 2)
    cv2.putText(frame, 'Bag Check Region', (min_x, min_y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 1)

    # Display simplified counts
    cv2.putText(frame, f"Total Entering: {len(entering)}", (20, 44), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Total Exiting: {len(exiting)}", (20, 82), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Total with Bags: {len(people_with_bags)}", (20, 120), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 0), 2)

    frame_for_writer = cv2.resize(frame, (frame_width, frame_height))
    out.write(frame_for_writer)
    
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# --- Save Prediction Data ---
output_dir = r'../data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_file = os.path.join(output_dir, 'predicted_bags.json')
with open(output_file, 'w') as f:
    json.dump(predicted_bags, f, indent=4)
    
print("-" * 50)
print(f"Prediction data saved to: {output_file}")
print("Now you can compare 'predicted_bags.json' with 'ground_truth_bags_auto.json'.")

cap.release()
out.release()
cv2.destroyAllWindows()