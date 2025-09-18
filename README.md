# Entrance and Exit Monitoring and Bag Detection System
## Project Overview
- This project is a computer vision system built on the YOLOv8 model for real-time video surveillance and analysis. 
- The system is designed to perform several key functions:

  - Person Tracking: Identifies and tracks each unique person moving through the video frame.

  - Entry and Exit Counting: Counts the number of people entering and exiting designated areas of interest (ROIs).

  - Bag Detection: Determines if people are carrying bags or other predefined items as they pass through a specific checkpoint region.

- Data Logging: Saves bag detection predictions to a JSON file for later analysis.

- Key Features
Multi-Object Tracking (MOT): Employs a tracking algorithm to assign and maintain a unique ID for each person, ensuring consistent monitoring.

- Customizable Regions of Interest (ROIs): The areas for entry, exit, and bag checks can be easily customized by adjusting the coordinate points in the code.

- Visual Output: Generates a new video file with on-screen annotations, including bounding boxes, person IDs, and real-time counts, for easy visual monitoring.

- Data Export: Predictions about bag presence are saved to a JSON file, enabling statistical analysis and comparisons.
## How Bag Detection Works: Intersection over Union (IoU)
- The core of the bag detection logic relies on a concept called Intersection over Union (IoU). IoU is a metric used to measure the overlap between two bounding boxes.

- ## What is IoU?

  - The IoU value is a ratio calculated by dividing the Area of Intersection by the Area of Union of two bounding boxes.

  - Area of Intersection: The area where the two bounding boxes overlap.

   - Area of Union: The total area covered by both bounding boxes combined.

  - The result is a value between 0 and 1, where:

     - IoU = 0: The boxes do not overlap at all.

     - IoU = 1: The boxes are a perfect match.

   - IoU in This Project

     - the system performs the following steps for bag detection:

      - For each detected person object, it gets its bounding box.

      - It then compares the person's bounding box to the bounding box of every other detected object that is classified as a bag.

      - It calculates the IoU between the person's bounding box and the bag's bounding box.

      - A threshold of 0.1 is used. If the calculated IoU value is greater than this threshold (IoU > 0.1), the system determines that the person is carrying the bag.

    - This approach allows the model to reliably associate a bag with the correct person, even in crowded scenes where bounding boxes might be close together.
## Automated Ground Truth Creation
- To accurately measure the performance of your main bag detection system, you need a "ground truth" dataset. This script automates the process of creating this dataset interactively.

- It works by playing the video and pausing automatically whenever a person is detected in a predefined region. At each pause, you are prompted to manually annotate whether the person has a bag (1) or not (0).

- This process ensures that your evaluation is based on a high-quality, manually verified dataset, which is crucial for calculating the model's accuracy. The script saves your annotations to a ground_truth_bags_auto.json file.

- ### How to Use the Ground Truth Script
   - Open and run the ground_truth_creator.py script.

   - Follow the instructions in your terminal to provide annotations.

   - Enter 1 for a person with a bag or 0 for a person without a bag.

   - Press q at any time to stop the process and save your annotations.

## Model Performance Metrics
- Based on a test of 5 manually annotated samples, the model achieved the following performance metrics
 ```
 Total samples evaluated: 5
Accuracy: 1.0000
Precision: 1.0000
Recall: 1.0000
F1-Score: 1.0000
--------------------------------------------------
Confusion Matrix:
           Predicted (No Bag)  Predicted (Bag)
True (No Bag)      4                   0
True (Bag)         0                   1
 ```
- ## How to Calculate the Accuracy
  - The model's performance metrics are calculated by comparing its predictions (predicted_bags.json) against the manually annotated ground_truth data (ground_truth_bags_auto.json).

   - The calculation relies on four key categories of outcomes for each prediction:

   - ### True Positive (TP): 
       - The model correctly predicted that a person has a bag.

   - ### True Negative (TN): 
       - The model correctly predicted that a person does not have a bag.

   - ### False Positive (FP): 
       - The model incorrectly predicted that a person has a bag (when they do not).

    - ### False Negative (FN): 
       - The model incorrectly predicted that a person does not have a bag (when they do).

   - The Accuracy of the model is then calculated using the following formula:
   ```
       Accuracy= TP+TN /TP+TN+FP+FN 
    ``` 


- ### Code Structure
  - The script is logically structured into the following main sections:

    - Model Loading: The YOLOv8 model is loaded for object detection.

    - Object Detection: The model is used to detect persons and goods in each video frame.

    - Tracking: The Tracker class assigns a unique ID to each detected person.

    - Entry/Exit Logic: cv2.pointPolygonTest is used to determine a person's location relative to the defined ROIs to log entries and exits.

    - Bag Check: The Intersection over Union (IoU) of person and goods bounding boxes is calculated to determine if a person is carrying a bag.

    - Data Persistence: The final detection data is saved to a JSON file for easy access and analysis.
## Run the script using the following command:
- 1. Create and activate a Python virtual environment:
```
# On Windows
python -m venv venv
```
```
venv\Scripts\activate
```
```
cd venv
```
```
md src
```
```
cd src
```
```
cd Entrance-and-Exit-Monitoring-and-Bag-Detection-System
```
- 2. Clone the repository:
```
git clone https://github.com/lamiaakhairyibrahim/Entrance-and-Exit-Monitoring-and-Bag-Detection-System.git
```
- 3. Install the required libraries 
```
pip install -r requirements.txt
```
- 4. Run the main script
```
python collect_predect.py
```
- 5. caculat the accurcy
```
python calculat_accurcy.py
```



- A window will appear showing the processed video in real-time.,You can press the q key to exit the program.

- After processing is complete, the output video will be saved to the path specified by output_path, and the bag detection data will be saved to predicted_bags.json.



