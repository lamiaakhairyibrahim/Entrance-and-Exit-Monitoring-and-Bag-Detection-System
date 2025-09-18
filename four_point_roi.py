import cv2

# Set the path to your video file
VIDEO_PATH = r'../data\video3.mp4'

# Dimensions for the display window
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 450

# List to store the ROI points
roi_points = []
# A copy of the display frame to draw on without re-reading
temp_display_frame = None

# Mouse callback function to get the points and draw circles
def get_roi_points(event, x, y, flags, param):
    global roi_points, temp_display_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(roi_points) < 4:
            # Convert mouse coordinates from the display window to original frame dimensions
            original_x = int(x * (frame.shape[1] / DISPLAY_WIDTH))
            original_y = int(y * (frame.shape[0] / DISPLAY_HEIGHT))
            roi_points.append((original_x, original_y))
            
            # Draw a circle on the temporary display frame at the clicked point
            cv2.circle(temp_display_frame, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('Select 4 ROI Points (Press Q to exit)', temp_display_frame)
            
            print(f"Point {len(roi_points)} selected at: ({original_x}, {original_y})")
        
        if len(roi_points) == 4:
            print("\nFour points have been selected. The ROI boundary points are:")
            print(roi_points)

# Open the video file
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Error: Could not open video file at {VIDEO_PATH}")
    exit()

# Read the first frame
ret, frame = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    exit()

# Resize the original frame for display purposes
display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
# Create a copy to draw on
temp_display_frame = display_frame.copy()

# Create a window and set the mouse callback function
cv2.namedWindow('Select 4 ROI Points (Press Q to exit)')
cv2.setMouseCallback('Select 4 ROI Points (Press Q to exit)', get_roi_points)

print("Please click on 4 points with your mouse to define the ROI boundary.")
print("The frame is resized for display only. The points will be saved relative to the original frame.")
print("Press 'Q' to exit at any time.")

# Display the resized frame and wait for user input
while True:
    cv2.imshow('Select 4 ROI Points (Press Q to exit)', temp_display_frame)
    # Press 'Q' or select 4 points to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q') or len(roi_points) == 4:
        break

cap.release()
cv2.destroyAllWindows()