from imutils import paths
import dlib
import cv2
from google.colab.patches import cv2_imshow

# Path to your trained detector
detector_path = "/content/drive/MyDrive/image_process/crab_crayfish_detector.svm"

# Paths to your testing images directories
crab_testing_path = "/content/drive/MyDrive/image_process/Final_resources/testing/crab"
crayfish_testing_path = "/content/drive/MyDrive/image_process/Final_resources/testing/crayfish"

# Load the detector
detector = dlib.simple_object_detector(detector_path)

# Initialize class counters
class_counts = {"crab": 0, "crayfish": 0, "unknown": 0}

# Function to test on a directory
def test_on_directory(directory_path, label):
    for imagePath in paths.list_images(directory_path):
        # Load the image and make predictions
        image = cv2.imread(imagePath)
        boxes = detector(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Determine prediction
        prediction = "unknown"  # Default to unknown
        if len(boxes) > 0:
            prediction = label  # If boxes are detected, use the provided label

        # Update class counters
        class_counts[prediction] += 1

        # Print prediction for the current image
        print(f"Image: {imagePath}, Prediction: {prediction}")

        # Loop over the bounding boxes and draw them (if any)
        for b in boxes:
            (x, y, w, h) = (b.left(), b.top(), b.right(), b.bottom())
            cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)
            cv2.putText(image, prediction, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show the image
        cv2_imshow(image)
        cv2.waitKey(0)

# Test on crab images
print("[INFO] Testing on crab images...")
test_on_directory(crab_testing_path, "crab")

# Test on crayfish images
print("[INFO] Testing on crayfish images...")
test_on_directory(crayfish_testing_path, "crayfish")

# Print the total count for each class
print("\n[INFO] Classification Summary:")
for class_label, count in class_counts.items():
    print(f"{class_label}: {count} images")

cv2.destroyAllWindows()