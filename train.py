# import the package
from __future__ import print_function
from imutils import paths
import dlib
import sys
import os
from scipy.io import loadmat
from skimage import io

# detect Python 3
if sys.version_info > (3,):
    long = int

# define the path
dataset_path = "/content/drive/MyDrive/image_process/Final_resources/training"  # base dataset path
annotations_base_path = "/content/drive/MyDrive/image_process/Final_resources/annotations" # base annotation path
output_path = "/content/drive/MyDrive/image_process/crab_crayfish_detector.svm"  # output path

# define class
class_labels = ["crab", "crayfish"]

# initialize trainer's imageand edge
images = []
all_boxes = []

# Set aspect ratio and size thresholds
min_aspect_ratio = 0.5  # Adjust as needed
max_aspect_ratio = 2.0  # Adjust as needed
min_box_area = 400      # Adjust as needed

# loop for every class
for label in class_labels:
    # get dataset and annotation path
    class_path = os.path.join(dataset_path, label)
    annotations_path = os.path.join(annotations_base_path, label) # usting base path of annotation

    # loop for image's path
    for imagePath in paths.list_images(class_path):
        # get image ID from image and load the annotation
        imageID = imagePath[imagePath.rfind("/") + 1:].split(".")[0]
        annotation_file = os.path.join(annotations_path, "annotation_" + imageID.split("_")[1] + ".mat")

        # Check if the annotation file exists
        if not os.path.exists(annotation_file):
            raise FileNotFoundError(f"Annotation file not found for image: {imagePath} "
                                    f"Expected annotation file: {annotation_file}")

        # Load .mat file using loadmat
        annotations = loadmat(annotation_file)["box_coord"]

        # Store bounding boxes for each image separately
        image_boxes = []

        # Extract bounding box information and filter
        for (y, h, x, w) in annotations:
            bb = dlib.rectangle(left=long(x), top=long(y), right=long(w), bottom=long(h))
            aspect_ratio = float(bb.width()) / bb.height()
            box_area = bb.width() * bb.height()

            if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio and box_area >= min_box_area:
                image_boxes.append(bb)

        # Append the filtered image_boxes to all_boxes
        all_boxes.append(image_boxes)

        # add the image into the image list
        images.append(io.imread(imagePath))

# get HOG + Linear SVM detector
options = dlib.simple_object_detector_training_options()
options.add_left_right_image_flips = True
options.C = 5
options.num_threads = 4
options.be_verbose = True

# train object detector
print("[INFO]YTrain Object Detector...")
detector = dlib.train_simple_object_detector(images, all_boxes, options)

# save detector into document
print("[INFO] Save Detector Into Document...")
detector.save(output_path)

print("[INFO] Train Complete！")