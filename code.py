import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from google.colab import drive

# Set image size
image_size = 128

# Paths to image directories
train_messy = "/content/drive/MyDrive/clean-vs-dirty-room-project-main/Images/train/messy"
train_clean = "/content/drive/MyDrive/clean-vs-dirty-room-project-main/Images/train/clean"
test_messy = "/content/drive/MyDrive/clean-vs-dirty-room-project-main/Images/test_messy"
test_clean = "/content/drive/MyDrive/clean-vs-dirty-room-project-main/Images/test_clean"

# Function to load and resize images alternately from messy and clean folders
def load_data_alternate(messy_dir, clean_dir):
    data = []
    labels = []
    valid_extensions = ('.jpg', '.jpeg', '.png')  # Valid image extensions

    messy_images = [img for img in os.listdir(messy_dir) if img.lower().endswith(valid_extensions)]
    clean_images = [img for img in os.listdir(clean_dir) if img.lower().endswith(valid_extensions)]

    # Ensure equal lengths to avoid index issues
    min_len = min(len(messy_images), len(clean_images))
    messy_images = messy_images[:min_len]
    clean_images = clean_images[:min_len]

    for messy_img, clean_img in tqdm(zip(messy_images, clean_images), desc="Loading images", total=min_len):
        # Load messy image
        messy_path = os.path.join(messy_dir, messy_img)
        messy_img_data = cv2.imread(messy_path, cv2.IMREAD_GRAYSCALE)
        if messy_img_data is not None:
            messy_img_data = cv2.resize(messy_img_data, (image_size, image_size))
            data.append(messy_img_data)
            labels.append(1)  # Label for messy

        # Load clean image
        clean_path = os.path.join(clean_dir, clean_img)
        clean_img_data = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)
        if clean_img_data is not None:
            clean_img_data = cv2.resize(clean_img_data, (image_size, image_size))
            data.append(clean_img_data)
            labels.append(0)  # Label for clean

    return np.array(data), np.array(labels)

# Function to normalize and standardize the images (preprocessing)
def preprocess_images(img_data):
    # Normalize and standardize
    img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))  # Normalization
    return img_data

# Function to append images to the test set and visualize
def append_images_to_test():
    append_choice = input("Do you want to append any images to the test set? (yes/no): ").strip().lower()

    # If user chooses "no", simply exit the function
    if append_choice == 'no':
        print("No images will be appended to the test set.")
        return  # Exit the function without appending any data

    if append_choice == 'yes':
        # Ask for the folder with new test images from Google Drive
        drive.mount('/content/drive')  # Mount Google Drive to access the folder
        folder_path = "/content/drive/MyDrive/ext_testing"  # The folder containing new images

        # Check if the folder exists
        if not os.path.exists(folder_path):
            print(f"Error: The folder {folder_path} does not exist.")
            return

        # Get all image files in the folder
        images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not images:
            print(f"Error: No valid images found in {folder_path}.")
            return

        # Load and preprocess the images
        new_test_data = []
        new_test_labels = []

        for img in tqdm(images, desc="Loading new test images"):
            img_path = os.path.join(folder_path, img)
            img_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img_data is not None:
                img_data = cv2.resize(img_data, (image_size, image_size))
                img_data = preprocess_images(img_data)  # Normalize and standardize
                new_test_data.append(img_data)

                # Assuming the first half of the images are messy (label 1) and second half are clean (label 0)
                # This is a simple rule; feel free to modify based on your labeling convention
                label = 1 if len(new_test_labels) < len(images) // 2 else 0
                new_test_labels.append(label)

        # Convert to numpy arrays and flatten
        new_test_data = np.array(new_test_data).reshape(len(new_test_data), -1)  # Flatten to 2D
        new_test_labels = np.array(new_test_labels)

        # Append the new test data to the existing test data
        global x_test, y_test
        x_test = np.concatenate((x_test, new_test_data), axis=0)
        y_test = np.concatenate((y_test, new_test_labels), axis=0)

        print("New images have been added to the test set.")

        # Store the appended data for use in predictions and accuracy calculation
        global appended_test_data, appended_test_labels
        appended_test_data = new_test_data
        appended_test_labels = new_test_labels

        # Set the flag indicating appended images
        global predictions_displayed
        predictions_displayed = True  # Indicate that predictions for appended images should be shown

# Load train and test data alternately
train_data, train_labels = load_data_alternate(train_messy, train_clean)
print("Loaded training images.")

test_data, test_labels = load_data_alternate(test_messy, test_clean)
print("Loaded testing images.")

# Normalize the image data
x_data = np.concatenate((train_data, test_data), axis=0)
x_data = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

# Combine training and test labels
y_data = np.concatenate((train_labels, test_labels), axis=0)

# Split into training and testing sets (15% test data)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=42)

# Keep the original test set for evaluation
x_test_original = x_test.copy()
y_test_original = y_test.copy()

# Reshape image data for the model (flatten the images)
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# Flatten the original test set as well for compatibility with the model
x_test_original = x_test_original.reshape(x_test_original.shape[0], -1)

# Ask user to append images to the test set
predictions_displayed = False  # Initialize the flag
append_images_to_test()

# Train Logistic Regression model
model = LogisticRegression(max_iter=150)
model.fit(x_train, y_train)

# Evaluate the model using only the original test data (not the appended images)
score = model.score(x_test_original, y_test_original)
print(f"Model accuracy on original test data: {score * 100:.2f}%")

# Show predictions for the original test set
predictions_original = model.predict(x_test_original)
pred_labels_original = ["Messy" if pred == 1 else "Clean" for pred in predictions_original]

# Display predictions for the original test images
print("Predictions for the original test images:")
for i in range(len(pred_labels_original)):
    plt.imshow(x_test_original[i].reshape(image_size, image_size), cmap='gray')
    plt.title(f"Predicted: {pred_labels_original[i]}")
    plt.axis('off')
    plt.show()

# If new images were appended and the user chose to append images, display the predictions for appended images
if predictions_displayed:
    predictions = model.predict(appended_test_data)

    # Map predictions to labels (Messy or Clean)
    pred_labels = ["Messy" if pred == 1 else "Clean" for pred in predictions]

    # Display accuracy for appended test images
    # appended_accuracy = model.score(appended_test_data, appended_test_labels)
    # print(f"Accuracy on appended test images: {appended_accuracy * 100:.2f}%")

    # Print the accuracy on the original test data first, before predictions for the appended images
    print("Model accuracy on original test data:")
    print(f"{score * 100:.2f}%")

    # Visualize the results for the newly appended test images
    print("Predictions for appended images:")
    for i in range(len(appended_test_labels)):
        plt.imshow(appended_test_data[i].reshape(image_size, image_size), cmap='gray')
        plt.title(f"Predicted: {pred_labels[i]}")
        plt.axis('off')
        plt.show()

# If no images were appended, skip predictions for appended images
else:
    print("No new images were appended to the test set.")