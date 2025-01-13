
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

# function to plot preditions of as ResNet50 as a bar chart
def plot_resnet50_predictions_as_chart(decoded_predictions):
    # extract class names and scores
    class_names = [prediction[1] for prediction in decoded_predictions]
    scores = [prediction[2] for prediction in decoded_predictions]

    # build chart
    fig, ax = plt.subplots(figsize=(10, 5))
    indices = np.arange(len(scores))  # Erstelle die x-Achsen-Indizes

    # plot bars
    ax.bar(indices, scores, color='#009999')
    ax.set_xlabel('Classes')
    ax.set_ylabel('Probability')
    ax.set_title('Model Prediction')
    ax.set_xticks(indices)
    ax.set_xticklabels(class_names)  # show class names on x axis
    ax.set_yticks(np.arange(0, 1.1, 0.1))  # set y axis from 0 to 1
    ax.grid(True)

    plt.tight_layout()
    plt.show()

# function to display several images from a given dataset
import os
from tensorflow.keras.preprocessing.image import load_img
def plot_images_from_dataset(dataset_path, class_names, start_index, num_images=12, image_size=(224, 224)):
    # prepare lists of all images and labels
    image_paths = []
    labels = []
    
    # go over all class folders
    for label, class_name in enumerate(class_names):
        class_folder = os.path.join(dataset_path, class_name)
        class_images = [os.path.join(class_folder, f) for f in os.listdir(class_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        
        # add images and corresponding labels
        image_paths.extend(class_images)
        labels.extend([label] * len(class_images))
    
    # select images & labels in given area
    selected_images = image_paths[start_index:start_index + num_images]
    selected_labels = labels[start_index:start_index + num_images]
    
    # build plot
    fig, axes = plt.subplots(1, len(selected_images), figsize=(15, 10))
    for i, ax in enumerate(axes.flat):
        # load and plot image
        image = load_img(selected_images[i], target_size=image_size)
        ax.imshow(image)
        ax.set_title(f"Label: {class_names[selected_labels[i]]}")
        ax.axis("off")
    
    plt.tight_layout()
    plt.show()      

# function to display an image on the left and the predictions a bar-chart on the right
def plot_image_and_chart(sample_image, sample_label, predictions, class_names):
    # create 2 plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))  # 1 row, 2 columns

    # left plot: show image and label
    axes[0].imshow(sample_image.numpy().astype("uint8"))
    axes[0].set_title(f"Label: {class_names[sample_label.numpy()]}")
    axes[0].axis("off")  

    # right plot: show predictions a bar-chart
    probabilities = predictions[0]
    indices = np.arange(len(probabilities))
    axes[1].bar(indices, probabilities, color='#009999')
    axes[1].set_xlabel('Classes')
    axes[1].set_ylabel('Probability')
    axes[1].set_title('Model Prediction')
    axes[1].set_xticks(indices)
    axes[1].set_xticklabels(class_names)  # show class names on x axis
    axes[1].set_yticks(np.arange(0, 1.1, 0.1))
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

# function to display an additional downloaded image (not part of TS)on the left and the predictions a bar-chart on the right
def plot_new_image_and_chart(image_path, label, predictions, class_names):
    # create 2 plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))  # 1 row, 2 columns

    # left plot: show image and label
    img = load_img(image_path)  # Load the image for displaying
    axes[0].imshow(img)
    axes[0].axis("off")  # Hide axes for better view
    axes[0].set_title("Label: " + label)

    # right plot: show predictions as a bar-chart
    probabilities = predictions[0]
    indices = np.arange(len(probabilities))
    axes[1].bar(indices, probabilities, color='#009999')
    axes[1].set_xlabel('Classes')
    axes[1].set_ylabel('Probability')
    axes[1].set_title('Model Prediction')
    axes[1].set_xticks(indices)
    axes[1].set_xticklabels(class_names)  # show class names on x-axis
    axes[1].set_yticks(np.arange(0, 1.1, 0.1))
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()  