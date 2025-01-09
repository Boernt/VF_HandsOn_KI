
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

# Define a custom colormap for plotting diagrams:
colors = ['darkred', 'lightgrey', '#009999']  # Dark red for negative, light grey for near zero, and petrol for positive
cmap_custom = mcolors.LinearSegmentedColormap.from_list('custom', colors, N=256)

# Plot images from the Trainings-Set
def plot_images(start_index, X, Y):
    # plot first few images
    fig, axes = plt.subplots(1, 12, figsize=(25, 10))# create subplots

    # plot each image
    for i, ax in enumerate(axes.flat):
        # plot raw pixel data
        ax.imshow(X[start_index + i], cmap=plt.get_cmap('gray'))
        ax.set_title('Label: ' + str(Y[start_index + i]))

    # adjust spacing between subplots
    plt.subplots_adjust(hspace=0.1, wspace=1.0)

    # display the plot
    plt.show()

def plot_weights_layer_1(model, num_units=16): # Assuming standard layer 1 has 16 units
    # Get the weights of the first Dense layer
    weights = model.layers[1].get_weights()[0]

    # Create subplots
    fig, axes = plt.subplots(2, 8, figsize=(16, 6))

    # Plot weights for each unit
    for i in range(num_units):
        # Get weights for the current unit
        weights_unit = weights[:, i]
        
        # Reshape the weights into a 28x28 array
        weights_reshaped = np.reshape(weights_unit, (28, 28))
        
        # Plot the weights in the corresponding subplot
        ax = axes[i // 8, i % 8]
        ax.imshow(weights_reshaped, cmap=cmap_custom)
        ax.set_title(f'Unit {i+1}')
        ax.axis('off')

    # Adjust layout
    plt.tight_layout()
    plt.show()

def plot_weights_layer_1_10units(model): # Plot function for 10 units in layer 1 to have same format as small TS to fit this plot nicely above the small TS plot
    # Get the weights of the first Dense layer
    weights = model.layers[1].get_weights()[0]

    # Create subplots
    fig, axes = plt.subplots(1, 10, figsize=(20, 10))

    # Plot weights for each unit
    for i in range(10):
        # Get weights for the current unit
        weights_unit = weights[:, i]
        
        # Reshape the weights into a 28x28 array
        weights_reshaped = np.reshape(weights_unit, (28, 28))
        
        plt.subplot(1, 10, i+1)  # 1 row, 10 columns, i+1 plot
        plt.imshow(weights_reshaped, cmap=cmap_custom)
        plt.title(str(f'Unit {i+1}'))

def plot_weights_cnn_filter(cnn_model):
    # Get the weights of the first Conv2D layer
    weights = cnn_model.layers[0].get_weights()[0]

    # Create subplots
    fig, axes = plt.subplots(2, 16, figsize=(16, 3))

    # Plot weights for 32 filters
    for i in range(32):
        # Get weights for the current filter
        filter_weights = weights[:, :, :, i]  # Shape: (filter_height, filter_width, input_channels)
        
        # Plot the weights in the corresponding subplot
        ax = axes[i //16, i %16]
        ax.imshow(filter_weights[:, :, 0], cmap=cmap_custom)  # Assuming only one input channel
        ax.set_title(f'Filter {i+1}')
        ax.axis('off')

    # Adjust layout
    plt.tight_layout()
    plt.show()

# plot history of training and valdation progress
def plot_models_history(history):
    df_loss_acc = pd.DataFrame(history.history)

    df_loss= df_loss_acc[['loss','val_loss']]
    df_loss.rename(columns={'loss':'Training','val_loss':'Test'},inplace=True)

    df_acc= df_loss_acc[['sparse_categorical_accuracy','val_sparse_categorical_accuracy']]
    df_acc.rename(columns={'sparse_categorical_accuracy':'Training','val_sparse_categorical_accuracy':'Test'},inplace=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Plot loss
    df_loss.plot(ax=axes[0], title='Model Loss', xlabel='Epoch', ylabel='Loss', figsize=(12, 3))

    # Plot accuracy
    df_acc.plot(ax=axes[1], title='Model Accuracy', xlabel='Epoch', ylabel='Accuracy', figsize=(12, 3))

    plt.tight_layout()
    plt.show()


# plot only history of training progress
def plot_models_training_history(history):
    df_loss_acc = pd.DataFrame(history.history)

    df_loss= df_loss_acc[['loss']]
    df_loss.rename(columns={'loss':'Training'},inplace=True)

    df_acc= df_loss_acc[['sparse_categorical_accuracy']]
    df_acc.rename(columns={'sparse_categorical_accuracy':'Training'},inplace=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Plot loss
    df_loss.plot(ax=axes[0], title='Model Loss', xlabel='Epoch', ylabel='Loss', figsize=(12, 3))

    # Plot accuracy
    df_acc.plot(ax=axes[1], title='Model Accuracy', xlabel='Epoch', ylabel='Accuracy', figsize=(12, 3))

    plt.tight_layout()
    plt.show()

# plot predictions to a chosen TE as bar graph, plot the corresponding TE (image), too
def plot_prediction(index, model_prediction, X_dataset, Y_dataset):
    fig, axes = plt.subplots(1, 2, figsize=(20, 5))  # 1 row, 2 columns

    # Plot chosen image and label
    axes[0].imshow(X_dataset[index], cmap=plt.get_cmap('gray'))
    axes[0].set_title('Label: ' + str(Y_dataset[index]))

    # Plot the prediction as fbar graph
    probabilities = model_prediction[0]
    indices = np.arange(len(probabilities))

    axes[1].bar(indices, probabilities, color='#009999')
    axes[1].set_xlabel('Classes (Numbers 0-9)')
    axes[1].set_ylabel('Probability')
    axes[1].set_title('Model Prediction')
    axes[1].set_xticks(indices)
    axes[1].set_yticks(np.arange(0, 1.0, 0.1))
    axes[1].grid(True)

    plt.show()

    # Print predictions for all classes as percentage with two decimal places
    print('Model prediction:', ' '.join(['{:.2f}%'.format(value * 100) for value in model_prediction[0]]))

# Plot a subset of wrong predicted TEs. Set the starting index to select different images
def plot_wrong_predictions(prediction_array, X_dataset, Y_dataset, index):
    # Creating the new array bulk_argmax with input index and predicted class
    bulk_argmax = []
    for i in range(len(prediction_array)):
        prediction = prediction_array[i]
        predicted_class = np.argmax(prediction)
        input_index = i  # Index of the input element from X_train
        bulk_argmax.append((input_index, predicted_class))

    wrong_predictions = []
    for i in range(len(bulk_argmax)):
        predicted_class = bulk_argmax[i][1]
        true_label = Y_dataset[bulk_argmax[i][0]]  # Use the input index to get the true label
        if predicted_class != true_label:
            wrong_predictions.append((bulk_argmax[i][0], predicted_class))

    num_plots = min(16, len(wrong_predictions))
    num_rows = 2  # 2 rows

    fig, axes = plt.subplots(num_rows, 8, figsize=(16, 4))  # 8 images per row

    for i, (index, predicted_class) in enumerate(wrong_predictions[index:index+num_plots]):
        row = i // 8
        col = i % 8
        ax = axes[row, col]
        ax.imshow(X_dataset[index], cmap='gray')
        ax.set_title(f"Label: {Y_dataset[index]}, Pred.: {predicted_class}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()

#Plot the small TS for Overfitting in one row
def plot_small_ts(X_dataset, Y_dataset):
       # plot chosen image and label:
    fig, axes = plt.subplots(1, X_dataset.shape[0], figsize=(20, 10))

    for i in range(X_dataset.shape[0]):
        plt.subplot(1, X_dataset.shape[0], i+1)  # 1 row, X_train_overfit.shape[0] columns, i+1 plot
        plt.imshow(X_dataset[i], cmap=plt.get_cmap('gray'))
        plt.title(str(Y_dataset[i]))

#Plot intermediate results after convolutions in the CNN
def plot_intermediate_conv_results(layer_names, intermediate_results, layer_types): 
    # Plot results
    for layer_name, layer_results, layer_type in zip(layer_names, intermediate_results, layer_types):

        if layer_type == 'Conv2D': # only plot conv_layers
           # layer shape: shape [batch_size, height, width, channels]
            n_channels = layer_results.shape[-1]  # get number of channels in current layer
            size       = layer_results.shape[ 1]  # get channel size (hight = width)

    
            display_grid = np.zeros((size, size * n_channels))# build plot grid corresponding to layer sizes (one plot for each channel)


    
            for i in range(n_channels):# loop over all channels to sequentially plot them
                x  = layer_results[0, :, :, i]
                # Process the intermediate result back to pixel values to plot them
                x -= x.mean()
                x /= x.std ()
                x *=  64
                x += 128
                x  = np.clip(x, 0, 255).astype('uint8') # check if in range [0-255] and change type to plot
                display_grid[:, i * size : (i + 1) * size] = x # plot each channel into the grid

            #-----------------
            # Display the grid
            #-----------------
            scale = 20. / n_channels
            plt.figure( figsize=(scale * n_channels, scale) )
            plt.title ( layer_name )
            plt.grid  ( False )
            


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
