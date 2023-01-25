import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from IPython import display
import imageio as imageio
from PIL import Image
import cv2


def set_seed(seed):
    """
    Set seed for random, numpy, and torch modules
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def plot_charts(fig, epochs=None, losses=None, train_accuracies=None, test_accuracies=None, simplifications=None, inference_times=None):
    """
    Plot charts for losses, accuracies, simplifications, and inference times
    """
    ax = []
    rows = 0
    rows += 1 if losses is not None else 0
    rows += 1 if (train_accuracies is not None or test_accuracies is not None) else 0
    rows += 1 if simplifications is not None else 0
    rows += 1 if inference_times is not None else 0
    index = 1

    if losses is not None:
        # Create a single loss chart
        ax.append(fig.add_subplot(rows, 1, index))
        ax[-1].plot(epochs, losses)
        ax[-1].set_xlabel("Epoch")
        ax[-1].set_ylabel("Loss")
        index += 1
        
    if train_accuracies is not None or test_accuracies is not None:
        # Create a single train/test accuracies chart
        ax.append(fig.add_subplot(rows, 1, index))
        if train_accuracies is not None:
            ax[-1].plot(epochs, train_accuracies, label="Train accuracy")
        if test_accuracies is not None:
            ax[-1].plot(epochs, test_accuracies, label="Test accuracy")
        ax[-1].set_xlabel("Epoch")
        ax[-1].set_ylabel("Accuracy (%)")
        ax[-1].legend()
        # print last value of test accuracies on the chart
        if test_accuracies is not None and len(test_accuracies)>0:
            ax[-1].annotate("{:.2f}%".format(test_accuracies[-1]),
                            xy=(epochs[-1], test_accuracies[-1]),
                            xytext=(epochs[-1], test_accuracies[-1]))
        index += 1

    if simplifications is not None:
        # Create a single simplification chart
        ax.append(fig.add_subplot(rows, 1, index))
        ax[-1].plot(epochs, simplifications)
        ax[-1].set_xlabel("Epoch")
        ax[-1].set_ylabel("Pruning (%)")
        # print last value of simplifications on the chart
        if len(simplifications)>0:
            ax[-1].annotate("{:.1f}%".format(simplifications[-1]),
                        xy=(epochs[-1], simplifications[-1]),
                        xytext=(epochs[-1], simplifications[-1]))
        index += 1

    if inference_times is not None:
        # Create a single inference time chart
        ax.append(fig.add_subplot(rows, 1, index))
        ax[-1].plot(epochs, inference_times)
        ax[-1].set_xlabel("Epoch")
        ax[-1].set_ylabel("Inference time (ms)")
        # print last value of inference times on the chart
        if len(inference_times)>0:
            ax[-1].annotate("{:.2f} ms".format(inference_times[-1]),
                        xy=(epochs[-1], inference_times[-1]),
                        xytext=(epochs[-1], inference_times[-1]))
        index += 1
        
    fig.tight_layout()

    display.clear_output(wait=True)
    display.display(plt.gcf())

def plot_weights(fig, weights, title, ax_title=None, vmin=-1, vmax=1, colormap='coolwarm'):
    """
    Plot weights as images
    """
    ax = []
    columns = len(weights)
    cmap = plt.get_cmap(colormap)  # specify the colormap to use

    for i in range(1, columns+1):
        ax.append(fig.add_subplot(1, columns, i))
        if ax_title is not None:
            ax[-1].set_title(ax_title[i-1], fontsize=9)
        plt.imshow(weights[i-1].cpu(), cmap=cmap, vmin=vmin, vmax=vmax)

    plt.suptitle(title, fontsize=16)
    
    fig.tight_layout()

    display.clear_output(wait=True)
    display.display(plt.gcf())


def plot_neural_network(fig, weights, title, input_list=None, output_list=None, label_threshold=50, ax_title=None, 
                        linewidth_scale=0.5, vmin=-1, vmax=1, colormap='coolwarm'):
    """
    Plot a neural network using a scatter plot for the neurons and line plots for the connections
    """
    ax = fig.add_subplot(111)
    ax.set_title(title)
    # Get the number of layers
    layers = len(weights)

    # list to store coordinates
    X = []
    Y = []



    # Plot the neurons
    for i in range(layers):
        if i==0:
            x = [i] * np.shape(weights[i])[0]
            y = list(range(len(weights[i])))
            y.reverse()
            X.append(x)
            Y.append(y)

        x = [i+1] * np.shape(weights[i])[1]
        y = list(range(len(weights[i][0])))
        y.reverse()
        # align next layer center
        offset = (len(weights[0]) - len(weights[i][0]))/2
        y = [y[j] + offset for j in range(len(weights[i][0]))]
        X.append(x)
        Y.append(y)

    
    # plot the connections
    norm = colors.Normalize(vmin=vmin, vmax=vmax)  # create an instance of the Normalize class
    cmap = plt.get_cmap(colormap)  # specify the colormap to use
    for i in range(layers):
        for j in range(len(weights[i])):
            for k in range(len(weights[i][j])):
                color = cmap(norm(weights[i][j][k].numpy())) # get the color for the current weight value
                plt.plot([X[i][j], X[i+1][k]], [Y[i][j], Y[i+1][k]], c=color, linewidth = abs(weights[i][j][k])*linewidth_scale)
                
    
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()        
    # marker size
    marker_size = 120/(ymax-ymin)
    # plot the neurons
    for i in range(layers+1):
        for x, y in zip(X[i], Y[i]):
            # plot neurons above the connections
            plt.scatter(x, y, c='gray', s = marker_size, zorder=3)




    # plot the labels only if ymax-ymin is below label_threshold
    if ymax-ymin < label_threshold:
        # Add input labels
        if input_list is not None:
            for i, label in zip(range(len(X[0])), input_list):
                plt.text(X[0][i] - 0.25, Y[0][i], label, ha="center", va="center", rotation=0, color='gray', fontsize=7)
        # Add output labels
        if output_list is not None:
            # if the output is a single value, plot it in the middle of the last layer
            if len(output_list) != len(Y[-1]) and len(output_list) == 1:
                plt.text(X[-1][0] + 0.25, np.mean(Y[-1]), output_list[0], ha="center", va="center", rotation=0, color='gray', fontsize=7)
            else:
                for i, label in zip(range(len(X[-1])), output_list):
                    plt.text(X[-1][i] + 0.25, Y[-1][i], label, ha="center", va="center", rotation=0, color='gray', fontsize=7)

    # axis off
    plt.axis('off')
    display.clear_output(wait=True)
    display.display(plt.gcf())  


def images_to_gif(filenames, gif_name, tail=0):
    """
    Convert a list of image filenames to gif
    """
    with imageio.get_writer(gif_name, mode='I') as writer:
        n_file = len(filenames)
        for i, filename in enumerate(filenames):
            image = imageio.imread(filename)

            writer.append_data(image)
            if i==n_file-1:
                for t in range(tail):
                    writer.append_data(image)

def images_to_avi(filenames, video_name):
    """
    Convert a list of image filenames to avi video
    """
    frame = cv2.imread(filenames[0])
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video = cv2.VideoWriter(video_name, fourcc, 8, (width,height))
    for image in filenames:
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()

def merge_images(image_list):
    """
    Merge a list of images vertically
    """
    images = [Image.open(img) for img in image_list]
    widths, heights = zip(*(i.size for i in images))

    # Set the size of the resulting image
    result_size = (max(widths), sum(heights))

    # Create an empty image with the desired size
    result = Image.new('RGB', result_size)

    # Paste the images into the resulting image
    y_offset = 0
    for img in images:
        result.paste(img, (0, y_offset))
        y_offset += img.size[1]

    return result
