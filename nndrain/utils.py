import torch
import numpy as np
import random
import matplotlib.pyplot as plt
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

def plot_weights(fig, weights, title, ax_title=None, vmin=None, vmax=None):
    """
    Plot weights as images
    """
    ax = []
    columns = len(weights)

    for i in range(1, columns+1):
        ax.append(fig.add_subplot(1, columns, i))
        if ax_title is not None:
            ax[-1].set_title(ax_title[i-1], fontsize=9)
        plt.imshow(weights[i-1].cpu(), vmin=vmin, vmax=vmax)

    plt.suptitle(title, fontsize=16)
    
    fig.tight_layout()

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

def merge_images(fw_name, fc_name):
    """
    Merge two images vertically
    """
    fw = Image.open(fw_name)
    fc = Image.open(fc_name)
    # Set the size of the resulting image
    result_size = (fw.size[0], fw.size[1] + fc.size[1])

    # Create an empty image with the desired size
    result = Image.new('RGB', result_size)

    # Paste the two images into the resulting image
    result.paste(fw, (0, 0))
    result.paste(fc, (0, fw.size[1]))

    return result

