import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from IPython import display
import imageio as imageio
import cv2
from nndrain.simplify_linear import SimplifyLinear

def set_seed(seed):
    """
    Set seed for random, numpy, and torch modules
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def plot_weights(fig, weights, title, ax_title=None, vmin=None, vmax=None):
    """
    Plot weights as images
    """
    ax = []
    columns = len(weights)
    rows = 1
    for i in range(1, columns*rows+1):
        ax.append(fig.add_subplot(rows, columns, i))
        if ax_title is not None:
            ax[-1].set_title(ax_title[i-1], fontsize=9)
        plt.imshow(weights[i-1].cpu(), vmin=vmin, vmax=vmax)

    plt.suptitle(title, fontsize=18, y=0.12)
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
