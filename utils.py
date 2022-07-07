import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from IPython import display
import imageio.v2 as imageio
import cv2

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def plot_weights(fig, weights, title):
    ax = []
    columns = len(weights)
    rows = 1
    for i in range(1, columns*rows+1):
        ax.append(fig.add_subplot(rows, columns, i))
        ax[-1].set_title("W{}".format(i-1))
        plt.imshow(weights[i-1], vmin=-1.0, vmax=1.0)

    plt.suptitle(title, fontsize=18)
    display.clear_output(wait=True)
    display.display(plt.gcf())

def images_to_gif(filenames, gif_name, tail=0):
    with imageio.get_writer(gif_name, mode='I') as writer:
        n_file = len(filenames)
        for i, filename in enumerate(filenames):
            image = imageio.imread(filename)
            writer.append_data(image)
            if i==n_file-1:
                for t in range(tail):
                    writer.append_data(image)

def images_to_avi(filenames, video_name):
    frame = cv2.imread(filenames[0])
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video = cv2.VideoWriter(video_name, fourcc, 8, (width,height))
    for image in filenames:
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()