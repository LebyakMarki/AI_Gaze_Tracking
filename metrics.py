import yaml
from dataclasses import dataclass
import matplotlib.pyplot as plt; plt.rcdefaults()
from matplotlib import cm
from math import log10
import numpy as np


@dataclass
class Metrics(object):
    """
    Saving data from yaml
    """
    gaze_points: list
    n_frames: int = 0
    n_heads: int = 0
    up_side: int = 0
    down_side: int = 0
    left_side: int = 0
    right_side: int = 0
    center_side: int = 0


def load_from_yaml(metrics, filename="result.yaml"):
    with open(filename, 'r') as stream:
        try:
            loaded_data = yaml.safe_load(stream)
            fill_metrics_structure(loaded_data, metrics)
        except yaml.YAMLError as exc:
            print(exc)


def fill_metrics_structure(data, metrics):
    points_array = []
    metrics.n_frames = data["number of frames"]
    metrics.n_heads = data["number of faces"]
    first_dict = data["faces and directions"]
    for key1, value1 in first_dict.items():
        dict_items = list(value1.values())
        if dict_items[0] == "center":
            metrics.center_side += 1
        elif dict_items[0] == "left":
            metrics.left_side += 1
        elif dict_items[0] == "right":
            metrics.right_side += 1
        elif dict_items[0] == "up":
            metrics.up_side += 1
        elif dict_items[0] == "down":
            metrics.down_side += 1
        points_array.append([dict_items[2][0] - dict_items[1][0], dict_items[2][1] - dict_items[1][1]])
    metrics.gaze_points = points_array
    return 0


def bar_plots_original_count(metrics):
    objects = ("center", "left", "right", "up", "down")
    y_pos = np.arange(len(objects))
    performance = [metrics.center_side, metrics.left_side, metrics.right_side, metrics.up_side, metrics.down_side]
    plt.figure(figsize=(15, 6))
    plt.plot()
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Count')
    title = "Number of faces: " + str(metrics.n_heads)
    plt.title(title)
    plt.show()


def bar_plots_round_count(metrics):
    labels = ("center", "left", "right", "up", "down")
    data = [metrics.center_side, metrics.left_side, metrics.right_side, metrics.up_side, metrics.down_side]
    n = len(data)
    k = 10 ** int(log10(max(data)))
    m = k * (1 + max(data) // k)
    r = 1.5
    w = r / n
    colors = [cm.terrain(i / n) for i in range(n)]
    fig, ax = plt.subplots()
    ax.axis("equal")
    for i in range(n):
        innerring, _ = ax.pie([m - data[i], data[i]], radius=r - i * w, startangle=90, labels=["", labels[i]],
                              labeldistance=1 - 1 / (1.5 * (n - i)), textprops={"alpha": 0},
                              colors=["white", colors[i]])
        plt.setp(innerring, width=w, edgecolor="white")
    plt.legend()
    plt.show()


def bar_plots_attention(metrics):
    labels = ["Attentive", "Distracted"]
    data = [metrics.center_side, metrics.n_heads - metrics.center_side]
    fig = plt.figure(figsize=(4, 4))
    plt.pie(data, labels=labels)
    plt.show()


def heat_map():
    return 0


if __name__ == "__main__":
    metrics_structure = Metrics([])
    load_from_yaml(metrics_structure)
    # bar_plots_original_count(metrics_structure)
    # bar_plots_round_count(metrics_structure)
    # bar_plots_attention(metrics_structure)
    print(metrics_structure.gaze_points)
else:
    print("Sorry :(")
