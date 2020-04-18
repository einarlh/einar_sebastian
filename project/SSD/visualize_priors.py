import matplotlib.pyplot as plt
import sys
import torch
import numpy as np
import pathlib
path = pathlib.Path()
# Insert all modules a folder above
sys.path.insert(0, str(path.absolute().parent))
from ssd.config.defaults import cfg
from ssd.modeling.box_head.prior_box import PriorBox
from ssd.utils.box_utils import convert_locations_to_boxes

config_path = "configs/train_waymo.yaml"
cfg.merge_from_file(config_path)
prior_box = PriorBox(cfg)

priors = prior_box()
print("Prior box shape:", priors.shape)
print("First prior example:", priors[5])
locations = torch.zeros_like(priors)[None]
priors_as_location = convert_locations_to_boxes(locations, priors,cfg.MODEL.CENTER_VARIANCE, cfg.MODEL.SIZE_VARIANCE)[0]

def plot_bbox(box):
    cx, cy, w, h = box
    x1, y1 = cx + w/2, cy + h/2
    print(y1)
    x0, y0 = cx - w/2, cy - h/2
    print(y0)
    plt.plot(
        [x0, x0, x1, x1, x0],
        [y0, y1, y1, y0, y0]
    )

prior_idx = 10
#plt.clf()
plt.ylim([0, 1])
plt.xlim([0, 1])
# Visualizing all would take too much
priors_as_location = [x for x in priors_as_location]
# np.random.shuffle(priors_as_location)
for prior in priors_as_location[prior_idx-10:prior_idx]:
    plot_bbox(prior)

plt.savefig("Priorboxes.jpg")


print(prior)