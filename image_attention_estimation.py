import cv2
import numpy as np
from imread_from_url import imread_from_url

from sagenet import SAGENet

model_path = "models/sage_5epo_180600step.ckpt"
attention_estimator = SAGENet(model_path, use_gpu=True)

# Read inference image
img = imread_from_url("https://upload.wikimedia.org/wikipedia/commons/9/95/Axis_axis_crossing_the_road.JPG")

# Estimate attention and colorize it
attention_estimator(img)
color_heatmap = attention_estimator.draw_heatmap(img)

cv2.imwrite("output.jpg", color_heatmap)

cv2.namedWindow("Attention heatmap", cv2.WINDOW_NORMAL)
cv2.imshow("Attention heatmap", color_heatmap)
cv2.waitKey(0)

