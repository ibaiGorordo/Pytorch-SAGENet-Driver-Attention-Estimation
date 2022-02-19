import cv2
import pafy
import numpy as np

from sagenet import SAGENet

# Initialize video
# cap = cv2.VideoCapture("video.mp4")

videoUrl = 'https://youtu.be/bUhFfunT2ds'
start_time = 50 # skip first {start_time} seconds
videoPafy = pafy.new(videoUrl)
print(videoPafy.streams)
cap = cv2.VideoCapture(videoPafy.streams[-1].url)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time*30)

# Initialize model
model_path = "models/sage_5epo_180600step.ckpt"
attention_estimator = SAGENet(model_path, use_gpu=True)

# out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (1280*2,720))

cv2.namedWindow("Attention heatmap", cv2.WINDOW_NORMAL)	
while cap.isOpened():

	try:
		# Read frame from the video
		ret, frame = cap.read()
		if not ret:	
			break
	except:
		continue
	
	# Estimate attention and colorize it
	heatmap = attention_estimator(frame)
	color_heatmap = attention_estimator.draw_heatmap(frame)

	combined_img = np.hstack((frame, color_heatmap))

	cv2.imshow("Attention heatmap", combined_img)
	# out.write(combined_img)

	# Press key q to stop
	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
# out.release()
cv2.destroyAllWindows()