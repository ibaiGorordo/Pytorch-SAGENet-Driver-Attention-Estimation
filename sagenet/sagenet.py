import cv2
import torch
import numpy as np

from .network import Unet

INPUT_SIZE = 224

class SAGENet():

	def __init__(self, model_path, use_gpu=False):

		self.use_gpu = use_gpu

		# Initialize model
		self.model = self.initialize_model(model_path, use_gpu)

	def __call__(self, image):
		return self.estimate_attention(image)

	@staticmethod
	def initialize_model(model_path, use_gpu):

		# Load the model architecture
		net = Unet()

		# Load the weights from the downloaded model
		if use_gpu:
			net = net.cuda()
			state_dict = torch.load(model_path, map_location='cuda') # CUDA
		else:
			state_dict = torch.load(model_path, map_location='cpu') # CPU

		# Load the weights into the model
		net.load_state_dict(state_dict)
		net.eval()

		return net

	def estimate_attention(self, image):

		self.img_height, self.img_width = image.shape[:2]

		# Convert the image to match the input of the model
		input_tensor = self.prepare_input(image)

		# Perform inference on the image
		outputs = self.inference(input_tensor)

		# Process output data
		self.heatmap = self.process_output(outputs)
		
		return self.heatmap

	def prepare_input(self, img):

		# Transform the image for inference
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	
		img_input = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))

		img_input = img_input.astype(np.float32)/255
		img_input = img_input.transpose(2, 0, 1)
		img_input = img_input[np.newaxis,:,:,:]    

		input_tensor = torch.from_numpy(img_input)

		if self.use_gpu:
			input_tensor = input_tensor.cuda()

		return input_tensor

	def inference(self, input_tensor):
		with torch.inference_mode():
			outputs, _  = self.model(input_tensor)

		return outputs

	@staticmethod
	def process_output(outputs):		

		# Parse the output of the model
		heatmap = torch.squeeze(outputs[-1]).data.cpu().numpy()

		return heatmap
		
	def draw_heatmap(self, image = None, factor = 0.5):

		heatmap_min = self.heatmap.min()
		heatmap_max = self.heatmap.max()
		norm_heatmap = 255.0 *(self.heatmap-heatmap_min)/(heatmap_max-heatmap_min)
		color_heatmap = cv2.applyColorMap(norm_heatmap.astype(np.uint8), cv2.COLORMAP_JET)

		if image is not None:
			self.img_height, self.img_width = image.shape[:2]

			# Resize and combine it with the RGB image
			color_heatmap = cv2.resize(color_heatmap, (self.img_width, self.img_height))
			color_heatmap = cv2.addWeighted(image, factor, color_heatmap, (1-factor),0)

		return color_heatmap

if __name__ == '__main__':

	from imread_from_url import imread_from_url

	model_path = "../models/sage_5epo_180600step.ckpt"
	img_path = "../demo_img.jpg"

	img = imread_from_url("https://upload.wikimedia.org/wikipedia/commons/9/95/Axis_axis_crossing_the_road.JPG")

	attention_estimator = SAGENet(model_path, True)

	attention_estimator(img)
	color_heatmap = attention_estimator.draw_heatmap(img)

	cv2.imshow("Attention heatmap", color_heatmap)
	cv2.waitKey(0)


