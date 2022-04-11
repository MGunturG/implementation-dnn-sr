import os
import sys
import time
import argparse
import cv2 as cv
from cv2 import dnn_superres

if __name__ == "__main__":
	# Just add this for calculate processing time. You can ignore this.
	# start_time indicate the start time of the process.
	# end_time indicate the end time of the process (finish).
	start_time = time.time() 

	# Create an SuperRes object
	sr = dnn_superres.DnnSuperResImpl_create()

	# Read image or input image. The image format must be PNG or JPG.
	image_input = cv.imread('input.png')

	'''
	Read the pre-trained model.
	Model you can use are:
	1. EDSR
	2. ESPCN
	3. FSRCNN
	4. LapSRN
	
	The default is EDSR_3.pb.
	For more info which model you should use, please check README.md.
	'''
	model_path = "models/EDSR/EDSR_x3.pb"
	sr.readModel(model_path)

	# Set the desired model and scale to get correct pre- and post-processing
	sr.setModel("edsr", 3)

	# Upscale selected image
	result = sr.upsample(image_input)

	# end_time
	end_time = time.time()
	print("Processing time: {}".format(end_time-start_time))

	# Show the result (upscaled image)
	cv.imshow("Upscaled image", result)
	cv.waitKey(0)
	cv.destroyAllWindows()

	# Save image to local disk
	# Un-comment line below this to save the upscaled image
	cv.imwrite("result_upscaled.png", result)

	# end of the program
	print("Done.")

