# Important imports
from app import app
from flask import request, render_template, url_for
import cv2
import numpy as np
from PIL import Image
import string
import random
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Adding path to config
app.config['INITIAL_FILE_UPLOADS'] = 'app/static/uploads'

# Route to home page
@app.route("/", methods=["GET", "POST"])
def index():

	# Execute if request is get 
	# Shows us the homepage with white background
	if request.method == "GET":
		full_filename =  'images/white_bg.jpg'
		return render_template("index.html", full_filename = full_filename)

	# Execute if reuqest is post
	if request.method == "POST":

		image_upload = request.files['image_upload']
		imagename = image_upload.filename 

		# generating unique name to save image
		letters = string.ascii_lowercase
		name = ''.join(random.choice(letters) for i in range(10)) + '.png'
		full_filename =  'uploads/' + name
		image_upload.save(os.path.join(app.config['INITIAL_FILE_UPLOADS'], name))

		image = Image.open(image_upload)

		# Resize the image to 48x48 pixels
		image = image.resize((48, 48))

		# Convert the image to grayscale (L mode)
		image = image.convert('L')

		# Convert the grayscale image to a NumPy array
		img_array = np.array(image)

		# Ensure the array has the desired shape (48, 48)
		img = img_array.reshape(48, 48)

		# labels
		label_dict = {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}

		# load the model
		model = tf.keras.models.load_model('app/static/model_sixty_epochs')

		# predict
		img = np.expand_dims(img,axis = 0) 
		img = img.reshape(1,48,48,1) 
		result = model.predict(img)
		result = list(result[0])

		# show prediction
		img_index = result.index(max(result))
		emotion_prediction = label_dict[img_index]

		# Returning template, filename, extracted text
		result = str(f"The emotion prediction is {emotion_prediction}.")
		return render_template('index.html', full_filename = full_filename, pred = result)

# Main function
if __name__ == '__main__':
    app.run(debug=True)
