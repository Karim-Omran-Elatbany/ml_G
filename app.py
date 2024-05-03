from flask import Flask, request, render_template, url_for, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

def preprossing(image):
    image=Image.open(image)
    image = image.resize((150, 150))
    image_arr = np.array(image.convert('RGB'))
    image_arr.shape = (1, 150, 150, 3)
    return image_arr

classes = ['Akhenaten', 'Bent pyramid for senefru', 'Colossal Statue of Ramesses II', 'Colossoi of Memnon', 'Goddess Isis with her child', 'Hatshepsut', 'Khafre Pyramid', 'King Thutmose III', 'Mask of Tutankhamun', 'Nefertiti', 'Pyramid_of_Djoser', 'Ramessum', 'Statue of King Zoser', 'Statue of Tutankhamun with Ankhesenamun', 'Temple_of_Isis_in_Philae', 'Temple_of_Kom_Ombo', 'The Great Temple of Ramesses II', 'amenhotep iii and tiye', 'bust of ramesses ii', 'head Statue of Amenhotep iii', 'menkaure pyramid', 'sphinx']
model=load_model("mobilenet_model.h5")

@app.route('/')
def index():

    return render_template('index.html', appName="Intel Image Classification")


@app.route('/predictApi', methods=["POST"])
def api():
    # Get the image from post request
    try:
        if 'fileup' not in request.files:
            return "Please try again. The Image doesn't exist"
        image = request.files.get('fileup')
        image_arr = preprossing(image)
        print("Model predicting ...")
        result = model.predict(image_arr)
        print("Model predicted")
        ind = np.argmax(result)
        prediction = classes[ind]
        print(prediction)
        return jsonify({'prediction': prediction})
    except:
        return jsonify({'Error': 'Error occur'})


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print("run code")
    if request.method == 'POST':
        # Get the image from post request
        print("image loading....")
        image = request.files['fileup']
        print("image loaded....")
        image_arr= preprossing(image)
        print("predicting ...")
        result = model.predict(image_arr)
        print("predicted ...")
        ind = np.argmax(result)
        prediction = classes[ind]

        print(prediction)

        return render_template('index.html', prediction=prediction, image='static/IMG/', appName="Intel Image Classification")
    else:
        return render_template('index.html',appName="Intel Image Classification")


if __name__ == '__main__':
    app.run(debug=True)
