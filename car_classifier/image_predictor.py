import numpy as np
from keras.models import load_model
from PIL import Image


import sys
#function for for predicting if image is a car or not
def predict_image(img_path,model):
    # Load the image to be classified and get input
    if img_path == 'quit' or img_path == 'exit':
        sys.exit()
    else:
        # Load the saved model
        model = load_model(model)
        img = Image.open(img_path)
        #resize image to right size
        img = img.resize((32, 32))
        x = np.array(img)
        x = np.expand_dims(x, axis=0)
        # Make a prediction
        prediction = model.predict(x)
        # Get the predicted class label and probability
        class_names = ['a car', 'not a car']
        predicted_class = np.argmax(prediction)
        pred_class_name=class_names[predicted_class]
        class_prob = prediction[0][predicted_class]
        # Print the class label and probability
        if class_prob > 0.5:
            print('This image is',pred_class_name , 'with a probability of {:.0%}'.format(class_prob))
        elif  class_prob > 0:
            print('This image is',pred_class_name , 'with a probability of {:.0%}'.format(class_prob))
        else:
            print('This image is not a car')
        if img_path == 'quit' or img_path == 'exit':
            exit

#to test uncomment this 
#predict_image('E:/Users/bavique_3/Pictures/2020-toyota-corolla-sedan.jpg','car_classifier.h5')
    
    
