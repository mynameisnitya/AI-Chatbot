import tensorflow as tf
import cv2

def predict_car_in_video(video_path,model):
    # Load the SavedModel
    model = tf.keras.models.load_model(model)
    print("Model loaded successfully\nPreparing to scan video")
   
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Create a list to store the predictions
    predictions = []

    # Initialize a counter for the number of frames in which a car is detected
    car_count = 0
    all_count = 0
    

    # Process each frame in the video
    while cap.isOpened():
        # Read the next frame from the video
        ret, frame = cap.read()

        # If there are no more frames, break out of the loop
        if not ret:
            break

        # Preprocess the frame
        frame = cv2.resize(frame, (32, 32))
        frame = frame / 255.0
        frame = tf.expand_dims(frame, axis=0)

        # Use the model to make a prediction
        prediction = model.predict(frame)

        # Process the prediction as needed
        if prediction >= 0.5:
            print('Car detected')
            predictions.append('Car detected')
            car_count += 1
            all_count += 1
        else:
            all_count += 1
            print('No Car detected')
            predictions.append('No car detected')

    # Release the video file and clean up
    cap.release()
    cv2.destroyAllWindows()
    if(all_count/4<car_count):
        print(f"{car_count} frames with a car detected in the video.")
    else:
        print("No cars detected in the video.")
    return predictions
#uncoment this to test
#predict_car_in_video('Alpine.mp4','car_classifier.h5')
