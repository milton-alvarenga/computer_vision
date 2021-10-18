#Load the image library OpenCV
import cv2

# Load the Cascade Classifier - Open Source
car_cascade = cv2.CascadeClassifier("car_classifier.xml")

videos_to_analyze = ['clean_traffic_good_camera_position.mp4','urban_traffic_same_level_camera_cars.mp4']

for video in videos_to_analyze:

    #Load the video
    cap = cv2.VideoCapture(video)

    print(f"{video} loaded")

    while True:
        
        #read image from video
        respose, color_img = cap.read()
        
        #Fail to load the video
        if respose == False:
            print(f"{video} finished")
            break
        
        # Convert to grayscale
        gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        
        # Detect the cars
        faces = car_cascade.detectMultiScale(gray_img, 1.1, 1)
        
        #display rectangle
        for (x, y, w, h) in faces:
            cv2.rectangle(color_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
            # display image
            cv2.imshow('img', color_img)
                
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the VideoCapture object
    cap.release()
    cv2.destroyAllWindows()