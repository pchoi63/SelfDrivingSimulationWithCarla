# Self Driving Simulation With Carla
## Motivation
As many automobile companies have released self-driving vehicles in the market, the research for autonomous driving has grown exponentially grown over the decade. Even though those self-driving cars already have a big impact on our lives, there are areas where the vehicles struggle, in other words, do not perform well which can lead to a serious issue since it is responsible for human life.

![maxresdefault](https://user-images.githubusercontent.com/110645615/211204387-46b72f02-48c5-4028-b765-6fd86f8dd464.jpg)

For my senior research at Connecticut College, I wanted to explore the computer vision side of self-driving vehicles using Carla, CAR Learning to Act. Carla is an open-source autonomous driving simulator powered by Unreal Engine, which can be found in high-graphic gaming today. It was developed to support the training and validation of the autonomous driving system. This makes Carla efficient regarding time, cost, and safety by testing algorithms in the simulator rather than testing on a real-life car. This process helps eliminate the errors of a self- driving system which can result in an accident. Carla also provides various environmental conditions such as weather, map, etc., allowing users to explore all the situations a vehicle might face in real life.

## Carla
<img width="1616" alt="Screen Shot 2023-05-31 at 5 53 35 PM" src="https://github.com/pchoi63/SelfDrivingSimulationWithCarla/assets/110645615/47143f37-85ae-44da-8c7e-43ee7eaea3a0">

Carla has features that allow vehicles in simulation to work similarly to real cars. The main features I will use in this research are features such as cameras, control, and sensors. The camera in Carla is free of positioning, meaning multiple cameras can be attached anywhere in the car. For example, a dashcam camera and a rear-view camera can work synchronously. For image processing purposes, my camera was placed on the dashcam of the vehicle. This dashcam perspective provided the best angle to look at objects such as speed signs and traffic lights frame by frame. Control of the vehicle has three functions, throttle, steer, and break, where each function can be given any numeric value to make changes in its control. Lastly, sensors in Carla
provide ground truth, such as collisions with objects or lanes and location. This feature was only used to validate my predictions and not to control the vehicle.

## Goal
<img width="586" alt="Screen Shot 2023-05-31 at 5 54 12 PM" src="https://github.com/pchoi63/SelfDrivingSimulationWithCarla/assets/110645615/4beff964-8124-48d4-89a5-fe78a52b00e8">

I focused on developing a YOLO v5 model using the Deep Neural Network module via Open-cv. YOLO, You Only Look Once, is Ultralytic’s open-source Neural Network-based computer vision model. It can achieve state-of-the-art results for object detection. Even though Lidar is becoming popular in the market as companies like Mercedes and Honda have implemented it in their products, Tesla focuses and relies on computer vision. They believe lidar is inefficient in terms of its cost and performance. This research aims to achieve autonomous driving through simulation, including computer vision, vehicle control, and routing.

## Lane Detection
<img width="612" alt="Screen Shot 2023-05-31 at 5 54 20 PM" src="https://github.com/pchoi63/SelfDrivingSimulationWithCarla/assets/110645615/221cbab1-172f-4262-9482-65b80861d41e">

Lane detection was implemented to keep the vehicle in the center of the road. Hough transform was used to find the lines by looking at the lanes on each side of the road, a feature extraction technique in computer vision. It is a voting procedure to find imperfect instances of objects within a certain class of shapes. Once the two lines, the predicted lane, are detected, it finds the center point of the two, which is the center point of the road. Finally, I compare the vehicle's center point to the center point of the road detected, then steer as needed. The value of steering was calculated based on the angle the vehicle was off by multiplying it by the chosen value.

## Yolo Objection Detection on Carla
When YOLO detects one of the objects in the class list, it can draw a box around it with a name indicator. Based on its speed and light state predictions, the vehicle gives its calculated value of each function, steer, throttle, and break. Since the vehicle is in closed-loop control, meaning it works in real-time operation in simulation time, it can react to sensitive changes from frame to frame.

### Traffic Light Detection
<img width="1340" alt="Screen Shot 2023-05-31 at 6 07 25 PM" src="https://github.com/pchoi63/SelfDrivingSimulationWithCarla/assets/110645615/a35028b7-6837-4804-b2a9-117053b88525">

### Speed Sign Detection
<img width="1340" alt="Screen Shot 2023-05-31 at 6 07 50 PM" src="https://github.com/pchoi63/SelfDrivingSimulationWithCarla/assets/110645615/d0904f7b-dd3d-4945-9ff3-ef89c160f94d">

## Data Set
My data set consists of 1.6k training-labeled images imported from Roboflow. The model was developed using YOLO v5s which had the perfect balance of what I needed regarding speed and accuracy. It was trained with 32 batches and 120 epochs, including nine classes:
### Classes
1. Red light
2. Yellow light
3. Green light,
4. 30 Speed Sign,
5. 60 Speed Sign,
6. 90 Speed Sign,
7. Human
8. Bicycle
9. Vehicle

## Vehicle Control
<img width="978" alt="Screen Shot 2023-05-31 at 5 53 45 PM" src="https://github.com/pchoi63/SelfDrivingSimulationWithCarla/assets/110645615/ec21493a-6069-4cd8-ba83-aababac98f7f">

1. The controller tries to match the dashcam's center point to the lane's center point. It provides a sharper steer if the distance between the two center points exceeds a threshold.
2. The vehicle matches the target speed and provides the needed throttle or null if the current speed exceeds the target speed.
3. When the red light is detected, it can completely stop without violating the law.
4. When the green light is detected, it cruises through or accelerates out of the brake coming
from red to green.

## Result
The accuracy was tested under four conditions. The purpose of testing in different
conditions is to test how well the model will adjust or react to lightning and fog. 158 images were tested for each condition as Point A to Point B in Town 02 provided the same number of frames. This allowed me to test with consistent images while monitoring the predictions of each.
Accuracy includes: 
Clear Sunset
• 132 / 158 (83.5%) 

Clear Evening
• 130 / 158 (82.2%) 

Foggy Sunset
• 128 / 158 (81%) 

Foggy Evening
• 124 / 158 (78.5%)


## Future work
1. Making decisions when multiple objects are detected.
2. Routing system for interactions with no lanes
3. Evaluate how well comfort, travel time, and safety.
4. Implement multiple cameras to create a 3D Map of my surroundings.


## Reference
“Carla Settings.” CARLA Settings - CARLA Simulator, https://carla.readthedocs.io/en/stable/carla_settings/.
Am. “120 Dog Breeds - Classification.” Kaggle, 20 Apr. 2022, https://www.kaggle.com/datasets/66c1d9bf5dc19d7b625c8dc2ab926fdfba7a66b7ccaac60c f70f8fa480f086ae.

“3.3. Scikit-Image: Image Processing¶.” 3.3. Scikit-Image: Image Processing - Scipy Lecture Notes, http://scipy-lectures.org/packages/scikit-image/.

“1.4. Support Vector Machines.” Scikit, https://scikit-learn.org/stable/modules/svm.html. “Sklearn.model_selection.GRIDSEARCHCV.” Scikit, https://scikit-
learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV. 

“Generate Maps with OpenStreetMap.” Generate Maps with OpenStreetMap - CARLA
Simulator, https://carla.readthedocs.io/en/latest/tuto_G_openstreetmap/.

“[CAVH] Carla Simulation Project.” CAVH Research Group, https://cavh.cee.wisc.edu/carla-
simulation-project/.

“Carla Object Detection Dataset by Alec.Hantson@student.Howest.Be.” Roboflow, universe.roboflow.com/alec-hantson-student-howest-be/carla-izloa. Accessed 14 May 2023.

Ultralytics. “Ultralytics/Yolov5: Yolov5 🚀 in PyTorch > ONNX > CoreML > TFLite.” GitHub, github.com/ultralytics/yolov5. Accessed 14 May 2023.
