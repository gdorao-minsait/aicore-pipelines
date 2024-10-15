# next-level-case1
Bottle detection for a return optimizer.

## Pipelines
Storing AI Pipelines for SAP AI Core.

## Serving
Everything needed to create a Docker image with a working CV model.

### Environment
The script runs on a container based on the Docker image **python:3.12-slim** available in DockerHub.  
To create the image, just run 'docker compose -f "serving\docker-compose.yml" up -d --build' or 'docker compose up -d --build' if you are already at the folder serving/.  
Using the docker compose is highly recommendable, but if the container is run without it there are some important steps to follow:
- Without the command ["/bin/sh", "-c", "python /app/src/main.py"], the container will just close immediately. You must add an entrypoint or keep it awake and call main.py manually.
- The port 9001 must be forwarded
- (optional) Map the volumnes to /mnt/models and /runs to change the CV models more easily and check the results of the detections.

### main.py
main.py is the main script and it is composed of the following functions:

#### correct_image_orientation
When taking photos from a smartphone, they might appear flipped depending on the position of the phone when the image was taken. This function takes the metadata of the image into consideration to check if the file was passed in the correct orientation and fix it if necessary.

#### is_enclosed and is_almost_enclosed
This checks if a Bounding Box is inside another. This can happen because in the dataset some bottles were only partially visible. The model sometimes recognizes the top of a bottle as a full bottle and the full bottle as another independent bottle. This function removes these false positives.

#### calculate_area and calculate_intersection
Aux functions for is_almost_enclosed. Calculates area of BB and intersection between 2 BBs

#### init
Loads the model and warms it up with a dummy image.

#### status
Called by the endpoint **/v1/check**. Checks if the model is loaded and returns the current version of the script.

#### predict
Called by the endpoint **/v1/detect**. Receives a file or an image in base64 and passes it to the CV model. Counts the number of BBs of each detected class and returns this result. If **confidence_threshold** was passed as an argument, it will not count BBs which confidences were below this value.



## Training
Everything needed to train a CV model locally (with CPU or CUDA).  
Just run train.py

### Paths
- **DATA_PATH**: Path where the dataset is stored (images and labels)
- **MODEL_PATH**: Path to save the trained model

### Parameters
Can be set as environment variable or hardcoded in train.py
- **EPOCHS**: Number of training epochs (default: 10)
- **BATCH_SIZE**: Batch size for training (default: 16)
- **IMGSZ**: Image size for training (default: 640)
- **WORKERS**: Number of worker threads for data loading (default: 4)
- **PRETRAINED_MODEL**: Name of pretrained model (default: yolo11s.pt)
