# Staff_Detection
Combination of object detection and classification models to identify staff in a video.
Steps and flows can be found in the documentation.

# Flows:
## The [Demo.mp4](https://github.com/GanYihWee/Staff_Detection/blob/main/demo.mp4) demostrate the flow of this system.

### Step 1: Self-generate staff images (with name tag) for training using [Human-Aligned Bounding Boxes from Overhead Fisheye cameras dataset (HABBOF)](https://vip.bu.edu/projects/vsns/cossy/datasets/habbof/). 
### The dataset can be found in: [self-generated dataset](https://drive.google.com/file/d/1E0Swr1u6TP0xTS-p3NZ1wfzdCZGlf15k/view?usp=drive_link) Unzip and place it under the dataset folder. 
### The code to generate the name tag dataset can be found in: [image_augmentation.ipynb](https://github.com/GanYihWee/Staff_Detection/blob/main/image_augmentation.ipynb)


<img width="458" alt="combined" src="https://github.com/GanYihWee/Staff_Detection/assets/102400483/c1930487-9292-4dd5-9bf3-fe5c5dabe8c3">


### Step 2: Trained a classification model with the [self-generated dataset](https://drive.google.com/file/d/1E0Swr1u6TP0xTS-p3NZ1wfzdCZGlf15k/view?usp=drive_link) using [pytorch pertrained model](https://drive.google.com/file/d/1cYHbVX6igWY61qPOPv__vduZv8flPD9l/view?usp=drive_link). The code can be found: [train.py](https://github.com/GanYihWee/Staff_Detection/blob/main/train.py)

### Step 3: Use a [pretrained yolov7 model](https://drive.google.com/file/d/1ePMnNw9wbaPAxzMi7ByItbii5MTiH7Da/view?usp=sharing) for the people detection.

### Step 4: Combined the both models (object detection and classification) to capture the staff and their respective bounding boxes. The code can be found: [detect.py](https://github.com/GanYihWee/Staff_Detection/blob/main/detect.py)







