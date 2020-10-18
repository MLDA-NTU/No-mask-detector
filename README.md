No-mask detector is an AI app that can detect people not wearing a mask

Please downaload the dataset here: https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset

Create a folder -> Dataset folder under No-mask-detector, following is the project structure:
- No-mask-detector
  - Dataset
    - Train
    - Test
    - Validation
  - dataAugmentation.py
  - README.md
  - .gitignore

Data Augmentation part: (Done the basic -> needs more improvements)

In order to run the dataAugmentation file you have to install the following packages(windows):
- pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
- pip install matplotlib
- pip install numpy
- pip install albumentations

