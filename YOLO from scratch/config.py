import os

# OUTPUT_DIR = "models/"
# os.makedirs(OUTPUT_DIR, exist_ok =True)

# MODEL_PATH = "models/model.pth"
# ANNOTATION_DIR = "/content/dataset/VOCdevkit/VOC2012/Annotations"
# IMAGE_DIR = "/content/dataset/VOCdevkit/VOC2012/JPEGImages"
# DATASETPATH = "/content/dataset"

# BATCH_SIZE = 16 # TODO CHANGE LATER
# SEED = 42
cellSize = (7, 7)
# object_threshold = 0.5 # the threshold of object exist in a cell
INPUT_SHAPE = (1, 3, 448, 448)
OUTPUT_SHAPE = (1, 70, 70)
# TRAIN_RATIO = 0.8
lambda_obj=5
lambda_noobj=0.5
# lr = 3e-4
# momentum = 0.9
# weight_decay = 0.0005
# EPOCHS = 3
# EARLY_STOPPING_EPOCH = 3 # the number of epochs to wait for improvement in the model
