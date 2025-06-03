# YOLOv1 from scratch (Work on progress)
This is the implementation of this [paper](https://arxiv.org/abs/1506.02640) which I summeries [here](https://medium.com/@otmanheddouchai/summary-of-you-only-look-once-unified-real-time-object-detection-yolov1-70fb0fafaea1?source=user_profile_page---------0-------------e7d361ca183e----------------------) 

## TODO 
- [ ] Make the model more accurate
- [ ] Deploy the model on Flask
- [ ] Dockerize the inference

## Installation

`pip install -r requirements.txt`


## Dataset

This repo is using [VOC dataset](https://paperswithcode.com/dataset/pascal-voc) to train the model but you are welcome to use your own

## Training

```
python train.py --epoch 3 --output_dir [OUTPUT_DIR] --model_name [MODEL_NAME] --batch_size [BATCH_SIZE] --annotation_path [ANNOTATION_PATH] --lr [LEARNING_RATE] --images_path [IMAGES_PATH] --seed [SEED] --early_stopping [EARLY_STOPPING] --dataset_path [DATASET_PATH] --train_ratio [TRAIN_RATIO] --momentum [MOMENTUM] --wieght_decay [WIEGHT_DECAY]
```


For help type:

```
python train.py -h
```

## Inference 
```
python predict.py --input [INPUT_PATH] --model_path [CHECKPOINT]
```


