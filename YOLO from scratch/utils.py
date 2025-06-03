import xml.etree.ElementTree as ET
from PIL import Image
import torch
from tqdm import tqdm
import numpy as np
def read_content(xml_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()
    classes = []
    list_with_all_boxes = []

    for boxes in root.iter('object'):
        filename = root.find('filename').text
        name = boxes.find('name').text


        ymin, xmin, ymax, xmax = None, None, None, None

        ymin = float(boxes.find("bndbox/ymin").text)
        xmin = float(boxes.find("bndbox/xmin").text)
        ymax = float(boxes.find("bndbox/ymax").text)
        xmax = float(boxes.find("bndbox/xmax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax, name]
        classes.append(name)
        list_with_all_boxes.append(list_with_single_boxes)

    return list_with_all_boxes, classes

def read_image(image_path):
  image = Image.open(image_path)
  return image

def fun_iou(box1, box2):
  x1, y1, x2, y2, = box1[:4]
  x3, y3, x4, y4 = box2[:4]

  x_inter1 = max(x1, x3)
  y_inter1 = max(y1, y3)

  x_inter2 = min(x2, x4)
  y_inter2 = min(y2, y4)

  width_inter = abs(x_inter2 - x_inter1)
  height_inter = abs(y_inter2 - y_inter1)
  area_inter = width_inter * height_inter

  width_box1 = abs(x2 - x1)
  height_box1 = abs(y2 - y1)
  width_box2 = abs(x4 - x3)
  height_box2 = abs(y4 - y3)

  area_box1 = width_box1 * height_box1
  area_box2 = width_box2 * height_box2

  area_union = area_box1 + area_box2 - area_inter

  iou = area_inter / area_union
  return iou
def non_max_suppression(bounding_boxes, conf_threshold=0.5, iou_threshold=0.4):

  """
  apply non max suppression
  bounding_boxes: bounding boxes of one class
  """
  box_threshold = []
  final_boxes = []
  sorted_boxes = sorted(bounding_boxes, reverse=True, key=lambda x: x[5]) # sort the boxes in descending orders
  for sorted_box in sorted_boxes: # loop through the boxes and remove the box with lowest score
    if sorted_box[-1] < conf_threshold:
      continue
    box_threshold.append(sorted_box)
  # calculate iou
  while len(box_threshold) > 0:
    current_box = box_threshold.pop(0)
    final_boxes.append(current_box)
    for box in box_threshold:
      iou = fun_iou(current_box, box)
      if iou > iou_threshold:
        box_threshold.remove(box)
  return final_boxes

def get_scaled_box(box, image_shape):
  x, y, w, h, = box
  w_original, h_original = image_shape
  x_center = x * w_original
  y_center = y * h_original
  w = w * w_original
  h = h * h_original
  return [x_center, y_center, w, h]

def extract_boxes(y_prediction, conf_threshold=0.5, num_classes=20, input_shape=(448, 448)):
  bounding_boxes = {}
  y_prediction = y_prediction.cpu().detach().numpy()
  for cl_idx in range(num_classes):
    bounding_boxes[cl_idx] = []
  for i in range(y_prediction.shape[0]):
    for j in range(y_prediction.shape[1]):
      # get the data of one cell
      grid_cell = y_prediction[i, j]
      # get the responsible box by getting higher iou
      # loop through the bounding boxes
      for b in range(2):
        # check if the confidence score is smaller then the threshold
        confidence_score = grid_cell[4 + b * 5] # index 4 then 9
        # compute class-specific confidence scores for each box and get the class probabilities
        x, y, w, h = get_scaled_box(grid_cell[5 * b:5 * b + 4], input_shape)
        # scaled the bounding box to the original
        for cl_idx, c in enumerate(grid_cell[10:]):
          # calculate the confident score
          final_confidente_score = confidence_score * c
          # check if it is great then the threshold
          if final_confidente_score < conf_threshold:
            continue
          bounding_boxes[cl_idx].append([x, y, w, h, cl_idx, final_confidente_score])
  return bounding_boxes

def clean_prediction(predictions):
  # extract boxes to each class
  prediction = []
  extracted_boxes = extract_boxes(predictions)
  # apply non max suppression
  for cl_idx, boxes in extracted_boxes.items():
    if len(boxes) == 0:
      continue
    prediction.extend(non_max_suppression(boxes))
  return prediction

  # Sum square error loss function
def sum_square_error_loss(pred_outputs, true_outputs, cellSize=(7, 7), object_threshold=0.5, lambda_obj=5, lambda_noobj=0.5):
  total_loss = 0
  for min_batch_pred, min_batch_true in zip(pred_outputs, true_outputs):
    coordinates_loss = torch.tensor(0.0, device=pred_outputs.device)
    size_loss = torch.tensor(0.0, device=pred_outputs.device)
    class_loss = torch.tensor(0.0, device=pred_outputs.device)
    noobj_class_loss = torch.tensor(0.0, device=pred_outputs.device)
    probability_loss = torch.tensor(0.0, device=pred_outputs.device)
    for i in range(cellSize[0]): # cell size x
      for j in range(cellSize[1]): # cell size y

          if min_batch_true[i][j][4] == 0 or min_batch_true[i][j][8] == 0: # check if there is object in the cell
            noobj_class_loss += torch.square(min_batch_true[i][j][4] - min_batch_pred[i][j][4])
            noobj_class_loss += torch.square(min_batch_true[i][j][9] - min_batch_pred[i][j][9])

          else:
            for k in range(20): # number of classes
              probability_loss += torch.square(min_batch_pred[i][j][10+k]*min_batch_pred[i][j][4] - min_batch_true[i][j][10+k])

            # get the responsible box by getting higher iou
            iou_1 = fun_iou(min_batch_pred[i][j][0:4], min_batch_true[i][j][0:4])
            iou_2 = fun_iou(min_batch_pred[i][j][5:9], min_batch_true[i][j][5:9])
            b = 0 if iou_1 > iou_2 else 1
            x_pred, y_pred, w_pred, h_pred, confidence_score_pred = min_batch_pred[i][j][5 * b:5 * b + 5]
            x_true, y_true, w_true, h_true, confidence_score_true = min_batch_true[i][j][5 * b:5 * b + 5]

            coordinates_loss += torch.square(x_pred - x_true)  + torch.square(y_pred - y_true)
            size_loss += torch.square(torch.sqrt(w_pred) - torch.sqrt(w_true))  + torch.square(torch.sqrt(h_pred) - torch.sqrt(h_true))
            class_loss += torch.square(confidence_score_pred - confidence_score_true)

    coordinates_loss = coordinates_loss * lambda_obj
    size_loss = size_loss * lambda_obj
    noobj_class_loss = noobj_class_loss * lambda_noobj
    loss = coordinates_loss + size_loss + noobj_class_loss + probability_loss + class_loss
    total_loss += loss
  return total_loss /len(pred_outputs)

def train_one_epoch(model, dataloader, loss_fn, optimizer, device, lambda_obj=5, lambda_noobj=0.5):
    model.to(device)
    model.train()
    total_loss = 0
    progress = tqdm(dataloader, desc="Training Epoch")
    for data, labels in progress:
        optimizer.zero_grad()
        data = data.to(device).float()
        labels = labels.to(device).float()

        with torch.autocast(device_type=str(device)):
            outputs = model(data)
            # loss = loss_fn(outputs, labels)
            loss = loss_fn(outputs, labels, lambda_obj=lambda_obj, lambda_noobj=lambda_noobj)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=f'{loss.item():3f}')

    return total_loss / len(dataloader)

def val(model, loss_fn, dataloader, device, lambda_obj=5, lambda_noobj=0.5):
    model.eval()
    total_loss = 0
    progress = tqdm(dataloader, desc="Validation")
    i = 0 # index to tracking storing the validation output of the first output
    first_output = None
    first_label = None
    with torch.no_grad():
        for data, labels in progress:
            data = data.to(device).float()
            labels = labels.to(device).float()

            outputs = model(data)
            # loss = loss_fn(outputs, labels)
            loss = loss_fn(outputs, labels, lambda_obj=lambda_obj, lambda_noobj=lambda_noobj)

            total_loss += loss.item()
            progress.set_postfix(loss=f'{loss.item():3f}')
    return total_loss / len(dataloader)



def inference(model, path, device, transform ):
  print("Runing the inference...")
  model.to(device)
  model.eval()
  image = Image.open(path)
  data = torch.tensor(np.array(transform(image)))
  image =  data.reshape(-1, 3, 448, 448).to(device)
  prediction = model(image).squeeze()
  print(prediction.shape)
  print("Inference run succefully")
  return  prediction
