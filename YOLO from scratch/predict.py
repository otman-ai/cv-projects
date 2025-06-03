
import torch
from model import YOLONetwork
from utils import inference
from torchvision import transforms
import argparse


if __name__=="__main__":
  parser = argparse.ArgumentParser(prog="Inference YOLOv1")
  parser.add_argument("--model_path", help="Path where the model is located", default="models/models.pth")
  parser.add_argument("--input", help="Input path", default="/content/assets/2007_000027.jpg")

  # setup the deivce
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Auto-detect GPU
  print("Device:", device)
  cuda_count = torch. cuda. device_count()
  print(f"We have {cuda_count} cuda devices")

  # get the arguments
  args = parser.parse_args()

  # Setup and load the model
  network = YOLONetwork().to(device)
  network.load_state_dict(torch.load(args.model_path, weights_only=True))
  network.eval()

  # transforms 
  transform = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor()])
  
  # make the inference
  prediction = inference(network,args.input, device, transform)
  print(prediction)
