import torch
import torchvision.transforms as transforms
from PIL import Image

class ID_Classificaiton:
    def __init__(self, model_path, height = 256, width = 256):
            
        self.HEIGHT = height
        self.WIDTH = width
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(model_path, map_location=self.device)
        

    def transform(self, image):
        rgb_loader = transforms.Compose([
            transforms.Resize((self.HEIGHT, self.WIDTH)),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float)
        ])
        

        return rgb_loader(image).to(device=self.device)
        
    def predict(self, image):
        

        self.model.to(device = self.device)
        self.model.eval()
        with torch.no_grad():
            pred_rgb = self.model(image.unsqueeze(0))
            
        return {'cs': pred_rgb.cpu().data.numpy()[0][0]}
    
    def output(self, image):
        
        results = self.predict(self.transform(Image.fromarray(image)))
        return results['cs']