import torch
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from . import preprocessing
from facenet_pytorch.models.utils.detect_face import extract_face
from PIL import Image

class ExtractedFaceFeaturesExtractor:
    def __init__(self):
        self.facenet_preprocess = transforms.Compose([preprocessing.Whitening()])
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval()

    def extract_features(self, img):
        img = img.resize((299,299), Image.BILINEAR)
        faces = torch.stack([img])
        embeddings = self.facenet(self.facenet_preprocess(faces)).detach().numpy()

        return embeddings

    def __call__(self, img):
        return self.extract_features(img)
