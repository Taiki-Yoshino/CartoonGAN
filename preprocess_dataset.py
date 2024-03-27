from torch.utils.data import Dataset, Sampler
from PIL import ImageFilter
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils

class ShuffledDatasetSampler(Sampler):
    def __init__(self, data_source, seed=42):
        self.data_source = data_source
        self.seed = seed

    def __iter__(self):
        g = np.random.Generator(np.random.PCG64(self.seed))
        
        indices = list(range(len(self.data_source)))
        g.shuffle(indices)
        
        for i in indices:
            yield i

    def __len__(self):
        return len(self.data_source)

from PIL import ImageFilter

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

class SelectiveEdgeSmoothing(object):
    def __init__(self, img_size, radius=2, alpha=0.5):
        self.radius = radius
        self.alpha = alpha
        self.img_size = img_size

    def __call__(self, img):
        img_cv = np.array(img)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        edges = cv2.Canny(img_cv, self.img_size, self.img_size)
        blurred = cv2.GaussianBlur(img_cv, (self.radius, self.radius), 0)
        mask = edges != 0
        img_cv[mask] = cv2.addWeighted(img_cv, 1-self.alpha, blurred, self.alpha, 0)[mask]
        img_smoothed = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_smoothed = Image.fromarray(img_smoothed)
        return img_smoothed
    
    
    
class FiftyOnePyTorchDataset(Dataset):
    def __init__(self, fiftyone_dataset, transform=None):
        self.samples = [sample.filepath for sample in fiftyone_dataset]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image = Image.open(self.samples[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def visualize_batch(batch, title, nrow=8, normalize=True):
    plt.figure(figsize=(16, 30))
    plt.axis("off")
    plt.title(title)
    grid = vutils.make_grid(batch[:16], nrow=nrow, padding=2, normalize=normalize)
    plt.imshow(np.transpose(grid.cpu().numpy(), (1, 2, 0)))
    plt.show()