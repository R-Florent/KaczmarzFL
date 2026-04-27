import numpy as np
import matplotlib.pyplot as plt
import warnings

import torch
import torchvision
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler


# Supprime les warnings NumPy
warnings.filterwarnings("ignore", category=np.exceptions.VisibleDeprecationWarning)

#%% md
# Settings and Dataset
#%%
# --- hyperparamètres ---
RANDOM_SEED = 123
BATCH_SIZE = 256
NUM_EPOCHS = 200  # Corrigé: NUM_EPOCHES -> NUM_EPOCHS
NUM_CLASSES = 10  # Corrigé: CIFAR-10 a 10 classes, pas 3
LEARNING_RATE = 0.0005  # Corrigé: LEARING_RATE -> LEARNING_RATE
SCALE = 0.1
IMG_SIZE = (64, 64)  # Ajusté pour CIFAR-10
LATENT_DIM = 10

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

#%%
##########################
### CIFAR 10 - VERSION UNIQUE ET OPTIMISÉE
##########################

# Transform avec augmentation (MEILLEURE MÉTHODE)
train_transforms = transforms.Compose([
    transforms.Resize((70, 70)),
    transforms.RandomCrop((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transforms = transforms.Compose([
    transforms.Resize((70, 70)),
    transforms.CenterCrop((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load datasets
trainset = datasets.CIFAR10(root='./data',
                            train=True,
                            download=True,
                            transform=train_transforms)

testset = datasets.CIFAR10(root='./data',
                           train=False,
                           download=True,
                           transform=test_transforms)

# Create DataLoaders
trainloader = DataLoader(trainset,
                         batch_size=BATCH_SIZE,
                         shuffle=True,
                         num_workers=0)  # 0 pour éviter les crashes Jupyter

testloader = DataLoader(testset,
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        num_workers=0)

# Class labels
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Vérification du dataset
print("Dataset chargé avec succès!")
for images, labels in trainloader:
    print(f'Image batch dimensions: {images.shape}')
    print(f'Image label dimensions: {labels.shape}')
    print(f'Class labels of 10 examples: {labels[:10]}')
    break

#%%
# Fonction pour afficher une image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()  # Ajouté .cpu() pour gérer GPU
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
print("Images affichées:")
imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(min(4, len(labels)))))

#%% md
# Model Creation
#%%

class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()  # Ajouté pour bonne pratique
        self.features = nn.Sequential(
            # Feature extraction bloc 1
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Bloc 2
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),  # Corrigé kernel et stride
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Bloc 3
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # Bloc 4
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # Bloc 5
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Linear(6 * 6 * 256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),  # Corrigé: output = num_classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        logits = self.classifier(x)
        return logits

#%% 
# Model instantiation
model = AlexNet(num_classes=NUM_CLASSES)
model = model.to(DEVICE)

print(f"Model créé et envoyé sur {DEVICE}")

optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       factor=0.1,
                                                       mode='max',
                                                       verbose=True)

# Note: Les fonctions train_model() et plot_*() doivent être définies ailleurs
# ou tu dois les implémenter. Voici un exemple minimaliste pour tester:

print("Model prêt pour l'entraînement!")