import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import clip
from utils import *
def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class ConceptAutoencoder(nn.Module):
    def __init__(self, num_concepts,channel):
        super(ConceptAutoencoder, self).__init__()
        # self.encoder = nn.Sequential(
        #     nn.Linear(28 * 28, 128), nn.ReLU(True),
        #     nn.Linear(128, 64), nn.ReLU(True),
        #     nn.Linear(64, 12), nn.ReLU(True),
        #     nn.Linear(12, num_concepts)
        # )
        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.fc1 = nn.Linear(20 * 20 * 20, 16)
        self.fc2 = nn.Linear(16, num_concepts)
        self.relu = nn.ReLU(inplace=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model = CLIP(device, model='ViT-B/32')
        if channel==1:
            self.decoder = nn.Sequential(
                nn.Linear(num_concepts, 16), nn.ReLU(True),
                nn.Linear(16, 64), nn.ReLU(True),
                nn.Linear(64, 128), nn.ReLU(True),
                nn.Linear(128, 28 * 28),
                nn.Tanh()
            )
        elif channel==3:
            self.decoder = nn.Sequential(
                nn.Linear(num_concepts, 16), nn.ReLU(True),
                nn.Linear(16, 64), nn.ReLU(True),
                nn.Linear(64, 128), nn.ReLU(True),
                nn.Linear(128, 3 * 28 * 28),
                nn.Tanh()
            )
        self.transform_grayscale_to_rgb = transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x)
        self.transform_resize = transforms.Resize((224, 224))
        self.transform_pipeline = transforms.Compose([
            self.transform_grayscale_to_rgb,
            self.transform_resize
        ])
        self.clip_to_encoder_dim = nn.Linear(512, num_concepts)


    def forward(self, x):
        x_resized_rgb = torch.stack([self.transform_pipeline(img) for img in x])
        clip_feature = self.clip_model.image_embeds(x_resized_rgb).float().detach()
        cpt = self.relu(self.conv1(x))
        cpt = self.relu(self.conv2(cpt))
        cpt = cpt.view(-1, num_flat_features(cpt))
        cpt = self.relu(self.fc1(cpt))
        encoder = self.fc2(cpt)+self.clip_to_encoder_dim(clip_feature)
        # encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return encoder, decoder


## RelevanceParametrizer == LeNet
class RelevanceParametrizer(nn.Module):
    def __init__(self, num_concepts,channel):
        super(RelevanceParametrizer, self).__init__()
        self.conv1 = nn.Conv2d(channel, 6, (5,5), padding=2)
        self.conv2 = nn.Conv2d(6, 16, (5,5))
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_concepts)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.maxpool(self.relu(self.conv1(x)))
        out = self.maxpool(self.relu(self.conv2(out)))
        out = out.view(-1, num_flat_features(out))
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        return out


class Aggregator(nn.Module):
    def __init__(self,class_num):
        super(Aggregator, self).__init__()
        self.fc1 = nn.Linear(5, 10)
        self.fc2 = nn.Linear(10, class_num)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, concept, relevance):
        out = concept + relevance
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class SENN(nn.Module):
    def __init__(self, num_concepts=5,channel=1,class_num=10):
        super(SENN, self).__init__()
        self.concept_autoencoder = ConceptAutoencoder(num_concepts,channel)
        self.relevance_parametrizer = RelevanceParametrizer(num_concepts,channel)
        self.aggregator = Aggregator(class_num)

    def forward(self, x):
        ## concept encoder
        # concept_encoder, concept_decoder = self.concept_autoencoder(x.view(x.size(0), -1))
        concept_encoder, concept_decoder = self.concept_autoencoder(x)
        ## relevance parametrizer
        relevance = self.relevance_parametrizer(x)

        ## aggregator
        out = self.aggregator(concept_encoder, relevance)

        return concept_encoder, concept_decoder, relevance, out
    def explainability(self, x,targets,criterion):
        x.requires_grad = True
        # Forward propagation, obtaining classification results and attention output
        concept_encoder, concept_decoder = self.concept_autoencoder(x)
        ## relevance parametrizer
        relevance = self.relevance_parametrizer(x)

        ## aggregator
        output = self.aggregator(concept_encoder, relevance)

        # Calculate loss function
        loss = criterion(output, targets)
        loss.backward()

        # Obtain the gradients
        gradients = x.grad

        return gradients

if __name__ == '__main__':
    model = SENN(num_concepts=5)
    inp = torch.rand((2,1,28,28))
    h ,h_hat, theta, g = model(inp)
    print(h_hat.shape)
