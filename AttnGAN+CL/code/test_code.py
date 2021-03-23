

# file_name = 'text/180.Wilson_Warbler/Wilson_Warbler_0007_175618.txt'
# with open(file_name, "r") as f:
#     print('success')

import torchvision.models as models
import torch.utils.model_zoo as model_zoo


model = models.inception_v3()

print('model')

url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
model.load_state_dict(model_zoo.load_url(url))
for param in model.parameters():
    param.requires_grad = False
print('Load pretrained model from ', url)