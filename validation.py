import torch
from torch.autograd import Variable
from torchvision.transforms import transforms
from matplotlib import pyplot as plt
import numpy as np
from torchvision import datasets
from torchvision import models

# https://towardsdatascience.com/how-to-train-an-image-classifier-in-pytorch-and-use-it-to-perform-basic-inference-on-single-images-99465a1e9bf5
data_dir = './data'
test_transforms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), ])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = models.resnet50(pretrained=True)
# cp = torch.load('./aerialmodel2020-06-15_12:12:07.187282.pth')
# model = model.load_state_dict(cp['state_dict'])
model = torch.load('./aerialmodel2020-06-15_14:41:48.969373.pth')
model.eval()


def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index


def get_random_images(num):
    data = datasets.ImageFolder(data_dir, transform=test_transforms)
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    idx = indices[:num]
    from torch.utils.data.sampler import SubsetRandomSampler
    sampler = SubsetRandomSampler(idx)
    loader = torch.utils.data.DataLoader(data, sampler=sampler, batch_size=num)
    dataiter = iter(loader)
    images, labels = dataiter.next()
    return images, labels


classes = ['daisy', 'dandelion', 'roses', 'sunfloweres', 'tulips']
to_pil = transforms.ToPILImage()
images, labels = get_random_images(5)
fig = plt.figure(figsize=(10, 10))
for ii in range(len(images)):
    image = to_pil(images[ii])
    index = predict_image(image)
    # index 是预测的结果，label是实际的结果
    sub = fig.add_subplot(1, len(images), ii + 1)
    res = int(labels[ii]) == index
    sub.set_title(str(classes[index]) + ":" + str(res))
    plt.axis('off')
    plt.imshow(image)
plt.show()
