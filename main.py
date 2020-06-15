import torch
import torchvision
from torch import optim
from torchvision import models
from torchvision.transforms import transforms
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from datetime import datetime

PATH = './data'


def load_both_train_test(data_dir, valid_size=0.2):
    train_transform = transforms.Compose([transforms.Resize(298), transforms.ToTensor(), ])
    test_transform = transforms.Compose([transforms.Resize(298), transforms.ToTensor()])
    train_data = torchvision.datasets.ImageFolder(data_dir, transform=train_transform)
    test_data = torchvision.datasets.ImageFolder(data_dir, transform=test_transform)
    num_train = len(train_data)
    print(num_train)
    index = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(index)
    train_inx = index[split:]
    test_inx = index[:split]
    train_sampler = SubsetRandomSampler(train_inx)
    test_sampler = SubsetRandomSampler(test_inx)
    train_loader = torch.utils.data.DataLoader(dataset=train_data, sampler=train_sampler, batch_size=64)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, sampler=test_sampler, batch_size=64)
    return train_loader, test_loader


if __name__ == '__main__':
    device = torch.device('cpu')
    model = models.resnet50(pretrained=True)
    for p in model.parameters():
        p.requires_grad = False

    model.fc = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, 10), nn.LogSoftmax(dim=1))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
    model.to(device)

    epochs = 1
    step = 0
    running_loss = 0
    print_every = 10
    train_losses, test_losses = [], []

    train_loader, test_loader = load_both_train_test(PATH)
    print("length are ", len(train_loader), len(test_loader))
    for e in range(epochs):
        for inputs, labels in train_loader:
            step += 1
            # inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print('step ===', step)
            if step % print_every == 9:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                train_losses.append(running_loss / len(train_loader))
                test_losses.append(test_loss / len(test_loader))
                print(f"Epoch {e + 1}/{epochs}.. "
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"Test loss: {test_loss / len(test_loader):.3f}.. "
                      f"Test accuracy: {accuracy / len(test_loader):.3f}")
                running_loss = 0
                model.train()

    time = datetime.now().__str__().replace(' ', '_')
    torch.save(model, './aerialmodel' + time + '.pth')
    # torch.save({'state_dict': model.state_dict()}, './aerialmodel' + time + '.pth')
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Validation loss')
    plt.legend(frameon=False)
    plt.show()
    print("all done...")
