%matplotlib inline
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNetGenerator, self).__init__()
        self.down1 = self.conv_block(in_channels, 64, batch_norm=False)
        self.down2 = self.conv_block(64, 128)
        self.down3 = self.conv_block(128, 256)
        self.bottleneck = self.conv_block(256, 512)
        self.up1 = self.upconv_block(512, 256)
        self.up2 = self.upconv_block(512, 128)
        self.up3 = self.upconv_block(256, 64)
        self.final = nn.Sequential(nn.ConvTranspose2d(128, out_channels, 4, 2, 1), nn.Tanh())
    
    def conv_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not batch_norm)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)
    
    def upconv_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        b = self.bottleneck(d3)
        u1 = self.up1(b)
        u2 = self.up2(torch.cat([u1, d3], dim=1))
        u3 = self.up3(torch.cat([u2, d2], dim=1))
        return self.final(torch.cat([u3, d1], dim=1))

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=6):
        super(PatchGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, 1, 1), nn.Sigmoid()
        )
    
    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))

class PairedImageDataset(Dataset):
    def __init__(self, root_dir="./data/facades/train"):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3)
        ])
        if os.path.exists(root_dir):
            self.images = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.jpg')]
        else:
            print(f"Directory {root_dir} not found. Using synthetic data.")
            self.images = ["synthetic"] * 8
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if "synthetic" in self.images[idx]:
            return torch.rand(3, 256, 256), torch.rand(3, 256, 256)
        img = Image.open(self.images[idx]).convert('RGB')
        w, h = img.size
        input_img = self.transform(img.crop((0, 0, w//2, h)))
        target_img = self.transform(img.crop((w//2, 0, w, h)))
        return input_img, target_img

def train_pix2pix(generator, discriminator, dataloader, epochs=1, device='cpu'):
    criterion_gan, criterion_l1 = nn.BCELoss(), nn.L1Loss()
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    generator, discriminator = generator.to(device), discriminator.to(device)
    
    for epoch in range(epochs):
        for i, (input_imgs, target_imgs) in enumerate(dataloader):
            input_imgs, target_imgs = input_imgs.to(device), target_imgs.to(device)
            optimizer_d.zero_grad()
            real_loss = criterion_gan(discriminator(input_imgs, target_imgs), torch.ones_like(discriminator(input_imgs, target_imgs)))
            fake_imgs = generator(input_imgs)
            fake_loss = criterion_gan(discriminator(input_imgs, fake_imgs.detach()), torch.zeros_like(discriminator(input_imgs, fake_imgs)))
            loss_d = (real_loss + fake_loss) * 0.5
            loss_d.backward()
            optimizer_d.step()
            optimizer_g.zero_grad()
            loss_g_gan = criterion_gan(discriminator(input_imgs, fake_imgs), torch.ones_like(discriminator(input_imgs, fake_imgs)))
            loss_g_l1 = criterion_l1(fake_imgs, target_imgs) * 100
            loss_g = loss_g_gan + loss_g_l1
            loss_g.backward()
            optimizer_g.step()
            if i % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} Batch {i}/{len(dataloader)} Loss D: {loss_d:.4f} Loss G: {loss_g:.4f}")

def visualize_results(generator, input_img, target_img, device='cpu'):
    generator.eval()
    with torch.no_grad():
        generated_img = generator(input_img.unsqueeze(0).to(device)).squeeze(0).cpu()
    images = [(input_img * 0.5 + 0.5).permute(1, 2, 0).numpy(),
              (generated_img * 0.5 + 0.5).permute(1, 2, 0).numpy(),
              (target_img * 0.5 + 0.5).permute(1, 2, 0).numpy()]
    titles = ["Input Image", "Generated Image", "Target Image"]
    plt.figure(figsize=(12, 4))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 3, i+1)
        plt.title(title)
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = PairedImageDataset()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    generator, discriminator = UNetGenerator(), PatchGANDiscriminator()
    train_pix2pix(generator, discriminator, dataloader, epochs=1, device=device)
    input_img, target_img = next(iter(dataloader))
    visualize_results(generator, input_img[0], target_img[0], device)
except Exception as e:
    print(f"Error: {e}")
    
