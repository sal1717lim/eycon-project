#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from torch import nn
from torchvision import transforms
import random
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dset
from torchvision.utils import save_image
import os
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
import torch.optim as optim


# # CONFIG

# In[3]:


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#paths and sets for the data
TRAIN_DIR = "../input/eycon/dataset500"
TRAIN_LIST = ["set00" , 'set01' , 'set02' , 'set06' , 'set07']
#TRAIN_LIST = ["set03" , 'set04' , 'set05' , 'set09' , 'set10', "set11"]
#TEST_LIST = ["set11"]
#TEST_LIST = ["set08"]
#the list of models implemented.
MODEL_LIST = ["ResUnet", "Unet"]
#choosing the model to train.
MODEL = MODEL_LIST[0]
#hyper-parameters
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
#the number of images saved by save_some_images
EVAL_BATCH_SIZE = 16
#the ressources allocated to loading the data
NUM_WORKERS = 2
IMAGE_SIZE = 256
#when true, initializes the wieghts with a normal distro with mean 0 and std 0.02 (paper values) if False random Init
INIT_WEIGHTS = True
CHANNELS_IMG = 3
L1_LAMBDA = 100
NUM_EPOCHS = 20
#when true loads the models saved as "disc.pth.tar" and "gen.pth.tar", they need to be in ./
LOAD_MODEL = True
#when true saves a checkpoint every 5 epochs
SAVE_MODEL = True
#the names of the checkpoints
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"


# In[30]:


workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 256

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 16

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1


# # fonction

# In[5]:


def save_some_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(DEVICE), y.to(DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        if epoch == 0:
            save_image(x * 0.5 + 0.5, folder + f"/input.png")
            save_image(y * 0.5 + 0.5, folder + f"/label.png")
    gen.train()

def save_checkpoint(model, optimizer, epoch, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, str(epoch) + filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


# # DATASET

# In[6]:


transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
class EYCON(Dataset):
    def __init__(self, path,train=True, shuffle=True):
        self.path = path
        x=os.listdir(self.path)
        if shuffle:
            for i in range(3):
                random.shuffle(x)
        self.data =x[:int(len(x)*0.8)] if train  else x[int(len(x)*0.8):]
        self.nbdata = len(self.data)
        #if shuffle true, the data will be shuffeled before loading (used only in test data, in the trainign data is shuffeled using the loaded)
        

    def __getitem__(self, index):
        x=torch.randn( 100, 1, 1, device=DEVICE)
        y = Image.open(self.path+"/"+self.data[index]).convert('RGB')
        y = transform(y)
        
        return x, y

    def __len__(self):
        return self.nbdata


# In[7]:


traindataset=EYCON("./eycon/dataset500")
testdataset=EYCON("./eycon/dataset500",train=False)
print("train dataset:",len(traindataset))
print("test  dataset:",len(testdataset))


# # Descriminateur

# In[57]:


def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, mean=0, std=0.02)
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            # input is (nc) x 256 x 256
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            # input is (nc) x 128 x 128
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            # input is (nc) x 64 x 64
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            # input is (nc) x 32 x 32
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            # input is (nc) x 16 x 16
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (nc) x 8 x 8
            nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (nc) x 4 x 4
            #nn.Conv2d(ndf * 32, ndf * 64, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 64),
            #nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 32, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# # Generateur

# In[95]:


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf , 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d( ngf , int(ngf/2), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(ngf/2)),
            nn.ReLU(True),
            # state size. (ngf*2) x 64 x 64
            nn.ConvTranspose2d( int(ngf/2), int(ngf/4), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(ngf/4)),
            nn.ReLU(True),
            # state size. (ngf*2) x 128 x 128
            #nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ngf),
            #nn.ReLU(True),
            # state size. (ngf) x 256 x 256
            nn.ConvTranspose2d( int(ngf/4), nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 256 x 256
        )

    def forward(self, input):
        return self.main(input)


# In[10]:


dataset = EYCON("../input/eycon/dataset500")
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)
# Plot some training images
real_batch = next(iter(dataloader))
print(real_batch[1].shape)
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[1].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))


# In[13]:


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# In[58]:


netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Print the model
print(netD)


# In[96]:


netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.02.
netG.apply(weights_init)

# Print the model
print(netG)


# In[16]:


get_ipython().system('pip install torch-summary')


# In[97]:


from torchsummary import summary

summary(netG,(100,1,1))


# In[98]:


criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


# # train

# In[119]:


img_list = []
G_losses = []
D_losses = []
iters = 0
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from time import localtime
print("Starting Training Loop...")
# For each epoch
writer = SummaryWriter("train{}-{}-{}".format(localtime().tm_mon, localtime().tm_mday, localtime().tm_hour))
print(writer)
for epoch in range(num_epochs):
    # For each batch in the dataloader
    print("epoch:",epoch)
    loop=tqdm(enumerate(dataloader, 0))
    for i, data in loop:

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[1].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 1 == 0:
            loop.set_postfix(
                Loss_G=errG.item(),
                Loss_D=errD.item(),
                D_X=D_x,
                D_G_Z1=D_G_z1,
                D_G_Z2=D_G_z2
            )
        if i % 10 == 0:
            writer.add_scalar("LossG",errG.item(),epoch*(len(dataloader))+i)
            writer.add_scalar("LossD",errD.item(),epoch*(len(dataloader))+i)
            writer.add_scalar("D_X",D_x,epoch*(len(dataloader))+i)
            writer.add_scalar("D_G_Z1",D_G_z1,epoch*(len(dataloader))+i)
            writer.add_scalar("D_G_Z2",D_G_z2,epoch*(len(dataloader))+i)
        

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            fride=vutils.make_grid(fake, padding=2, normalize=True)
            writer.add_image('images'+str(i)+"_"+str(epoch), fride, 0)

        iters += 1


# # pixtopix

# In[9]:


def train_fn(
    disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler, writer, epoch=0
):
    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * L1_LAMBDA
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            writer.add_scalar("L1 train loss",L1.item()/L1_LAMBDA,epoch*(len(loop))+idx)
            writer.add_scalar("D_real train loss", torch.sigmoid(D_real).mean().item(), epoch * (len(loop)) + idx)
            writer.add_scalar("D_fake train loss", torch.sigmoid(D_fake).mean().item(), epoch * (len(loop)) + idx)
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
                L1    =L1.item()
            )


# In[10]:


def test_fn(
    disc, gen, loader, l1_loss, bce, writer, epoch=0
):
    loop = tqdm(loader, leave=True)
    disc.eval()
    gen.eval()
    with torch.no_grad():
     resultat=[]
     for idx, (x, y) in enumerate(loop):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2



        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * L1_LAMBDA
            G_loss = G_fake_loss + L1
            resultat.append(L1.item())



        if idx % 10 == 0:
            writer.add_scalar("L1 test loss",L1.item()/L1_LAMBDA,epoch*(len(loop))+idx)
            writer.add_scalar("D_real test loss", torch.sigmoid(D_real).mean().item(), epoch * (len(loop)) + idx)
            writer.add_scalar("D_fake test loss", torch.sigmoid(D_fake).mean().item(), epoch * (len(loop)) + idx)
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
                L1    =L1.item()
            )
    disc.train()
    gen.train()
    return torch.tensor(resultat).mean()


# In[20]:


import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from time import localtime
def main():
    os.mkdir("evaluation")
    writer = SummaryWriter("train{}-{}-{}".format(localtime().tm_mon, localtime().tm_mday, localtime().tm_hour))
    #instancing the models
    disc = Discriminator(in_channels=3).to(DEVICE)
    #print(disc)
    gen = Generator(init_weight=INIT_WEIGHTS).to(DEVICE)
    #print(gen)
    #instancing the optims
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    #instancing the Loss-functions
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    #if true loads the checkpoit in the ./

    #training data loading
    train_dataset = EYCON(path=TRAIN_DIR)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    test_dataset = EYCON(path=TRAIN_DIR,train=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    #enabling MultiPrecision Mode, the optimise performance
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    best=10000000
    resultat=1
    save_some_examples(gen, test_loader, -1, folder="evaluation")
    for epoch in range(NUM_EPOCHS):
        print("\n epoch", epoch, "\n")
        train_fn(
           disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler, writer, epoch=epoch
        )
        resultat=test_fn(disc, gen, test_loader,  L1_LOSS, BCE, writer, epoch=epoch)
        if best>resultat:
            best=resultat
            print("improvement of the loss from {} to {}".format(best,resultat))
            save_checkpoint(gen, opt_gen, epoch, filename=CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, epoch, filename=CHECKPOINT_DISC)

        save_some_examples(gen, test_loader, epoch, folder="evaluation")


# In[21]:


main()

