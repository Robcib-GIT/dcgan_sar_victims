#%matplotlib inline
# https://onlinelibrary.wiley.com/doi/epdf/10.1155/2022/9005552 
# buen paper para metadatos de la red
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import matplotlib.pyplot as odfpy
import pandas as pd
import os
import ezodf
import time
fig, ax = plt.subplots()

########################################################################
import sys
from cargar_config import obtener_config

# Leer argumentos desde la línea de comandos
dia = int(sys.argv[1])
mes = int(sys.argv[2])
intento = int(sys.argv[3])
epoch_inicio = int(sys.argv[4])

# Obtener configuración desde el Excel
config = obtener_config(dia, mes, intento)

dataset       = config["dataset"]

# Suponiendo config es un diccionario con los valores ya correctos:
num_epochs    = int(config["num_epochs"])
image_size    = int(config["image_size"])
nz            = int(config["nz"])
ngf           = int(config["ngf"])
ndf           = int(config["ndf"])
batch_size    = int(config["batch_size"])

lr_gen        = float(config["lr_gen"])
lr_dis        = float(config["lr_dis"])
beta1         = float(config["beta1"])
beta2         = float(config["beta2"])

optimizador   = config["optimizador"]
use_batchnorm = config["use_batchnorm"]
activacion_d  = config["activacion_d"]

real_label    = float(config["real_label"])
fake_label    = float(config["fake_label"])

# Para mostrar con hasta 4 decimales
print(f"num_epochs = {num_epochs}")
print(f"image_size = {image_size}")
print(f"nz = {nz}")
print(f"ngf = {ngf}")
print(f"ndf = {ndf}")
print(f"batch_size = {batch_size}")
print(f"lr_gen = {lr_gen:.5f}")
print(f"lr_dis = {lr_dis:.5f}")
print(f"beta1 = {beta1:.4f}")
print(f"beta2 = {beta2:.4f}")
print(f"real_label = {real_label:.4f}")
print(f"fake_label = {fake_label:.4f}")
print(f"optimizador = {optimizador}")
print(f"use_batchnorm = {use_batchnorm}")
print(f"activacion_d = {activacion_d}")

print(f"\n🧩 Configuración cargada para {dia}/{mes} intento {intento}:")

#############################################################################


resultados = []


tiempo_inicio = time.time()
# Definir el nombre del archivo Excel
archivo_excel = 'datos_ensayos.ods'

# Define la hoja que quieres usar, por ejemplo:
nombre_hoja = f"{dia}-{mes}_{intento}"  # o cualquier nombre dinámico o fijo

# Set random seed for reproducibility
manualSeed = random.randint(1, 10000) # use if you want new results
# probar con 42
save_epoch = 100

print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True) # Needed for reproducible results

# Root directory for dataset

#dataroot = "/home/christyan/GAN/celeba"
dataroot = f"/home/robcib/Desktop/Mario/celeba/DATASETS/{dataset}/"


# Number of workers for dataloader
workers = 32



# Number of channels in the training images. For color images this is 3
nc = 3


# Create a directory to save the models' weights
weights_dir = os.path.join("model_weights", f"{dia}_{mes}", f"{intento}")
os.makedirs(weights_dir, exist_ok=True)

continuar_entrenamiento = True  # ✅ ponlo a True si quieres continuar desde pesos guardados
ruta_pesos_G = os.path.join(weights_dir, "netG_last_epoch.pth")
ruta_pesos_D = os.path.join(weights_dir, "netD_last_epoch.pth")


# Create a directory to save generated images
train_generated_images_dir = "generated_images/" + f"{dia}" + "_" + f"{mes}" + "/" + f"{intento}"
os.makedirs(train_generated_images_dir, exist_ok=True)



# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print("device", device)


# Plot some training images
real_batch = next(iter(dataloader))
#plt.figure(figsize=(8,8))
#plt.axis("off")
#plt.title("Training Images")
#plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
#plt.show()

# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)
    

    

# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-GPU if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the ``weights_init`` function to randomly initialize all weights
#  to ``mean=0``, ``stdev=0.02``.
netG.apply(weights_init)

# Print the model
print(netG)

if activacion_d == "leakyrelu":
    activation_fn = nn.LeakyReLU(0.2, inplace=True)
elif activacion_d == "relu":
    activation_fn = nn.ReLU(inplace=True)



class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # Input: (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),         # -> (ndf) x 32 x 32
            activation_fn,

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),     # -> (ndf*2) x 16 x 16
            nn.BatchNorm2d(ndf * 2),
            activation_fn,

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), # -> (ndf*4) x 8 x 8
            nn.BatchNorm2d(ndf * 4),
            activation_fn,

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), # -> (ndf*8) x 4 x 4
            nn.BatchNorm2d(ndf * 8),
            activation_fn,

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),       # -> 1 x 1 x 1
            nn.Sigmoid()
        )

    def forward(self, input):
        # Output shape: (batch_size, 1, 1, 1) → flatten to (batch_size)
        return self.main(input).view(-1)

# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-GPU if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the ``weights_init`` function to randomly initialize all weights
# like this: ``to mean=0, stdev=0.2``.
netD.apply(weights_init)

# Print the model
print(netD)

# Initialize the ``BCELoss`` function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator

fixed_noise = torch.randn(64, nz, 1, 1, device=device)

#fixed_noise = 10 * (fixed_noise)

if optimizador == "adam":
    optimizerG = optim.Adam(netG.parameters(), lr=lr_gen, betas=(beta1, beta2))
    optimizerD = optim.Adam(netD.parameters(), lr=lr_dis, betas=(beta1, beta2))
elif optimizador == "rmsprop":
    optimizerG = optim.RMSprop(netG.parameters(), lr=lr_gen)
    optimizerD = optim.RMSprop(netD.parameters(), lr=lr_dis)
elif optimizador == "sgd":
    optimizerG = optim.SGD(netG.parameters(), lr=lr_gen, momentum=0.9)
    optimizerD = optim.SGD(netD.parameters(), lr=lr_dis, momentum=0.9)

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0


if continuar_entrenamiento:
    if os.path.exists(ruta_pesos_G) and os.path.exists(ruta_pesos_D):
        print("📦 Cargando pesos previos del generador y discriminador...")
        netG.load_state_dict(torch.load(ruta_pesos_G))
        netD.load_state_dict(torch.load(ruta_pesos_D))
    else:
        print("⚠️ No se encontraron pesos guardados. Comenzando desde cero.")



print("Starting Training Loop...")
# For each epoch
for epoch in range(epoch_inicio, num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
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
        #noise = 10 * (noise)
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
        if i % 150 == 0:
            tiempo_actual = time.time()
            tiempo_transcurrido = tiempo_actual - tiempo_inicio
            tiempo_formateado = time.strftime("%H:%M:%S", time.gmtime(tiempo_transcurrido))
        
            # Acumula el diccionario con los datos en la lista resultados
            resultados.append({
                'Epoch': epoch,
                'Total_Epochs': num_epochs,
                'Batch': i,
                'Total_Batches': len(dataloader),
                'Loss_D': errD.item(),
                'Loss_G': errG.item(),
                'D(x)': D_x,
                'D(G(z))_1': D_G_z1,
                'D(G(z))_2': D_G_z2,
                'Tiempo (hh:mm:ss)': tiempo_formateado
            })
        
            print(f"{epoch}/{num_epochs} - - {nombre_hoja}")


        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 150 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            print("almacenando")
            vutils.save_image(fake, f"{train_generated_images_dir}/epoch_{epoch}_iter_{iters}.png", normalize=True)
        iters += 1
        
    #plt.plot(G_losses,label="G")
    #plt.plot(D_losses,label="D")
    #plt.xlabel("iterations")
    #plt.ylabel("Loss")
    #plt.legend()
    #plt.show()
    ax.clear()
    ax.plot(G_losses, label="G")
    ax.plot(D_losses, label="D")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")
    ax.legend()
    #plt.pause(0.001)
    a = epoch % save_epoch

    if epoch % 2 == 0:
        # Construye el path del directorio
        dir_path = os.path.join("curvas_train", f"{dia}_{mes}", f"{intento}")
    
        # Crea la carpeta si no existe
        os.makedirs(dir_path, exist_ok=True)
    
        # Guarda el gráfico
        filename = os.path.join(dir_path, f"loss_plot_epoch_{epoch}.png")
        plt.savefig(filename)

    if a == 0:
        torch.save(netG.state_dict(), os.path.join(weights_dir,f"netG_epoch_{epoch}.pth"))
        torch.save(netD.state_dict(), os.path.join(weights_dir,f"netD_epoch_{epoch}.pth"))

    if (epoch - epoch_inicio + 1) % save_epoch == 0 and len(resultados) > 0:
        nuevos_datos = pd.DataFrame(resultados)
        num_columnas = len(nuevos_datos.columns)

        if os.path.exists(archivo_excel):
            doc = ezodf.opendoc(archivo_excel)
        else:
            doc = ezodf.newdoc(doctype="ods")

        sheet = None
        for s in doc.sheets:
            if s.name == nombre_hoja:
                sheet = s
                break

        if sheet is None:
            sheet = ezodf.Sheet(nombre_hoja, size=(len(nuevos_datos) + 1, num_columnas))
            doc.sheets += sheet
            for col_idx, col_name in enumerate(nuevos_datos.columns):
                sheet[0, col_idx].set_value(col_name)
            current_row = 1
        else:
            current_row = 1
            while current_row < sheet.nrows():
                if sheet[current_row, 0].value is None:
                    break
                current_row += 1

        for row in nuevos_datos.itertuples(index=False):
            for col_idx, value in enumerate(row):
                if current_row < sheet.nrows():
                    sheet[current_row, col_idx].set_value(value)
            current_row += 1

        doc.saveas(archivo_excel)
        print(f"[Excel] Guardado datos de la época {epoch} en fila {current_row} de '{nombre_hoja}'")
        resultados.clear()



# Save the weights of the generator and discriminator
torch.save(netG.state_dict(), os.path.join(weights_dir, "netG_last_epoch.pth"))
torch.save(netD.state_dict(), os.path.join(weights_dir, "netD_last_epoch.pth"))

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join("generated_images/" + f"{dia}" + "_" + f"{mes}" + "/" + f"{intento}", "final_loss_plot.png"))
plt.close()


fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
ani.save(os.path.join("generated_images/" + f"{dia}" + "_" + f"{mes}" + "/" + f"{intento}", "training_animation.mp4"))  # o .gif si prefieres
plt.close()


HTML(ani.to_jshtml())

# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:264],padding=5, normalize=True).cpu(),(1,2,0)))

plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))

plt.savefig(os.path.join("generated_images/" + f"{dia}" + "_" + f"{mes}" + "/" + f"{intento}", "real_vs_fake.png"))
plt.close()




