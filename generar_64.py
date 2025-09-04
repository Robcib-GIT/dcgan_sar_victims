import torch
import torch.nn as nn
import torchvision.utils as vutils
import os
from cargar_config import obtener_config  # si lo usas para los parámetros

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # Input Z: (nz, 1, 1) -> (ngf*8, 4, 4)
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # (ngf*8, 4, 4) -> (ngf*4, 8, 8)
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # (ngf*4, 8, 8) -> (ngf*2, 16, 16)
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # (ngf*2, 16, 16) -> (ngf, 32, 32)
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # (ngf, 32, 32) -> (nc, 64, 64)
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


# Parámetros
dia = 24
mes = 7
intento = 3

config = obtener_config(dia, mes, intento)
nz = int(config["nz"])
ngf = int(config["ngf"])
nc = 3  # RGB

output_dir = f"solo_generadas/{dia}_{mes}/{intento}"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

netG = Generator(nz=nz, ngf=ngf, nc=nc, ngpu=1).to(device)
netG.load_state_dict(torch.load("model_weights/netG_last_epoch.pth", map_location=device))
netG.eval()

with torch.no_grad():
    for i in range(1797):
        noise = torch.randn(1, nz, 1, 1, device=device)
        fake_image = netG(noise).detach().cpu()
        vutils.save_image(fake_image, os.path.join(output_dir, f"imagen_{i}.png"), normalize=True)

print("✅ Imágenes 64x64 generadas.")
