import torch
import torch.nn as nn
import torchvision.utils as vutils
import os
from cargar_config import obtener_config  # asumiendo que quieres seguir usando esto

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf // 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)




# Parámetros de entrada
dia = 17
mes = 7
intento = 15

def imprimir_parametros(dia, mes, intento):
    print(f"Día: {dia}")
    print(f"Mes: {mes}")
    print(f"Intento: {intento}")



# Cargar configuración (debes tener 'nz', 'ngf', 'nc' y 'image_size' definidos allí)
config = obtener_config(dia, mes, intento)

nz = int(config["nz"])
ngf = int(config["ngf"])
nc = 3  # Canales, por defecto 3 para RGB
image_size = int(config["image_size"])  # Esto debe ser 720 para que coincida

output_dir = f"solo_generadas/{dia}_{mes}/{intento}"
input_dir = f"model_weights/{dia}_0{mes}_{intento}/netG_last_epoch.pth"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Crear generador y cargar pesos entrenados
netG = Generator(nz=nz, ngf=ngf, nc=nc, ngpu=1).to(device)
netG.load_state_dict(torch.load(input_dir, map_location=device))
netG.eval()

# Generar imágenes
with torch.no_grad():
    for i in range(2106):  # Generar 10 imágenes
        noise = torch.randn(1, nz, 1, 1, device=device)
        fake_image = netG(noise).detach().cpu()
        vutils.save_image(fake_image, os.path.join(output_dir, f"imagen_{i}.png"), normalize=True)

print("✅ Imágenes generadas.")
