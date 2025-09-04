import os
from PIL import Image
import subprocess
import torch
import re
import shlex

# === PARÁMETROS PRINCIPALES ===
input_reales_hd = "DATASETS/OBSTÁCULOS SIN LUZ/"
output_reales_256 = "DATASETS/OBSTÁCULOS SIN LUZ 256/"
input_generadas = "solo_generadas/17_7/14/"
output_generadas_256 = "solo_generadas/17_7/14_prueba/"
max_imgs =2338
target_size = (256, 256)

# === FUNCIONES DE PREPROCESADO ===
def crop_center_square(img):
    width, height = img.size
    min_dim = min(width, height)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    return img.crop((left, top, left + min_dim, top + min_dim))

def preprocess_images(input_folder, output_folder, output_size=(256, 256), max_imgs=2000, prefix="img"):
    os.makedirs(output_folder, exist_ok=True)
    count = 0
    for fname in sorted(os.listdir(input_folder)):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(input_folder, fname)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"⚠ Error al abrir {fname}: {e}")
            continue
        img = crop_center_square(img)
        img = img.resize(output_size, Image.BICUBIC)
        img.save(os.path.join(output_folder, f"{prefix}_{count:04d}.png"))
        count += 1
        if count >= max_imgs:
            break
    print(f"✅ {count} imágenes procesadas y guardadas en '{output_folder}'.")

def contar_imagenes(path):
    return len([f for f in os.listdir(path) if f.lower().endswith((".png", ".jpg", ".jpeg"))])

# === PASO 1: Preprocesar reales y generadas ===
print("📦 Preprocesando imágenes reales...")
preprocess_images(input_reales_hd, output_reales_256, target_size, max_imgs, prefix="real")

print("📦 Preprocesando imágenes generadas...")
preprocess_images(input_generadas, output_generadas_256, target_size, max_imgs, prefix="gen")

# === PASO 2: Verificar carpetas ===
if not os.path.exists(output_generadas_256):
    raise FileNotFoundError(f"❌ No se encontró la carpeta de imágenes generadas: {output_generadas_256}")
if not os.path.exists(output_reales_256):
    raise FileNotFoundError(f"❌ No se creó correctamente la carpeta de reales procesadas: {output_reales_256}")

print(f"📊 Imágenes reales procesadas: {contar_imagenes(output_reales_256)}")
print(f"📊 Imágenes generadas procesadas: {contar_imagenes(output_generadas_256)}")

# === PASO 3: Cálculo del FID con pytorch-fid ===
print("🔍 Calculando FID entre imágenes reales y generadas...")

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print("🚀 GPU detectada. Usando CUDA.")
else:
    print("🐢 No se detectó GPU. Usando CPU (más lento).")

cmd = f"python -m pytorch_fid {shlex.quote(output_reales_256)} {shlex.quote(output_generadas_256)} --device {device}"

try:
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    output_text = (result.stdout + "\n" + result.stderr).strip()

    print("✅ Resultado FID bruto (stdout + stderr):")
    print(output_text)

    match = re.search(r"([0-9]+\.[0-9]+)", output_text)
    if match:
        fid_value = float(match.group(1))
        print(f"📊 FID final: {fid_value:.4f}")
    else:
        print("⚠ No se pudo extraer el valor FID. Texto devuelto:")
        print(output_text)

except Exception as e:
    print("❌ Error al ejecutar pytorch-fid:", e)
