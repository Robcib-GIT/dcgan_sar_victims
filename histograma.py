import cv2
import numpy as np
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

def compute_average_histogram(images, bins=(8,8,8)):
    # Histograma acumulado
    hist_sum = np.zeros((bins[0]*bins[1]*bins[2],), dtype=np.float32)
    
    for img in images:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0,1,2], None, bins, [0,180,0,256,0,256])
        hist = cv2.normalize(hist, hist).flatten()
        hist_sum += hist
    
    # Promedio
    hist_avg = hist_sum / len(images)
    return hist_avg

def compare_histograms(hist1, hist2, method='bhattacharyya'):
    if method == 'bhattacharyya':
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    elif method == 'correlation':
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    elif method == 'intersection':
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)
    else:
        raise ValueError("Método desconocido")

# Carpetas de imágenes
folder_real = "DATASETS/robocup_etsii_256"
folder_fake = "solo_generadas/14_7/3"

# Cargar imágenes
images_real = load_images_from_folder(folder_real)
images_fake = load_images_from_folder(folder_fake)

# Calcular histogramas promedio
hist_real = compute_average_histogram(images_real)
hist_fake = compute_average_histogram(images_fake)

# Comparar histogramas
score_bhatt = compare_histograms(hist_real, hist_fake, method='bhattacharyya')
score_corr = compare_histograms(hist_real, hist_fake, method='correlation')
score_inter = compare_histograms(hist_real, hist_fake, method='intersection')

print("Bhattacharyya:", score_bhatt)  # 0 = idéntico
print("Correlación:", score_corr)     # 1 = idéntico
print("Intersección:", score_inter)   # mayor = más parecido