import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, Input # Input est nécessaire si vous reconstruisez le modèle pour inspecter
from glob import glob
from tqdm import tqdm
from scipy import ndimage
from tensorflow.keras import mixed_precision # Pour être cohérent avec l'entraînement
import matplotlib.pyplot as plt

# =============================================================================
# 0. CONFIGURATION AUTONOME POUR LA VISUALISATION
# (Doit correspondre aux chemins et paramètres de votre entraînement)
# =============================================================================

# --- Montage Drive (Assurez-vous que le Drive est monté) ---
if not os.path.exists('/content/drive'):
    from google.colab import drive
    drive.mount('/content/drive')
    print("[INFO] Google Drive mounted.")
else:
    print("[INFO] Google Drive already mounted.")

# --- Paramètres globaux (Doivent correspondre à ceux de votre entraînement) ---
IMG_H = 512
IMG_W = 512
N_FOLDS = 5 # Nombre de modèles à charger pour l'ensemble

# --- Chemins vers VOS DONNÉES ET MODÈLES ---
# Ces chemins DOIVENT pointer vers les mêmes emplacements que ceux de votre entraînement.
# Si vous avez copié localement pendant l'entraînement, il faudra peut-être adapter.
IMAGES_PATH = "/content/drive/MyDrive/Montgomery/images" # Ou "/content/temp_data/images" si vous avez copié localement
MASKS_PATH = "/content/drive/MyDrive/Montgomery/masks"   # Ou "/content/temp_data/masks" si vous avez copié localement
MODELS_SAVE_DIR = "/content/drive/MyDrive/DeepSup_Ensemble_512_augmented"

# --- Optimisation GPU (pour que l'inférence soit cohérente si utilisée) ---
try:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print("[INFO] Mixed Precision activated for inference (if models were trained with it).")
except Exception as e:
    print(f"[WARN] Could not activate Mixed Precision: {e}. Falling back to float32.")
    pass

# =============================================================================
# 1. CUSTOM FUNCTIONS (Re-définies car nécessaires pour charger les modèles)
# =============================================================================

# --- Métriques ---
# smooth doit être défini ici
smooth = 1e-15 
def dice_coef(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def iou_coef(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

def bce_dice_loss(y_true, y_pred):
    return tf.keras.losses.BinaryCrossentropy()(y_true, y_pred) + (1.0 - dice_coef(y_true, y_pred))

# Custom objects dictionary for loading models
custom_objects_dict = {
    'dice_coef': dice_coef,
    'iou_coef': iou_coef,
    'bce_dice_loss': bce_dice_loss,
    # Ajoutez toutes les autres fonctions custom si votre modèle les utilise (ex: 'squeeze_excite_block')
    # Pour MobileNetV2-UNet3+, les couches SepConvBlock ou SqueezeExciteBlock sont généralement compilées
    # comme des couches Keras standard et n'ont pas besoin d'être explicitement listées
    # SAUF si elles étaient des classes personnalisées héritant de layers.Layer
}

# =============================================================================
# 2. LOAD MODELS
# =============================================================================
tf.keras.backend.clear_session()

print(f"[INFO] Listing models from: {MODELS_SAVE_DIR}")
model_paths_list = sorted(glob(os.path.join(MODELS_SAVE_DIR, "*.keras")))

if not model_paths_list:
    print(f"[ERROR] No models found in {MODELS_SAVE_DIR}. Please check the path and ensure models were saved.")
    exit()
if len(model_paths_list) != N_FOLDS:
    print(f"[WARN] Expected {N_FOLDS} models but found {len(model_paths_list)}. Proceeding with available models.")

loaded_models = []
print(f"[INFO] Loading {len(model_paths_list)} models...")
for path in model_paths_list:
    try:
        # compile=False est crucial ici car nous n'allons pas entraîner
        m = tf.keras.models.load_model(path, custom_objects=custom_objects_dict, compile=False)
        loaded_models.append(m)
        print(f"   - Loaded: {os.path.basename(path)}")
    except Exception as e:
        print(f"[ERROR] Failed to load model {os.path.basename(path)}: {e}. This model will be skipped.")

if not loaded_models:
    print("[CRITICAL ERROR] No models could be loaded. Exiting.")
    exit()

# =============================================================================
# 3. RE-RUN INFERENCE TO GET final_dices AND final_ious
# (Nécessaire pour sélectionner Best/Median/Worst cases)
# =============================================================================
print("\n=== RE-CALCULATING FINAL SCORES FOR VISUALIZATION SELECTION ===")

all_images = np.array(sorted(glob(os.path.join(IMAGES_PATH, "*"))))
all_masks = np.array(sorted(glob(os.path.join(MASKS_PATH, "*"))))

if len(all_images) == 0:
    print(f"[CRITICAL ERROR] No images found in {IMAGES_PATH}. Please check paths.")
    exit()

final_dices = []
final_ious = []

print(f"[INFO] Evaluating on {len(all_images)} images to get scores for selection...")
for img_p, msk_p in tqdm(zip(all_images, all_masks), total=len(all_images)):
    # Read and preprocess image for model input
    x = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
    if x is None:
        print(f"[WARN] Image not found: {img_p}. Skipping.")
        continue
    x = cv2.resize(x, (IMG_W, IMG_H))
    img_in = np.stack((x,)*3, axis=-1).astype(np.float32) / 127.5 - 1.0
    img_in = np.expand_dims(img_in, axis=0) # Add batch dimension

    # Read and preprocess ground truth mask
    y = cv2.imread(msk_p, cv2.IMREAD_GRAYSCALE)
    if y is None:
        print(f"[WARN] Mask not found: {msk_p}. Skipping.")
        continue
    y = cv2.resize(y, (IMG_W, IMG_H))
    msk_true = (y > 127).astype(np.float32) # Strict binarization (0.0 or 1.0)
    msk_true = np.expand_dims(msk_true, axis=-1)

    # Ensemble prediction with TTA
    preds_sum = np.zeros((1, IMG_H, IMG_W, 1), dtype=np.float32)
    for m in loaded_models:
        out_norm = m.predict(img_in, verbose=0)
        p_norm = out_norm[0] if isinstance(out_norm, list) else out_norm

        img_in_flip = np.fliplr(img_in)
        out_flip = m.predict(img_in_flip, verbose=0)
        p_flip = out_flip[0] if isinstance(out_flip, list) else out_flip
        
        preds_sum += (p_norm + np.fliplr(p_flip)) / 2.0

    pred_avg = preds_sum / len(loaded_models)
    
    # Post-process (same as in training script)
    def post_process_inference(mask_tensor): # Renamed to avoid confusion with local function
        mask = (mask_tensor > 0.5).astype(np.float32)
        mask = (mask * 255).astype(np.uint8)
        nb, output, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if nb > 1:
            sizes = stats[1:, -1]
            img2 = np.zeros(output.shape, dtype=np.uint8)
            sorted_indices = np.argsort(sizes)[::-1]
            for comp_idx in range(min(2, len(sorted_indices))):
                if sizes[sorted_indices[comp_idx]] > 100:
                    img2[output == sorted_indices[comp_idx] + 1] = 1
            mask = img2
        mask = ndimage.binary_fill_holes(mask).astype(np.float32)
        return np.expand_dims(mask, axis=-1)

    pred_clean = post_process_inference(pred_avg[0])

    # Calculate Dice and IoU for this image
    t = msk_true.flatten()
    p = pred_clean.flatten()

    inter = np.sum(t * p)
    sum_u = np.sum(t) + np.sum(p)
    union = sum_u - inter

    d = (2. * inter + smooth) / (sum_u + smooth)
    i = (inter + smooth) / (union + smooth)

    final_dices.append(d)
    final_ious.append(i)

print(f"\n[INFO] Final scores re-calculated. Mean Dice: {np.mean(final_dices)*100:.2f}%, Mean IoU: {np.mean(final_ious)*100:.2f}%")


# =============================================================================
# 4. ROBUST READING FUNCTIONS FOR VISUALIZATION
# =============================================================================
# (Définies ici pour être sûres d'utiliser les variables globales IMG_W, IMG_H)
def read_image_safe_vis(path):
    if isinstance(path, bytes): path = path.decode()
    elif not isinstance(path, str): path = str(path)
    if not os.path.exists(path): raise FileNotFoundError(f"Image not found at {path}")
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (IMG_W, IMG_H))
    x = np.stack((x,)*3, axis=-1).astype(np.float32)
    x = (x / 127.5) - 1.0 
    return np.expand_dims(x, axis=0)

def read_image_display(path):
    if isinstance(path, bytes): path = path.decode()
    elif not isinstance(path, str): path = str(path)
    if not os.path.exists(path): raise FileNotFoundError(f"Image not found at {path}")
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return x

def read_mask_safe_vis(path):
    if isinstance(path, bytes): path = path.decode()
    elif not isinstance(path, str): path = str(path)
    if not os.path.exists(path): raise FileNotFoundError(f"Mask not found at {path}")
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (IMG_W, IMG_H))
    x = (x > 127).astype(np.float32)
    return x

def smooth_prediction(pred_mask, target_size=(1024, 1024)):
    pred_high_res = cv2.resize(pred_mask, target_size, interpolation=cv2.INTER_CUBIC)
    pred_blurred = cv2.GaussianBlur(pred_high_res, (9, 9), 0)
    mask_smooth = (pred_blurred > 0.5).astype(np.float32)
    mask_smooth = ndimage.binary_fill_holes(mask_smooth).astype(np.float32)
    return mask_smooth

# =============================================================================
# 5. GENERATE FINAL VISUALIZATION IMAGE
# =============================================================================
print("\n[INFO] Generating visualization image (Best, Median, Worst Cases)...")

# Sort scores to find examples
sorted_idx = np.argsort(final_dices)
# Best, Median, Worst cases
indices = [sorted_idx[-1], sorted_idx[len(sorted_idx)//2], sorted_idx[0]]
titles = ["BEST CASE", "MEDIAN CASE", "WORST CASE"]

plt.figure(figsize=(18, 12)) # Adjusted for better aspect ratio

for i, idx in enumerate(indices):
    img_path = all_images[idx]
    msk_path = all_masks[idx]
    
    # Re-predict for visualization if needed (using the same logic as inference)
    # We already have pred_raw from the previous step, but let's recompute for clarity if this block was separated
    x_input_model = read_image_safe_vis(img_path) # (1, 512, 512, 3)
    preds_sum_vis = np.zeros((1, IMG_H, IMG_W, 1), dtype=np.float32)
    
    for m in loaded_models:
        p1_output = m.predict(x_input_model, verbose=0)
        p1 = p1_output[0] if isinstance(p1_output, list) else p1_output
        
        p2_output = m.predict(np.fliplr(x_input_model), verbose=0)
        p2 = p2_output[0] if isinstance(p2_output, list) else p2_output
        
        preds_sum_vis += (p1 + np.fliplr(p2)) / 2.0
    
    # Raw average of ensemble predictions
    pred_raw_vis = preds_sum_vis[0, :, :, 0] / len(loaded_models)
    
    # Prepare for Display (Upscale to 1024 for aesthetics)
    img_disp = read_image_display(img_path)
    img_disp = cv2.resize(img_disp, (1024, 1024), interpolation=cv2.INTER_CUBIC)
    
    msk_true_disp = read_mask_safe_vis(msk_path)
    msk_true_disp = cv2.resize(msk_true_disp, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    
    pred_smooth = smooth_prediction(pred_raw_vis, target_size=(1024, 1024))
    
    # Plotting
    # Col 1: Original Image
    plt.subplot(3, 3, i*3 + 1)
    plt.imshow(img_disp, cmap='gray')
    plt.title(f"{titles[i]}\nChest X-ray", fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # Col 2: Ground Truth
    plt.subplot(3, 3, i*3 + 2)
    plt.imshow(img_disp, cmap='gray')
    plt.imshow(msk_true_disp, cmap='Greens', alpha=0.4)
    plt.title("Ground Truth (Green)", fontsize=12)
    plt.axis('off')
    
    # Col 3: Prediction
    plt.subplot(3, 3, i*3 + 3)
    plt.imshow(img_disp, cmap='gray')
    plt.imshow(pred_smooth, cmap='Reds', alpha=0.4)
    score = final_dices[idx] * 100
    plt.title(f"Our Model (Red)\nDice: {score:.2f}%", fontsize=12, color='darkred')
    plt.axis('off')

plt.tight_layout()
plt.savefig("Final_Results_512_HD.png", dpi=300, bbox_inches='tight')
plt.show()
print("[INFO] High-resolution image saved as 'Final_Results_512_HD.png'")
