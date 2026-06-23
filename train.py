# =============================================================================
# 1. SETUP & CONFIGURATION (MODE PRODUCTION 512x512) - AVEC ALBUMENTATIONS
# =============================================================================
import os
import shutil
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import KFold
from glob import glob
from tqdm import tqdm
from scipy import ndimage
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt

# --- Installation Albumentations (Si nécessaire) ---
try:
    import albumentations as A
except ImportError:
    os.system('pip install -U albumentations')
    import albumentations as A

# --- Montage Drive ---
if not os.path.exists('/content/drive'):
    from google.colab import drive
    drive.mount('/content/drive')

# --- Paramètres ---
BATCH_SIZE = 2      # Stabilité avant tout
IMG_H = 512         # HAUTE RÉSOLUTION
IMG_W = 512         # HAUTE RÉSOLUTION
EPOCHS = 100
LR = 1e-4
N_FOLDS = 5         # 5 FOLDS COMPLETS (Pas de break)

# --- Chemins ---
DRIVE_IMG = "/content/drive/MyDrive/Montgomery/images"
DRIVE_MSK = "/content/drive/MyDrive/Montgomery/masks"
SAVE_DIR  = "/content/drive/MyDrive/DeepSup_Ensemble_512_augmented"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Chemins Locaux ---
LOCAL_IMG = "/content/temp_data/images"
LOCAL_MSK = "/content/temp_data/masks"
IMAGES_PATH = LOCAL_IMG
MASKS_PATH = LOCAL_MSK

# --- Optimisation GPU ---
try:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print("[INFO] Mixed Precision activé.")
except:
    pass

# --- Copie Rapide ---
if not os.path.exists(LOCAL_IMG):
    print("[INFO] Copie des données vers le disque local Colab...")
    try:
        shutil.copytree(DRIVE_IMG, LOCAL_IMG)
        shutil.copytree(DRIVE_MSK, LOCAL_MSK)
        print("[INFO] Copie terminée.")
    except Exception as e:
        print(f"[WARN] Erreur copie : {e}")

# =============================================================================
# 2. PIPELINE D'AUGMENTATION (ALBUMENTATIONS)
# =============================================================================
def get_training_augmentation():
    """
    Définit les augmentations basées sur l'article de référence :
    - Rotation (+/- 30 deg pour couvrir 15 et 30)
    - Flip Horizontal & Vertical
    - Brightness (Luminosité)
    - Gaussian Blur
    """
    return A.Compose([
        # Rotations (+/- 30 deg)
        A.Rotate(limit=30, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
        
        # Flips (Horizontal & Vertical comme dans l'article)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        
        # Effets Photométriques
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.0, p=0.4), # Brightness only
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
    ])

# =============================================================================
# 3. FONCTIONS DE LECTURE & DATASET
# =============================================================================

def read_image_uint8(path):
    """Lit l'image et la retourne en uint8 (0-255) pour Albumentations"""
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (IMG_W, IMG_H))
    x = np.stack((x,)*3, axis=-1) # 3 channels
    return x # uint8

def read_mask_uint8(path):
    """Lit le masque et le retourne en uint8 (0, 255) pour Albumentations"""
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (IMG_W, IMG_H))
    # Binarisation stricte
    _, x = cv2.threshold(x, 127, 255, cv2.THRESH_BINARY)
    return x # uint8

def tf_dataset_deepsup(x_paths, y_paths, batch=2, augment_data=False):
    
    # On initialise l'augmenter une seule fois
    augmenter = get_training_augmentation() if augment_data else None

    def _parse_and_process(img_path, msk_path):
        # 1. Lecture en Numpy/OpenCV (CPU)
        img = read_image_uint8(img_path)
        msk = read_mask_uint8(msk_path)
        
        # 2. Augmentation (Si activée)
        if augment_data and augmenter:
            transformed = augmenter(image=img, mask=msk)
            img = transformed['image']
            msk = transformed['mask']
            
        # 3. Normalisation (MobileNetV2: [-1, 1]) & Casting
        img = img.astype(np.float32)
        img = (img / 127.5) - 1.0
        
        msk = msk.astype(np.float32)
        msk = (msk > 127).astype(np.float32) # 0.0 ou 1.0
        msk = np.expand_dims(msk, axis=-1)
        
        # 4. Deep Supervision (Duplication des masques)
        return img, msk, msk, msk, msk, msk

    # Création du Dataset
    ds = tf.data.Dataset.from_tensor_slices((x_paths, y_paths))
    
    # Mapping via tf.numpy_function pour utiliser OpenCV/Albumentations
    ds = ds.map(lambda x, y: tf.numpy_function(
            func=_parse_and_process,
            inp=[x, y],
            Tout=[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]
        ), num_parallel_calls=tf.data.AUTOTUNE)

    # Formatage des shapes (Nécessaire après numpy_function)
    def _format(x, y1, y2, y3, y4, y5):
        x.set_shape([IMG_H, IMG_W, 3])
        y1.set_shape([IMG_H, IMG_W, 1])
        y2.set_shape([IMG_H, IMG_W, 1])
        y3.set_shape([IMG_H, IMG_W, 1])
        y4.set_shape([IMG_H, IMG_W, 1])
        y5.set_shape([IMG_H, IMG_W, 1])
        return x, (y1, y2, y3, y4, y5)

    ds = ds.map(_format, num_parallel_calls=tf.data.AUTOTUNE)
    
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)

# --- Métriques ---
def dice_coef(y_true, y_pred):
    smooth = 1e-15
    y_true = tf.cast(y_true, tf.float32); y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def iou_coef(y_true, y_pred):
    smooth = 1e-15
    y_true = tf.cast(y_true, tf.float32); y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

def bce_dice_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + (1.0 - dice_coef(y_true, y_pred))

# =============================================================================
# 4. ARCHITECTURE (Model 4 + Deep Supervision)
# =============================================================================
def squeeze_excite_block(input_tensor, ratio=16):
    filters = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Reshape((1, 1, filters))(se)
    se = layers.Dense(filters // ratio, activation='relu', use_bias=False)(se)
    se = layers.Dense(filters, activation='sigmoid', use_bias=False)(se)
    x = layers.Multiply()([input_tensor, se])
    return x

def sep_conv_block(x, filters):
    x = layers.DepthwiseConv2D(3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.LeakyReLU(0.1)(x)
    x = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.LeakyReLU(0.1)(x)
    return squeeze_excite_block(x)

def build_model_deepsup(input_shape):
    inputs = Input(input_shape)
    base = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')

    base.trainable = True
    for layer in base.layers[:-50]: layer.trainable = False

    layer_names = ['block_1_expand_relu', 'block_3_expand_relu', 'block_6_expand_relu', 'block_13_expand_relu', 'block_16_project']
    feature_extractor = models.Model(inputs=base.input, outputs=[base.get_layer(name).output for name in layer_names])
    c1, c2, c3, c4, c5 = feature_extractor(inputs)

    cat_filters = 32

    d4 = sep_conv_block(layers.Concatenate()([layers.MaxPooling2D((8,8))(c1), layers.MaxPooling2D((4,4))(c2), layers.MaxPooling2D((2,2))(c3), c4, layers.UpSampling2D((2,2))(c5)]), cat_filters*5)
    d3 = sep_conv_block(layers.Concatenate()([layers.MaxPooling2D((4,4))(c1), layers.MaxPooling2D((2,2))(c2), c3, layers.UpSampling2D((2,2))(d4), layers.UpSampling2D((4,4))(c5)]), cat_filters*5)
    d2 = sep_conv_block(layers.Concatenate()([layers.MaxPooling2D((2,2))(c1), c2, layers.UpSampling2D((2,2))(d3), layers.UpSampling2D((4,4))(d4), layers.UpSampling2D((8,8))(c5)]), cat_filters*5)
    d1 = sep_conv_block(layers.Concatenate()([c1, layers.UpSampling2D((2,2))(d2), layers.UpSampling2D((4,4))(d3), layers.UpSampling2D((8,8))(d4), layers.UpSampling2D((16,16))(c5)]), cat_filters*5)

    # Output names explicit for checkpoints
    out1 = layers.Conv2D(1, (1, 1), activation='sigmoid', dtype='float32', name='final_output')(layers.UpSampling2D((2, 2))(d1))
    out2 = layers.UpSampling2D((4, 4))(layers.Conv2D(1, (1, 1), activation='sigmoid', dtype='float32', name='ds2')(d2))
    out3 = layers.UpSampling2D((8, 8))(layers.Conv2D(1, (1, 1), activation='sigmoid', dtype='float32', name='ds3')(d3))
    out4 = layers.UpSampling2D((16, 16))(layers.Conv2D(1, (1, 1), activation='sigmoid', dtype='float32', name='ds4')(d4))
    out5 = layers.UpSampling2D((32, 32))(layers.Conv2D(1, (1, 1), activation='sigmoid', dtype='float32', name='ds5')(c5))

    return models.Model(inputs=inputs, outputs=[out1, out2, out3, out4, out5], name="MobileNet_UNet3Plus_DeepSup")

# =============================================================================
# 5. ENTRAINEMENT K-FOLD (COMPLET)
# =============================================================================
if __name__ == "__main__":

    tf.keras.backend.clear_session()

    all_images = np.array(sorted(glob(os.path.join(IMAGES_PATH, "*"))))
    all_masks = np.array(sorted(glob(os.path.join(MASKS_PATH, "*"))))

    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    model_paths_list = []

    print(f"=== DÉMARRAGE K-FOLD 512x512 (Batch {BATCH_SIZE}) ===")

    for fold, (train_idx, val_idx) in enumerate(kfold.split(all_images)):
        print(f"\n" + "="*40)
        print(f">>> FOLD {fold+1}/{N_FOLDS}")
        print(f"="*40)

        # NOTE: augment_data=True pour le training
        train_ds = tf_dataset_deepsup(all_images[train_idx], all_masks[train_idx], batch=BATCH_SIZE, augment_data=True)
        # NOTE: augment_data=False pour la validation
        val_ds = tf_dataset_deepsup(all_images[val_idx], all_masks[val_idx], batch=BATCH_SIZE, augment_data=False)

        tf.keras.backend.clear_session()
        model = build_model_deepsup((IMG_H, IMG_W, 3))

        if fold == 0:
            print(f"[INFO] Paramètres Totaux : {model.count_params():,}")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(LR),
            loss=[bce_dice_loss] * 5,
            loss_weights=[1.0, 0.5, 0.4, 0.3, 0.2],
            metrics={'final_output': [dice_coef, iou_coef]}
        )

        save_path = os.path.join(SAVE_DIR, f"model_512_fold_{fold+1}.keras")
        model_paths_list.append(save_path)

        callbacks = [
            ModelCheckpoint(save_path, save_best_only=True, monitor='val_final_output_dice_coef', mode='max', verbose=0),
            ReduceLROnPlateau(monitor='val_final_output_loss', factor=0.2, patience=8, min_lr=1e-7, verbose=0),
            EarlyStopping(monitor='val_final_output_dice_coef', patience=15, restore_best_weights=True, mode='max')
        ]

        history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=1)
        print(f"--> Best Fold {fold+1}: {max(history.history['val_final_output_dice_coef']):.5f}")

    # =============================================================================
    # 6. INFÉRENCE FINALE
    # =============================================================================
    print("\n=== CALCUL DU SCORE FINAL (512x512 - ENSEMBLE) ===")

    loaded_models = []
    for path in model_paths_list:
        m = tf.keras.models.load_model(path, compile=False)
        loaded_models.append(m)

    def post_process(mask):
        mask = (mask > 0.5).astype(np.float32)
        mask = (mask * 255).astype(np.uint8)
        nb, output, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if nb > 1:
            sizes = stats[1:, -1]
            img2 = np.zeros(output.shape, dtype=np.uint8)
            sorted_indices = np.argsort(sizes)[::-1]
            for i in range(min(2, len(sorted_indices))):
                if sizes[sorted_indices[i]] > 100:
                    img2[output == sorted_indices[i] + 1] = 1
            mask = img2
        mask = ndimage.binary_fill_holes(mask).astype(np.float32)
        return np.expand_dims(mask, axis=-1)

    final_dices = []
    final_ious = []

    print(f"[INFO] Évaluation sur {len(all_images)} images...")
    for img_p, msk_p in tqdm(zip(all_images, all_masks), total=len(all_images)):
        # Lecture standard pour évaluation
        x = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
        x = cv2.resize(x, (IMG_W, IMG_H))
        img_in = np.stack((x,)*3, axis=-1).astype(np.float32) / 127.5 - 1.0
        img_in = np.expand_dims(img_in, axis=0)

        y = cv2.imread(msk_p, cv2.IMREAD_GRAYSCALE)
        y = cv2.resize(y, (IMG_W, IMG_H))
        msk_true = (y > 127).astype(np.float32)
        msk_true = np.expand_dims(msk_true, axis=-1)

        preds_sum = np.zeros((1, IMG_H, IMG_W, 1), dtype=np.float32)

        for m in loaded_models:
            # TTA: Standard + Flip Horizontal
            out = m.predict(img_in, verbose=0)
            p_norm = out[0] if isinstance(out, list) else out

            out_flip = m.predict(np.fliplr(img_in), verbose=0)
            p_flip = out_flip[0] if isinstance(out_flip, list) else out_flip

            preds_sum += (p_norm + np.fliplr(p_flip)) / 2.0

        pred_avg = preds_sum / len(loaded_models)
        pred_clean = post_process(pred_avg[0])

        t = msk_true.flatten()
        p = pred_clean.flatten()

        inter = np.sum(t * p)
        sum_u = np.sum(t) + np.sum(p)

        d = (2. * inter + smooth) / (sum_u + smooth)
        i = (inter + smooth) / (sum_u - inter + smooth)

        final_dices.append(d)
        final_ious.append(i)

    print("\n" + "="*50)
    print(f"RESULTAT FINAL 512x512 (K-Fold Ensemble + Augmentations Avancées) :")
    print(f"DICE SCORE : {np.mean(final_dices)*100:.4f} %")
    print(f"IoU SCORE  : {np.mean(final_ious)*100:.4f} %")
    print("="*50)