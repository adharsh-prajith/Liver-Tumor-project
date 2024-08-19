import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import io
import random
import nibabel
import numpy as np
from glob import glob
import nibabel as nib
import tensorflow as tf
from nibabel import load
import matplotlib.pyplot as plt
from keras.utils import Sequence
from IPython.display import Image, display
from skimage.exposure import rescale_intensity
from skimage.segmentation import mark_boundaries
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard


class NiiDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_filenames, mask_filenames, batch_size, image_size, **kwargs):
        super().__init__(**kwargs)
        self.image_filenames = image_filenames
        self.mask_filenames = mask_filenames
        self.batch_size = batch_size
        self.image_size = image_size

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.mask_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]

        x = np.zeros((self.batch_size, *self.image_size, 1), dtype=np.float32)
        y = np.zeros((self.batch_size, *self.image_size, 3), dtype=np.float32)

        for i, (image_filename, mask_filename) in enumerate(zip(batch_x, batch_y)):
            im = nib.load(image_filename).get_fdata().astype(np.float32)
            ma = nib.load(mask_filename).get_fdata().astype(np.float32)
            # get the data from the image object
            # image_data = image.get_fdata()
            # mask_data = mask.get_fdata()
            im = (im - np.min(im)) / (np.max(im) - np.min(im))
            # get random slice from the volumes
            slice_index = random.randint(0, im.shape[2] - 1)
            x[i, :, :, 0] = im[:, :, slice_index]
            # One-hot encode the masks
            y[i, :, :, 0] = (ma[:, :, slice_index] == 0).astype(np.float32)  # Background
            y[i, :, :, 1] = (ma[:, :, slice_index] == 1).astype(np.float32)  # Liver
            y[i, :, :, 2] = (ma[:, :, slice_index] == 2).astype(np.float32)  # Tumor

        return x, y
def load_nifti_file(filename):
    data = nib.load(filename).get_fdata()
    print(f"Data shape: {data.shape}")
    print(f"Sample values (before normalization): {data[:, :, 0].flatten()[:10]}")
    return data

images = 'Data/Task03_Liver'

train_images = sorted(glob(os.path.join(images, 'imagesTr', '*.nii.gz')))
train_masks = sorted(glob(os.path.join(images, 'labelsTr', '*.nii.gz')))
print("Train images paths:", train_images)
print("Train masks paths:", train_masks)
image = nib.load(train_images[0]).get_fdata()
mask = nib.load(train_masks[0]).get_fdata()
image_data = load_nifti_file('Data/Task03_Liver/imagesTr/liver_0.nii.gz')
mask_data = load_nifti_file('Data/Task03_Liver/imagesTr/liver_0.nii.gz')

print("Image data min:", np.min(image_data))

print("Image data max:", np.max(image_data))
print("Mask data min:", np.min(mask_data))
print("Mask data max:", np.max(mask_data))
print("Image shape:", image_data.shape)
print("Mask shape:", mask_data.shape)
batch_size = 1 # The batch size to use when training the model
image_size = (512, 512)  # The size of the images

train_generator = NiiDataGenerator(train_images[:10], train_masks[:10], batch_size, image_size)
val_generator = NiiDataGenerator(train_images[10:], train_masks[10:], batch_size, image_size)
x, y = train_generator[0]
print("Batch x shape:", x.shape)  # Should be (batch_size, 512, 512, 1)
print("Batch y shape:", y.shape)  # Should be (batch_size, 512, 512, 1)
print("Batch x dtype:", x.dtype)  # Expected: float32
print("Batch y dtype:", y.dtype)  # Expected: float32
print("Sample x values:", x[0, :, :, 0])  # Print a slice to check the values
print("Sample y values:", y[0, :, :, 0])  # Print a slice to check the values

def encoder(inputs, filters, pool_size):
    conv_pool = tf.keras.layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(inputs)
    conv_pool = tf.keras.layers.MaxPooling2D(pool_size=pool_size)(conv_pool)
    return conv_pool

def decoder(inputs, concat_input, filters, transpose_size):
    up = tf.keras.layers.Concatenate()([tf.keras.layers.Conv2DTranspose(filters, transpose_size, strides=(2, 2), padding='same')(inputs), concat_input])
    up = tf.keras.layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(up)
    return up

def UNet(img_size=(512,512,1)):
    inputs = tf.keras.Input(shape=img_size)
    print(inputs.shape)
    print()

    # Encoder
    conv_pool1 = encoder(inputs, 32, (2, 2))
    print("\t Enc. 1 ->", conv_pool1.shape)
    print()
    conv_pool2 = encoder(conv_pool1, 64, (2, 2))
    print("\t\t Enc. 2 ->", conv_pool2.shape)
    print()
    conv_pool3 = encoder(conv_pool2, 128, (2, 2))
    print("\t\t\t Enc. 3 ->", conv_pool3.shape)
    print()
    conv_pool4 = encoder(conv_pool3, 256, (2, 2))
    print("\t\t\t\t Enc. 4 ->", conv_pool4.shape)
    print()

    # Bottleneck
    bridge = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv_pool4)
    print("\t\t\t\t\t Bridge Conv ->", bridge.shape)
    print()

    #Decoder
    up6 = decoder(bridge, conv_pool3, 256, (2, 2))
    print("\t\t\t\t Dec. 4 ->", up6.shape)
    print()
    up7 = decoder(up6, conv_pool2, 128, (2, 2))
    print("\t\t\t Dec. 3 ->", up7.shape)
    print()
    up8 = decoder(up7, conv_pool1, 64, (2, 2))
    print("\t\t Dec. 2 ->", up8.shape)
    print()
    up9 = decoder(up8, inputs, 32, (2, 2))
    print("\t Dec. 1 ->", up9.shape)
    print()
    outputs = tf.keras.layers.Conv2D(3, (1, 1), activation='softmax')(up9)
    print(outputs.shape)
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return model


def check_nifti_files(image_filenames, mask_filenames):
    for img_file, mask_file in zip(image_filenames, mask_filenames):
        img_data = nib.load(img_file).get_fdata()
        mask_data = nib.load(mask_file).get_fdata()

        if np.all(img_data == 0):
            print(f"Image file {img_file} contains all zeros.")
        if np.all(mask_data == 0):
            print(f"Mask file {mask_file} contains all zeros.")


check_nifti_files(train_images[:10], train_masks[:10])


def dice_coef(y_true, y_pred, smooth=1.):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
          =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    y_true_f = tf.cast(y_true_f, tf.float32)  # Cast to float32
    y_pred_f = tf.cast(y_pred_f, tf.float32)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
model = UNet()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=[dice_coef])

checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1)

history = model.fit(train_generator, steps_per_epoch=len(train_images), epochs=200, validation_data=val_generator, validation_steps=len(train_images), callbacks=[checkpoint, early_stopping, tensorboard])

plt.figure(figsize=(12,3))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], color='r')
plt.plot(history.history['val_loss'])
plt.ylabel('BCE Losses')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val.'], loc='upper right')
plt.subplot(1,2,2)
plt.plot(history.history['dice_coef'], color='r')
plt.plot(history.history['val_dice_coef'])
plt.ylabel('Dice Score')
plt.xlabel('Epoch')
plt.tight_layout()
plt.show()
test_pred = model.predict("Data/Task03_Liver/imagesTs/liver_132.nii.gz")
test_pred_thresh = np.argmax(test_pred[0,...], axis=-1)  # Use argmax to get the class with the highest probability


plt.figure(figsize=(15,5))
plt.subplot(1,4,1)
plt.title('Test CT Slice')
plt.imshow(np.rot90(test_img[0,...,0], 1), cmap='gray')
plt.axis('off')

plt.subplot(1,4,2)
plt.title('Test Liver Mask')
plt.imshow(np.rot90(test_mask[0,...,1], 1), cmap='gray')  # Liver mask

plt.subplot(1,4,3)
plt.title('Test Tumor Mask')
plt.imshow(np.rot90(test_mask[0,...,2], 1), cmap='gray')  # Tumor mask

plt.subplot(1,4,4)
plt.title('Predicted Tumor Mask')
plt.imshow(np.rot90(test_pred_thresh, 1), cmap='gray')  # Predicted mask
plt.axis('off')

plt.tight_layout()
plt.show()
