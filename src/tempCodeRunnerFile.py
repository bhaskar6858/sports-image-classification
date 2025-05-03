import os
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def is_valid_image(file_path):
    """Check if file is a valid image"""
    try:
        img = Image.open(file_path)
        img.verify()
        return True
    except Exception as e:
        print(f"Invalid image {file_path}: {e}")
        return False

def filter_valid_images(image_directory):
    """Filter out invalid images from directory"""
    valid_images = []
    for subdir, dirs, files in os.walk(image_directory):
        for file in files:
            file_path = os.path.join(subdir, file)
            if is_valid_image(file_path):
                valid_images.append(file_path)
    return valid_images

def create_data_generators(img_size=224, batch_size=32):
    """Create train, validation, and test data generators"""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2]
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_data = train_datagen.flow_from_directory(
        'data/train',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_data = test_datagen.flow_from_directory(
        'data/val',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_data = test_datagen.flow_from_directory(
        'data/test',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_data, val_data, test_data