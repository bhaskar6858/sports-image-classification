import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import pickle
import os
import sys 

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root) 

def build_model(input_shape=(224, 224, 3), num_classes=100):
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  

  
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model

def save_training_artifacts(model, history, class_indices, base_path="models"):
    os.makedirs(base_path, exist_ok=True)
    
    
    model_path = os.path.join(base_path, "best_model.keras")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    class_indices_path = os.path.join(base_path, "class_indices.pkl")
    with open(class_indices_path, 'wb') as f:
        pickle.dump(class_indices, f)
    print(f"Class indices saved to {class_indices_path}")
    
    history_path = os.path.join(base_path, "training_history.pkl")
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"Training history saved to {history_path}")
    
    return {
        'model_path': model_path,
        'class_indices_path': class_indices_path,
        'history_path': history_path
    }

def train_and_save_model(train_data, val_data, num_classes, input_shape=(224, 224, 3)):
    """Complete training pipeline that saves all artifacts"""
    model = build_model(input_shape, num_classes)
    
    callbacks = [
        ModelCheckpoint('models/best_model.keras', 
                      monitor='val_accuracy', 
                      save_best_only=True,
                      verbose=1),
        EarlyStopping(monitor='val_accuracy', 
                     patience=10, 
                     restore_best_weights=True,
                     verbose=1),
        ReduceLROnPlateau(monitor='val_loss',
                         factor=0.2,
                         patience=3,
                         min_lr=1e-6,
                         verbose=1)
    ]
    

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=20,
        callbacks=callbacks
    )
    
    model.layers[1].trainable = True 
    for layer in model.layers[1].layers[:-20]: 
        layer.trainable = False
        
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history_fine = model.fit(
        train_data,
        validation_data=val_data,
        initial_epoch=history.epoch[-1] + 1,
        epochs=40,
        callbacks=callbacks
    )
    
    artifacts = save_training_artifacts(
        model=model,
        history=history_fine,
        class_indices=train_data.class_indices
    )
    
    return model, history_fine, artifacts


if __name__ == "__main__":
    from src.data_preprocessing import create_data_generators
    
    train_data, val_data, test_data = create_data_generators()
    
    model, history, artifacts = train_and_save_model(
        train_data=train_data,
        val_data=val_data,
        num_classes=len(train_data.class_indices)  
    )
    
    print("\nTraining completed!")
    print(f"Model saved to: {artifacts['model_path']}")
    print(f"Class indices saved to: {artifacts['class_indices_path']}")
