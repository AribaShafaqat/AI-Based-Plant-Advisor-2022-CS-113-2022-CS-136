from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Check GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Dataset directories
dataset_root_train = 'D:/AI_PROJECT/dataset/Train_Set_Folder'
dataset_root_val = 'D:/AI_PROJECT/dataset/Validation_Set_Folder'
dataset_root_test = 'D:/AI_PROJECT/dataset/Test_Set_Folder'

# Function to count the number of images in each category
def count_images_in_categories(dataset_root):
    # Dictionary to store the count of images per category
    image_count = {}

    # Iterate through each subdirectory (category) in the dataset directory
    for category in os.listdir(dataset_root):
        category_path = os.path.join(dataset_root, category)
        if os.path.isdir(category_path):
            # Count the number of images in the category
            num_images = len([f for f in os.listdir(category_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))])
            image_count[category] = num_images
    
    return image_count

# Count images in the train, validation, and test datasets
train_image_count = count_images_in_categories(dataset_root_train)
val_image_count = count_images_in_categories(dataset_root_val)
test_image_count = count_images_in_categories(dataset_root_test)

# Print the counts
print("Number of images in each category of the Train dataset:")
for category, count in train_image_count.items():
    print(f"{category}: {count} images")

print("\nNumber of images in each category of the Validation dataset:")
for category, count in val_image_count.items():
    print(f"{category}: {count} images")

print("\nNumber of images in each category of the Test dataset:")
for category, count in test_image_count.items():
    print(f"{category}: {count} images")

# ImageDataGenerator for data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
train_data = train_datagen.flow_from_directory(
    dataset_root_train,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_data = val_test_datagen.flow_from_directory(
    dataset_root_val,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_data = val_test_datagen.flow_from_directory(
    dataset_root_test,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Compute class weights based on the class distribution in your training dataset
class_weights = compute_class_weight('balanced', classes=np.unique(train_data.classes), y=train_data.classes)

# Convert to dictionary format
class_weight_dict = dict(zip(np.unique(train_data.classes), class_weights))

# Model path
model_path = 'best_model_2_15plants_final.keras'

# Define the CNN model (Model 2 configuration)
def build_model_2():
    model = tf.keras.Sequential([ 
        tf.keras.layers.Input(shape=[224, 224, 3]), 
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'), 
        tf.keras.layers.MaxPooling2D((2, 2)), 
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'), 
        tf.keras.layers.MaxPooling2D((2, 2)), 
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'), 
        tf.keras.layers.MaxPooling2D((2, 2)), 
        tf.keras.layers.Flatten(), 
        tf.keras.layers.Dense(256, activation='relu'), 
        tf.keras.layers.Dense(train_data.num_classes, activation='softmax')  # Softmax for multi-class classification
    ]) 
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), 
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy']) 
    return model

# Check if the model already exists
if os.path.exists(model_path):
    print(f"Loading pre-trained model from: {model_path}")
    best_model = load_model(model_path)
else:
    print("Training Model 2 from scratch...")
    # Build Model 2
    model_2 = build_model_2()

    # Train the model
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint_callback = ModelCheckpoint(model_path, save_best_only=True)

    model_2.fit(
        train_data,
        epochs=10,
        validation_data=val_data,
        class_weight=class_weight_dict,  # Pass the class weights here
        callbacks=[early_stopping, checkpoint_callback]
    )
    
    # Save the trained model
    model_2.save(model_path)
    best_model = model_2

# Evaluate the model on the test dataset
test_loss, test_accuracy = best_model.evaluate(test_data)
print(f"Model Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Function to predict a single image and display confidence scores for all classes
def predict_image_with_confidence(image_path, model, class_indices, threshold=0.5):
    try:
        # Load image and prepare it for prediction
        img = image.load_img(image_path, target_size=(224, 224))  # Resize image
        img_array = image.img_to_array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Get predictions
        predictions = model.predict(img_array)
        class_labels = {v: k for k, v in class_indices.items()}  # Reverse mapping of class indices

        # Print confidence scores for each class
        for idx, score in enumerate(predictions[0]):
            print(f"{class_labels[idx]}: {score:.2%}")

        # Get the predicted class
        predicted_class_idx = np.argmax(predictions[0])  # Use predictions[0] for batch of size 1
        predicted_class = class_labels[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx]

        # If confidence is below the threshold, consider it as "Unknown"
        if confidence < threshold:
            print(f"Prediction: Unknown (Confidence: {confidence:.2%})")
            return "Unknown"
        else:
            print(f"Prediction: {predicted_class} (Confidence: {confidence:.2%})")
            return predicted_class
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Predict on multiple images
test_image_paths = [
    'D:/AI_PROJECT/dataset/Test_Set_Folder/aloevera/aloevera561.jpg',
    'D:/AI_PROJECT/dataset/Test_Set_Folder/aloevera/aug_0_691.jpg',
    'D:/AI_PROJECT/dataset/Test_Set_Folder/aloevera/aug_0_8938.jpg',
    'D:/AI_PROJECT/Aloe_Vera.jpg',  # External image
]

for image_path in test_image_paths:
    predicted_class = predict_image_with_confidence(image_path, best_model, train_data.class_indices)
    if predicted_class:
        print(f"Predicted Class for {image_path}: {predicted_class}")
