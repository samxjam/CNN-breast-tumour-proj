import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict
import random
import os


# Set random seed for reproducibility
seed = 3888
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# Settings
img_width, img_height = 50, 50
batch_size = 512
AUTOTUNE = tf.data.AUTOTUNE

# Define your folders
tumour_folders = ["Invasive_Tumor", "DCIS_1", "DCIS_2", "Prolif_Invasive_Tumor"]
immune_folders = ["CD4+_T_Cells", "CD8+_T_Cells", "B_Cells", "Stromal"]

# Label mappings
sub_label_list = tumour_folders + immune_folders
sub_label_to_index = {name: idx for idx, name in enumerate(sub_label_list)}

def get_file_label_pairs(folders, main_label):
    file_label_pairs = []
    for folder in folders:
        files = tf.io.gfile.glob(os.path.join(folder, "*.png"))
        for f in files:
            sub_label = os.path.basename(folder)
            file_label_pairs.append((f, main_label, sub_label))
    return file_label_pairs

tumour_files = get_file_label_pairs(tumour_folders, main_label=1)
immune_files = get_file_label_pairs(immune_folders, main_label=0)

all_file_label_pairs = tumour_files + immune_files

print(f"Total images before balancing: {len(all_file_label_pairs)}")


def create_undersampled_dataset(file_label_pairs, target_size=3700):
    label_to_files = defaultdict(list)
    for f, main_label, sub_label in file_label_pairs:
        label_to_files[sub_label].append((f, main_label, sub_label))
   
    undersampled_pairs = []
    for sub_label, files in label_to_files.items():
        if len(files) > target_size:
            undersampled_pairs.extend(random.sample(files, target_size))
        else:
            undersampled_pairs.extend(files)
    random.shuffle(undersampled_pairs)
    return undersampled_pairs

undersampled_pairs = create_undersampled_dataset(all_file_label_pairs, target_size=3700)
print(f"Total images after undersampling: {len(undersampled_pairs)}")


def create_oversampled_dataset(file_label_pairs, target_size=41000):
    label_to_files = defaultdict(list)
    for f, main_label, sub_label in file_label_pairs:
        label_to_files[sub_label].append((f, main_label, sub_label))
   
    oversampled_pairs = []
    for sub_label, files in label_to_files.items():
        files_needed = target_size - len(files)
        original_files = files.copy()
       
        # Keep originals
        for f, main_label, s_label in original_files:
            oversampled_pairs.append((f, main_label, s_label, False))  # no augment
       
        # Create augmented
        for _ in range(files_needed):
            f, main_label, s_label = random.choice(original_files)
            oversampled_pairs.append((f, main_label, s_label, True))  # augment
    random.shuffle(oversampled_pairs)
    return oversampled_pairs

oversampled_pairs = create_oversampled_dataset(all_file_label_pairs, target_size=41000)
print(f"Total images after oversampling: {len(oversampled_pairs)}")

def decode_img(img_path, augment=False):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [img_width, img_height])
    img = img / 255.0
    if augment:
        img = tf.image.rot90(img, k=random.randint(0,3))
    return img

def prepare_dataset(pairs, oversample=False):
    file_paths = []
    main_labels = []
    sub_labels = []
    augment_flags = []

    if oversample:
        for f, main_label, sub_label, augment in pairs:
            file_paths.append(f)
            main_labels.append(main_label)
            sub_labels.append(sub_label_to_index[sub_label])
            augment_flags.append(augment)
    else:
        for f, main_label, sub_label in pairs:
            file_paths.append(f)
            main_labels.append(main_label)
            sub_labels.append(sub_label_to_index[sub_label])
            augment_flags.append(False)

    file_paths = np.array(file_paths)
    main_labels = np.array(main_labels)
    sub_labels = np.array(sub_labels)
    augment_flags = np.array(augment_flags)

    train_files, test_files, train_main, test_main, train_sub, test_sub, train_aug, test_aug = train_test_split(
        file_paths, main_labels, sub_labels, augment_flags, test_size=0.2, stratify=sub_labels, random_state=seed
    )

    def process_path(file_path, main_label, sub_label, augment):
        img = decode_img(file_path, augment)
        return img, {"binary_output": main_label, "multiclass_output": sub_label}

    train_ds = tf.data.Dataset.from_tensor_slices((train_files, train_main, train_sub, train_aug))
    train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.shuffle(1000).batch(batch_size).prefetch(AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((test_files, test_main, test_sub, test_aug))
    test_ds = test_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(AUTOTUNE)

    return train_ds, test_ds

def build_multitask_model():
    inputs = tf.keras.Input(shape=(img_width, img_height, 3))

    # Shared CNN
    x = tf.keras.layers.Conv2D(32, (3,3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)

    # Binary output
    binary_output = tf.keras.layers.Dense(1, activation='sigmoid', name="binary_output")(x)

    # Multiclass output
    multiclass_logits = tf.keras.layers.Dense(8, name="raw_multiclass_logits")(x)  # 8 subtypes (4 tumour, 4 immune)
    scaled_multiclass_logits = tf.keras.layers.Multiply()([multiclass_logits, binary_output])
    multiclass_output = tf.keras.layers.Activation('softmax', name="multiclass_output")(scaled_multiclass_logits)

    model = tf.keras.Model(inputs=inputs, outputs=[binary_output, multiclass_output])

    model.compile(
        optimizer='adam',
        loss={
            "binary_output": "binary_crossentropy",
            "multiclass_output": "sparse_categorical_crossentropy"
        },
        metrics={
            "binary_output": "accuracy",
            "multiclass_output": "accuracy"
        }
    )

    return model


def train_and_evaluate(train_ds, test_ds, title="Model"):
    model = build_multitask_model()
    model.summary()

    history = model.fit(train_ds, epochs=20, validation_data=test_ds)

    results = model.evaluate(test_ds)
    print(f"\nTest results - {title}:")
    print(f"Binary Accuracy: {results[3]:.4f}")
    print(f"Multiclass Accuracy: {results[4]:.4f}")

    return model


train_ds_under, test_ds_under = prepare_dataset(undersampled_pairs, oversample=False)

print("\n--- Training with UNDERSAMPLING ---")
model_under = train_and_evaluate(train_ds_under, test_ds_under, title="Undersampling")


train_ds_over, test_ds_over = prepare_dataset(oversampled_pairs, oversample=True)

print("\n--- Training with OVERSAMPLING ---")
model_over = train_and_evaluate(train_ds_over, test_ds_over, title="Oversampling")


import matplotlib.pyplot as plt

def plot_training_history(history, title_suffix=""):
    fig, axs = plt.subplots(1, 2, figsize=(14,5))

    # Binary accuracy
    axs[0].plot(history.history['binary_output_accuracy'], label='Train Binary Acc')
    axs[0].plot(history.history['val_binary_output_accuracy'], label='Val Binary Acc')
    axs[0].set_title(f'Binary Classification Accuracy {title_suffix}')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()
    axs[0].grid(True)

    # Multiclass accuracy
    axs[1].plot(history.history['multiclass_output_accuracy'], label='Train Multiclass Acc')
    axs[1].plot(history.history['val_multiclass_output_accuracy'], label='Val Multiclass Acc')
    axs[1].set_title(f'Multiclass Classification Accuracy {title_suffix}')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def train_and_evaluate(train_ds, test_ds, title="Model"):
    model = build_multitask_model()
    model.summary()

    history = model.fit(train_ds, epochs=20, validation_data=test_ds)

    results = model.evaluate(test_ds)
    print(f"\nTest results - {title}:")
    print(f"Binary Accuracy: {results[3]:.4f}")
    print(f"Multiclass Accuracy: {results[4]:.4f}")

    # Plot training curves
    plot_training_history(history, title_suffix=f"({title})")

    # Confusion Matrix for Multiclass
    print("\nGenerating confusion matrix...")
    y_true = []
    y_pred = []

    for batch in test_ds:
        images, labels = batch
        true_sub = labels['multiclass_output'].numpy()
        preds = model.predict(images, verbose=0)
        pred_multiclass = np.argmax(preds[1], axis=1)
       
        y_true.extend(true_sub)
        y_pred.extend(pred_multiclass)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    plot_confusion_matrix(y_true, y_pred, class_names=sub_label_list, title=f"Multiclass Confusion Matrix ({title})")

    return model
