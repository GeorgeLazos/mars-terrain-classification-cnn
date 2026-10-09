SEED=2003
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import random
import json
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

def _resolve_root_dir() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    return Path.cwd()

#Paths from root directory
ROOT_DIR = str(_resolve_root_dir())
IMAGE_PATH = os.path.join(ROOT_DIR, "Data", "map-proj-v3")
LABELS_PATH = os.path.join(ROOT_DIR, "Data", "labels-map-proj-v3.txt")
SAVE_DATA_PATH = os.path.join(ROOT_DIR, "processed_data")
SAVE_MODEL_PATH = os.path.join(ROOT_DIR, "models")
SAVE_RESULTS_PATH = os.path.join(ROOT_DIR, "results")

#Reproducibility
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

#Dataset parameters
IMAGE_SIZE = (160, 160)
DATA_SPLIT = (0.7, 0.1, 0.2) # train, val, test

#Model identifier
MODEL_NAME = "mars_cnn_12"
BEST_MODEL_NAME = f"mars_cnn_8"

#Model parameters
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
DROPOUT_RATE = 0.30

EARLY_STOP_PATIENCE = 10
LR_REDUCE_PATIENCE = 1
LR_REDUCE_FACTOR = 0.75
CONV_WEIGHT_INIT = 'he_normal'          # ALSO USED IN 1 DENSE LAYER
FINAL_WEIGHT_INIT = 'glorot_uniform'

USE_CLASS_WEIGHTS = True
USE_FOCAL_LOSS = False
FOCAL_GAMMA = 2.0

L2_REG_VAL = 5e-5
L2_REG = tf.keras.regularizers.l2(L2_REG_VAL) if (L2_REG_VAL is not None and L2_REG_VAL > 0) else None

CLASSMAP = {
0:"other",
1:"crater",
2:"dark dune",
3:"slope streak",
4:"bright dune",
5:"impact ejecta",
6:"swiss cheese",
7:"spider"
}

#Model architecture
MODEL = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(*IMAGE_SIZE, 1)),

    tf.keras.layers.Conv2D(32, (3, 3), activation=None, kernel_initializer=CONV_WEIGHT_INIT, kernel_regularizer=L2_REG),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation=None, kernel_initializer=CONV_WEIGHT_INIT, kernel_regularizer=L2_REG),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), activation=None, kernel_initializer=CONV_WEIGHT_INIT, kernel_regularizer=L2_REG),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),

    tf.keras.layers.Flatten(),
  
    tf.keras.layers.Dense(128, activation=None, kernel_initializer=CONV_WEIGHT_INIT, kernel_regularizer=L2_REG),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(DROPOUT_RATE),
    tf.keras.layers.Dense(len(CLASSMAP), activation='softmax', kernel_initializer=FINAL_WEIGHT_INIT),
])

#Load model parameters from previous run to enable retraining with same configuration and/or continued training from previous weights
def get_parameters(parameters_path):
    if not os.path.isfile(parameters_path):
        raise FileNotFoundError(f"model_parameters.csv not found: {parameters_path}")

    parameters_df = pd.read_csv(parameters_path)
    if not {"parameter", "value"}.issubset(parameters_df.columns):
        raise ValueError("model_parameters.csv must contain 'parameter' and 'value' columns")

    params = dict(zip(parameters_df["parameter"], parameters_df["value"]))

    def _has_value(key):
        return key in params and pd.notna(params[key])

    def _safe_eval(value):
        try:
            return ast.literal_eval(str(value))
        except (ValueError, SyntaxError):
            return value

    typed_params = dict(params)

    for key in ["seed", "epochs", "batch_size", "early_stop_patience", "lr_reduce_patience"]:
        if _has_value(key):
            typed_params[key] = int(float(params[key]))

    for key in ["learning_rate", "dropout_rate", "lr_reduce_factor", "l2_regularization"]:
        if _has_value(key):
            typed_params[key] = float(params[key])

    if _has_value("Image_size"):
        image_size = _safe_eval(params["Image_size"])
        if isinstance(image_size, (tuple, list)) and len(image_size) == 2:
            typed_params["Image_size"] = (int(image_size[0]), int(image_size[1]))

    if _has_value("Data_split"):
        data_split = _safe_eval(params["Data_split"])
        if isinstance(data_split, (tuple, list)) and len(data_split) == 3:
            typed_params["Data_split"] = tuple(float(x) for x in data_split)

    if _has_value("class_weights"):
        class_weights = _safe_eval(params["class_weights"])
        if isinstance(class_weights, dict):
            typed_params["class_weights"] = {int(k): float(v) for k, v in class_weights.items()}

    loss_name = str(typed_params.get("loss_function", "sparse_categorical_crossentropy")).lower()
    typed_params["use_focal_loss"] = "focal" in loss_name
    typed_params["use_class_weights"] = "class_weights" in typed_params

    return typed_params

#Load model from memory if available
def load_model_from_memory(model_name=MODEL_NAME, save_model_path=SAVE_MODEL_PATH):
    model_path = os.path.join(save_model_path, f"{model_name}.keras")
    model = tf.keras.models.load_model(model_path)
    return model

#Train model using weights from pretrained run
def train_old_run(stored_model_name, use_old_config=False, save_results_path=SAVE_RESULTS_PATH):
    if not stored_model_name:
       raise ValueError("stored_model_name must be the name of a .keras model in models/")
    model = load_model_from_memory(stored_model_name)

    if use_old_config:
        parameters_path = os.path.join(save_results_path, stored_model_name, "model_parameters.csv")
        run_parameters = get_parameters(parameters_path)
        image_size = run_parameters.get("Image_size", IMAGE_SIZE)
        data_split = run_parameters.get("Data_split", DATA_SPLIT)
        seed = run_parameters.get("seed", SEED)
        epochs = run_parameters.get("epochs", EPOCHS)
        learning_rate = run_parameters.get("learning_rate", LEARNING_RATE)
        early_stop_patience = run_parameters.get("early_stop_patience", EARLY_STOP_PATIENCE)
        lr_reduce_patience = run_parameters.get("lr_reduce_patience", LR_REDUCE_PATIENCE)
        lr_reduce_factor = run_parameters.get("lr_reduce_factor", LR_REDUCE_FACTOR)
        batch_size = run_parameters.get("batch_size", BATCH_SIZE)
        use_focal_loss = run_parameters.get("use_focal_loss", USE_FOCAL_LOSS)
        use_class_weights = run_parameters.get("use_class_weights", USE_CLASS_WEIGHTS)
        optimizer_name = run_parameters.get("optimizer", "Adam")
        dropout_rate = run_parameters.get("dropout_rate", DROPOUT_RATE)
        conv_weight_init = run_parameters.get("conv_weight_init", CONV_WEIGHT_INIT)
        final_weight_init = run_parameters.get("final_weight_init", FINAL_WEIGHT_INIT)
        l2_reg = run_parameters.get("l2_regularization", L2_REG_VAL)
    else:
        image_size = IMAGE_SIZE
        data_split = DATA_SPLIT
        seed = SEED
        epochs = EPOCHS
        learning_rate = LEARNING_RATE
        early_stop_patience = EARLY_STOP_PATIENCE
        lr_reduce_patience = LR_REDUCE_PATIENCE
        lr_reduce_factor = LR_REDUCE_FACTOR
        batch_size = BATCH_SIZE
        use_focal_loss = USE_FOCAL_LOSS
        use_class_weights = USE_CLASS_WEIGHTS
        optimizer_name = "Adam"
        dropout_rate = DROPOUT_RATE
        conv_weight_init = CONV_WEIGHT_INIT
        final_weight_init = FINAL_WEIGHT_INIT
        l2_reg = L2_REG_VAL

    try:
        train_df, val_df, test_df = load_ready_img(save_path=SAVE_DATA_PATH, image_size=image_size)
    except(FileNotFoundError, OSError):
        print("Ready data not found loading from memory")
        train_df, val_df, test_df = load_and_preprocess_data(image_size=image_size, data_split=data_split, seed=seed)
        save_ready_img(train_df, val_df, test_df, image_size=image_size)

    print(f"Loaded training data with shape {train_df.cardinality().numpy()} and validation data with shape {val_df.cardinality().numpy()} and test data with shape {test_df.cardinality().numpy()}.")

    continued_model_name = f"{stored_model_name}_continued"
    history, model, class_weights, loss_function = train_model(
        train_df,
        val_df,
        model_name=continued_model_name,
        base_model=model,
        epochs=epochs,
        learning_rate=learning_rate,
        early_stop_patience=early_stop_patience,
        lr_reduce_patience=lr_reduce_patience,
        lr_reduce_factor=lr_reduce_factor,
        use_focal_loss=use_focal_loss,
        use_class_weights=use_class_weights,
        batch_size=batch_size,
        clone_base_model=False
    )

    test_results = test_model(model, test_df)
    save_results(
        history,
        test_results,
        class_weights,
        loss_function,
        model=model,
        model_name=continued_model_name,
        l2_reg=l2_reg,
        seed=seed,
        conv_weight_init=conv_weight_init,
        final_weight_init=final_weight_init,
        image_size=image_size,
        data_split=data_split,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        early_stop_patience=early_stop_patience,
        lr_reduce_patience=lr_reduce_patience,
        lr_reduce_factor=lr_reduce_factor,
        optimizer_name=optimizer_name,
        use_class_weights=use_class_weights,
        use_focal_loss=use_focal_loss
    )
    plot_train_loss(history)

#Plot train and val loss over the epochs
def plot_train_loss(history=None):
    train_loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss")
        
    plt.figure(figsize=(10,10))
    plt.plot(train_loss, label="train_loss", linewidth=2)
    plt.plot(val_loss, label="val_loss", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

#Save training history, test results, confusion matrix and classification report to disk for later analysis
def save_results(history, test_results, class_weights=None, loss_function=None, model=None, model_name=MODEL_NAME, l2_reg=L2_REG_VAL, seed=SEED, conv_weight_init=CONV_WEIGHT_INIT, final_weight_init=FINAL_WEIGHT_INIT, save_path=SAVE_RESULTS_PATH, classmap=CLASSMAP, image_size=None, data_split=None, epochs=None, batch_size=None, learning_rate=None, dropout_rate=None, early_stop_patience=None, lr_reduce_patience=None, lr_reduce_factor=None, optimizer_name="Adam", use_class_weights=USE_CLASS_WEIGHTS, use_focal_loss=USE_FOCAL_LOSS):
    image_size = image_size if image_size is not None else IMAGE_SIZE
    data_split = data_split if data_split is not None else DATA_SPLIT
    epochs = epochs if epochs is not None else EPOCHS
    batch_size = batch_size if batch_size is not None else BATCH_SIZE
    learning_rate = learning_rate if learning_rate is not None else LEARNING_RATE
    dropout_rate = dropout_rate if dropout_rate is not None else DROPOUT_RATE
    early_stop_patience = early_stop_patience if early_stop_patience is not None else EARLY_STOP_PATIENCE
    lr_reduce_patience = lr_reduce_patience if lr_reduce_patience is not None else LR_REDUCE_PATIENCE
    lr_reduce_factor = lr_reduce_factor if lr_reduce_factor is not None else LR_REDUCE_FACTOR

    model_parameters = {
        "model_name": model_name,
        "seed": seed,
        "Image_size": str(tuple(image_size)),
        "Data_split": str(tuple(data_split)),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "dropout_rate": dropout_rate,
        "early_stop_patience": early_stop_patience,
        "lr_reduce_patience": lr_reduce_patience,
        "lr_reduce_factor": lr_reduce_factor,
        "optimizer": optimizer_name,
        "use_class_weights": bool(use_class_weights),
        "use_focal_loss": bool(use_focal_loss),
        "loss_function": loss_function if loss_function else "sparse_categorical_crossentropy"
    }
    if class_weights is not None:
        model_parameters["class_weights"] = class_weights
    if model is not None:
        model_parameters["conv_weight_init"] = conv_weight_init
        model_parameters["final_weight_init"] = final_weight_init
        model_parameters["total_parameters"] = int(model.count_params())
        model_parameters["trainable_parameters"] = int(np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]))
    if l2_reg is not None:
        model_parameters["l2_regularization"] = l2_reg

    save_dir = os.path.join(save_path, model_name)
    os.makedirs(save_dir, exist_ok=True)
    
    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    history_df.to_csv(os.path.join(save_dir, "training_history.csv"), index=False)

    parameters_df = pd.DataFrame({
        "parameter": list(model_parameters.keys()),
        "value": [str(v) for v in model_parameters.values()]
    })
    parameters_df.to_csv(os.path.join(save_dir, "model_parameters.csv"), index=False)

    if model is not None:
        summary_str = []
        model.summary(print_fn=lambda x: summary_str.append(x))
        with open(os.path.join(save_dir, "model_architecture.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(summary_str))
        with open(os.path.join(save_dir, "model_architecture.json"), "w", encoding="utf-8") as f:
            f.write(model.to_json())

    if "confusion_matrix" in test_results:
        cm = np.array(test_results["confusion_matrix"])
        class_names = [classmap[i] for i in sorted(classmap.keys())]
        cm_df = pd.DataFrame(
            cm,
            index=[f"True_{name}" for name in class_names], 
            columns=[f"Pred_{name}" for name in class_names]
            )
        cm_df.to_csv(os.path.join(save_dir, "confusion_matrix.csv"))

    if "classification_report" in test_results:
        with open(os.path.join(save_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
            f.write(test_results["classification_report"])

    scalar_results ={}
    for key, value in test_results.items():
        if key in ["confusion_matrix", "classification_report"]:
            continue
        if isinstance(value, (np.floating, np.integer)):
            scalar_results[key] = float(value)
        elif isinstance(value, (int, float, str, bool)):
            scalar_results[key] = value
        else:
            scalar_results[key] = str(value)

    with open(os.path.join(save_dir, "test_results.json"), "w", encoding="utf-8") as f:
        json.dump(scalar_results, f, indent=2)

#Evaluate model performance on test set and return metrics and classification report
def test_model(model, test_df, classmap=CLASSMAP):
    results = model.evaluate(test_df, return_dict=True, verbose=1)
    y_true = np.concatenate([y.numpy() for _, y in test_df], axis=0)
    y_prob = model.predict(test_df, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)
    
    results = {
        **results,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average='macro'),
        "weighted_f1": f1_score(y_true, y_pred, average='weighted'),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }
    
    names = [classmap[i] for i in sorted(classmap.keys())]
    results["classification_report"] = classification_report(y_true, y_pred, target_names=names, digits=4)
    return results

#Implement focal loss to address class imbalance by down-weighting easy examples and focusing training on hard examples
def sparse_focal_loss(gamma=FOCAL_GAMMA):
    def focal_loss(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        cross_entropy = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

        y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        prob_true = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1)
        epsilon_s = tf.keras.backend.epsilon()
        prob_true = tf.clip_by_value(prob_true, epsilon_s, 1.0 - epsilon_s)

        focal_factor = tf.pow(1.0 - prob_true, gamma)
        return focal_factor * cross_entropy
    return focal_loss

#Calculate class weights to address class imbalance in training data.
def get_class_weights(train_df, num_classes=len(CLASSMAP)):
    label_counts = np.zeros(num_classes, dtype=np.int32)
    for _, y in train_df:
        labels = np.array(y.numpy()).ravel()
        for label in labels:
            label_counts[int(label)] += 1

    total_samples = np.sum(label_counts)
    class_weights = {i: float(total_samples / (num_classes * count)) for i, count in enumerate(label_counts) if count > 0} 
    return class_weights

#Train CNN on predefined model architecture and training parameters
def train_model(train_df,  val_df, model_name=MODEL_NAME, base_model=MODEL, epochs=EPOCHS, learning_rate=LEARNING_RATE, early_stop_patience=EARLY_STOP_PATIENCE, lr_reduce_patience=LR_REDUCE_PATIENCE, lr_reduce_factor=LR_REDUCE_FACTOR, save_model_path=SAVE_MODEL_PATH, use_class_weights=USE_CLASS_WEIGHTS, batch_size=BATCH_SIZE, use_focal_loss=USE_FOCAL_LOSS, gamma=FOCAL_GAMMA, clone_base_model=True):
    train_df = make_batches(train_df, shuffle=True, batch_size=batch_size)

    if use_class_weights:
        class_weights = get_class_weights(train_df)
    else:
        class_weights = None

    if use_focal_loss:
        loss_function = sparse_focal_loss(gamma=gamma)
        loss_function_log = f"sparse_focal_loss(gamma={gamma})"
    else:
        loss_function = 'sparse_categorical_crossentropy'
        loss_function_log = None

    model = tf.keras.models.clone_model(base_model) if clone_base_model else base_model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=loss_function,
                  metrics=['accuracy'])
    
    os.makedirs(save_model_path, exist_ok=True)
    model_path = os.path.join(save_model_path, f"{model_name}.keras")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stop_patience, min_delta=1e-3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=lr_reduce_factor, patience=lr_reduce_patience, min_delta=1e-3, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(filepath=model_path, mode='min', monitor='val_loss', save_best_only=True, verbose=1)
    ]

    history = model.fit(train_df, epochs=epochs, validation_data=val_df, callbacks=callbacks, class_weight=class_weights, verbose=1)

    return history, model, class_weights, loss_function_log 
        
#Save preprocessed data to memory for faster loading in future runs
def save_ready_img(train_df, val_df, test_df, save_path=SAVE_DATA_PATH, image_size=IMAGE_SIZE):
    os.makedirs(save_path, exist_ok=True)
    train_df.save(os.path.join(save_path, f"train_{image_size[0]}x{image_size[1]}"))
    val_df.save(os.path.join(save_path, f"val_{image_size[0]}x{image_size[1]}"))
    test_df.save(os.path.join(save_path, f"test_{image_size[0]}x{image_size[1]}"))   

#Preproccess image: read, decode, resize, normalize
def preprocess_img(path, image_size=IMAGE_SIZE):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=1)
    img = tf.image.resize(img, image_size)
    img = tf.cast(img, tf.float32) / 255.0
    return img

#Create batches from dataset
def make_batches(dataset, shuffle=True, batch_size=BATCH_SIZE, seed=SEED):
    if shuffle:
        dataset = dataset.shuffle(1000, seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

#Load from memory and prepare data for CNN training
def load_and_preprocess_data(image_size=IMAGE_SIZE , data_split=DATA_SPLIT, labels_path=LABELS_PATH, image_path=IMAGE_PATH, seed=SEED):
                
    #Load labels and corresponding image paths from txt file
    labels_df = pd.read_csv(labels_path ,sep=r'\s+' ,header=None, names=['filename', 'label'])
    labels_df['label'] = labels_df['label'].astype(int)                        
    full_img_paths = [os.path.join(image_path, fname) for fname in labels_df['filename']]
    labels = labels_df['label'].to_numpy(dtype=np.int32)

    #Check for missing files and filter out invalid paths and labels
    valid_paths, valid_labels = [], []
    missing_files = 0
    for path, label in zip(full_img_paths, labels):
        if os.path.isfile(path):
            valid_paths.append(path)
            valid_labels.append(label)
        else:
            missing_files += 1
    print(f"Found {len(valid_paths)} valid image files. {missing_files} files were missing from image dataset.")
                                                                                                        
    if not valid_paths:
        raise FileNotFoundError("No valid image files found. Please check the dataset paths.")
                                                                                                                        
    valid_paths = np.array(valid_paths)
    valid_labels = np.array(valid_labels, dtype=np.int32)

    #Preprocess images
    images_df = tf.data.Dataset.from_tensor_slices(valid_paths)
    images_df = images_df.map(lambda x: preprocess_img(x, image_size), num_parallel_calls=tf.data.AUTOTUNE)

    x_data = np.stack(list(images_df.as_numpy_iterator())).astype(np.float32)
    y_data = valid_labels.astype(np.int32)

    #Split into train and test sets
    x_train, x_temp, y_train, y_temp = train_test_split(x_data, y_data, test_size=(1 - data_split[0]), random_state=seed, stratify=y_data)
    val_size = data_split[1] / (data_split[1] + data_split[2])
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=(1 - val_size), random_state=seed, stratify=y_temp)

    #Create TensorFlow datasets
    train_df = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_df = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_df = make_batches(val_df, shuffle=False)
    test_df = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_df = make_batches(test_df, shuffle=False)

    return train_df, val_df, test_df

#Load preprocessed data from memory if available
def load_ready_img(save_path=SAVE_DATA_PATH, image_size=IMAGE_SIZE):
    train_path = os.path.join(save_path, f"train_{image_size[0]}x{image_size[1]}")
    val_path = os.path.join(save_path, f"val_{image_size[0]}x{image_size[1]}")
    test_path = os.path.join(save_path, f"test_{image_size[0]}x{image_size[1]}")

    for dataset_path in [train_path, val_path, test_path]:
        if not os.path.isfile(os.path.join(dataset_path, "dataset_spec.pb")):
            raise FileNotFoundError(f"Cached dataset missing/incomplete: {dataset_path}")

    train_df = tf.data.Dataset.load(train_path)
    val_df = tf.data.Dataset.load(val_path)
    test_df = tf.data.Dataset.load(test_path)
    return train_df, val_df, test_df

#Runs new model with specified parameters and model (GLOBAL PARAMETERS)
def main():
    try:
        train_df, val_df, test_df = load_ready_img()
    except(FileNotFoundError, OSError):
        print("Ready data not found, loading and preprocessing raw data...")
        train_df, val_df, test_df = load_and_preprocess_data()
        save_ready_img(train_df, val_df, test_df)
  
    print(f"Loaded training data with shape {train_df.cardinality().numpy()} and validation data with shape {val_df.cardinality().numpy()} and test data with shape {test_df.cardinality().numpy()}.")
            
    history, model, class_weights, loss_function = train_model(train_df, val_df)
    test_results = test_model(model, test_df)
    save_results( history, test_results, class_weights, loss_function, model=model)
    plot_train_loss(history)

#============================PLOTS============================

def plot_test_samples_with_predictions(model_name="mars_cnn_8", num_images=20, image_size=IMAGE_SIZE):
    class_ids = sorted(CLASSMAP.keys())
    if num_images < len(class_ids):
        raise ValueError(f"num_images must be at least {len(class_ids)} to include every class.")

    _, _, test_df = load_ready_img(image_size=image_size)
    model = load_model_from_memory(model_name=model_name)

    per_class_first = {}
    extra_samples = []

    for image, label in test_df.unbatch():
        label_int = int(label.numpy())
        sample = (image.numpy(), label_int)
        if label_int not in per_class_first:
            per_class_first[label_int] = sample
        else:
            extra_samples.append(sample)

        if len(per_class_first) == len(class_ids) and (len(per_class_first) + len(extra_samples)) >= num_images:
            break

    missing_classes = [cid for cid in class_ids if cid not in per_class_first]
    if missing_classes:
        raise ValueError(f"Test set is missing classes: {missing_classes}")

    selected = [per_class_first[cid] for cid in class_ids]
    needed = num_images - len(selected)
    selected.extend(extra_samples[:needed])

    images = np.stack([item[0] for item in selected], axis=0)
    true_labels = np.array([item[1] for item in selected], dtype=np.int32)
    pred_probs = model.predict(images, verbose=0)
    pred_labels = np.argmax(pred_probs, axis=1)

    rows, cols = 4, 5
    fig, axes = plt.subplots(rows, cols, figsize=(14, 10), dpi=140)
    axes = axes.flatten()

    for i in range(num_images):
        ax = axes[i]
        ax.imshow(np.squeeze(images[i]), cmap="gray")
        true_name = CLASSMAP[int(true_labels[i])]
        pred_name = CLASSMAP[int(pred_labels[i])]
        correct = int(true_labels[i]) == int(pred_labels[i])
        title_color = "green" if correct else "red"
        ax.set_title(f"T: {true_name}\nP: {pred_name}", fontsize=9, fontweight="bold", color=title_color)
        ax.axis("off")

    output_dir = os.path.join(ROOT_DIR, "results", model_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "test20_true_vs_pred.png")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path

def plot_mars_cnn8_train_val_and_test() -> Path:
    root = Path(ROOT_DIR)
    model_dir = root / "results" / "mars_cnn_8"
    history_path = model_dir / "training_history.csv"

    if not history_path.is_file():
        raise FileNotFoundError(f"Missing training history file: {history_path}")

    hist_df = pd.read_csv(history_path)
    required_cols = {"epoch", "accuracy", "loss", "val_accuracy", "val_loss"}
    missing = required_cols.difference(hist_df.columns)
    if missing:
        raise ValueError(f"training_history.csv is missing required columns: {sorted(missing)}")

    epochs = hist_df["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), dpi=160)

    # Loss subplot
    axes[0].plot(epochs, hist_df["loss"], label="Train Loss", linewidth=1.8, color="#d62728")
    axes[0].plot(epochs, hist_df["val_loss"], label="Val Loss", linewidth=1.8, color="#ff9896")
    axes[0].set_title("Loss vs Epoch", fontsize=10, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=9)
    axes[0].set_ylabel("Loss", fontsize=9)
    axes[0].grid(True, alpha=0.35)
    axes[0].legend(fontsize=7)

    # Accuracy subplot
    axes[1].plot(epochs, hist_df["accuracy"], label="Train Accuracy", linewidth=1.8, color="#2ca02c")
    axes[1].plot(epochs, hist_df["val_accuracy"], label="Val Accuracy", linewidth=1.8, color="#98df8a")
    axes[1].set_title("Accuracy vs Epoch", fontsize=10, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=9)
    axes[1].set_ylabel("Accuracy", fontsize=9)
    axes[1].grid(True, alpha=0.35)
    axes[1].legend(fontsize=7)

    output_path = model_dir / "train_val_test_accuracy_loss.png"
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path

def plot_per_class_accuracy(model_name="mars_cnn_8"):
    root_dir = ROOT_DIR
    results_dir = os.path.join(root_dir, "results", model_name)
    confusion_path = os.path.join(results_dir, "confusion_matrix.csv")

    if not os.path.isfile(confusion_path):
        raise FileNotFoundError(f"Could not find confusion matrix: {confusion_path}")

    cm_df = pd.read_csv(confusion_path, index_col=0)

    class_names = [CLASSMAP[i] for i in sorted(CLASSMAP.keys())]
    row_labels = [f"True_{name}" for name in class_names]
    col_labels = [f"Pred_{name}" for name in class_names]

    cm_df = cm_df.loc[row_labels, col_labels]
    cm = cm_df.to_numpy(dtype=np.float64)

    row_totals = cm.sum(axis=1)
    per_class_acc = np.divide(
        np.diag(cm),
        row_totals,
        out=np.zeros_like(row_totals, dtype=np.float64),
        where=row_totals > 0,
    )
    per_class_acc_pct = per_class_acc * 100.0

    fig, ax = plt.subplots(figsize=(8, 4), dpi=140)
    bars = ax.bar(class_names, per_class_acc_pct, color="#1f77b4", edgecolor="#0f3f66")
    ax.set_ylim(0, 104)
    ax.set_ylabel("Per-class Accuracy (%)", fontsize=9)
    ax.set_xlabel("Class", fontsize=9)
    ax.set_title(f"{model_name}: Per-class Accuracy", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=25, ha="right", fontsize=8)
    plt.yticks(fontsize=8)

    for bar, val in zip(bars, per_class_acc_pct):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.8,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    output_path = os.path.join(results_dir, "per_class_accuracy.png")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path

if __name__ == "__main__":
    main()
    #plot_mars_cnn8_train_val_and_test()
    #plot_per_class_accuracy()
    #plot_test_samples_with_predictions()
