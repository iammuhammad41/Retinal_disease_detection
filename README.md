# Diabetic Retinopathy Detection with Attention-enhanced InceptionV3

This repository contains a TensorFlow/Keras pipeline to classify stages of diabetic retinopathy using retinal images. It employs a deeper spatial-attention mechanism on top of a frozen InceptionV3 backbone and performs stratified K‑fold cross-validation with class balancing.



## 🚀 Project Structure

```
diabetic-retinopathy-attention/
├── data/                                   # Unzipped image folders (e.g., data/train_11)
│   ├── train_11/                           # Subset of training images for demonstration
├── notebooks/                              # (Optional) Jupyter/Colab notebooks
├── scripts/                                # Python scripts
│   └── train_dr_attention.py               # Main training and evaluation script
├── model_architecture.png                  # Visualization of model graph
└── README.md                               # This file
```



## 🔧 Requirements

* Python 3.8+
* TensorFlow 2.x / Keras
* numpy, pandas, matplotlib, seaborn
* scikit-learn, scikit-image
* tqdm, colorama
* (Optional) p7zip-full for extracting split archives

Install via pip:

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn scikit-image tqdm colorama
```



## ⚙️ Data Preparation

1. **Labels CSV**: Unzip `trainLabels.csv.zip` to obtain `trainLabels.csv`.
2. **Image archive**: Extract images from `train.zip.001` using `7z`, e.g.:

   ```bash
   7z x train.zip.001 -o./data/train_11 "train/11*.jpeg"
   ```
3. **Subset & Filter**: The script maps `trainLabels.csv` entries to the extracted files, filters missing images, and shuffles the DataFrame.
4. **Visualization**: A small batch of one example per class (0–4) is displayed to verify data pipeline.


## 🏗 Model Architecture

The `create_deeper_attention_model` function builds a Keras model with:

* **InceptionV3** base (frozen, `include_top=False`)
* **BatchNormalization** on extracted features
* **Spatial attention** via stacked Conv2D (128→64→32→1 output) with sigmoid mask
* **Feature rescaling** and multiplication by attention mask
* **GlobalAveragePooling2D** of masked features
* **Rescale GAP** dividing pooled features by pooled mask
* **Dense layers** with Dropout and L2 regularization
* **Softmax** output for 5 classes




## 📊 Training & Evaluation

The code performs stratified 10‑fold cross‑validation:

1. **Balance classes** to the maximum fold-wise count by oversampling.
2. **ImageDataGenerator** for on‑the‑fly augmentation and rescaling.
3. **Callbacks**:

   * `ModelCheckpoint` (save best by val\_accuracy)
   * `EarlyStopping` (restore best weights, patience=15)
   * `ReduceLROnPlateau` (factor=0.1, patience=7)
4. **Metrics tracked**: categorical accuracy, loss, AUC.
5. **Fold results** are collected and averaged across folds.
6. **Visualizations**:

   * Training/validation accuracy, loss, and AUC curves (mean over folds)
   * Average confusion matrix heatmap

Outputs include per-fold logs, saved model checkpoints, and overall performance summaries.



## 🎨 Results Visualization

After training, the script:

* Plots mean accuracy, loss, and AUC vs. epochs.
* Displays the average confusion matrix over all folds with custom colormap.
