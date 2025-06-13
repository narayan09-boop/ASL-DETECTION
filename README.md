# ASL Detection - American Sign Language Image Classifier

This project is a machine learning pipeline for detecting American Sign Language (ASL) alphabets, numbers, and special signs (`nothing`, `space`, `del`) from images. The model is trained on labeled ASL hand gesture images and predicts the corresponding character for a given input image.

## Features

- Classifies ASL alphabets (A-Z), numbers, `nothing`, `space`, and `del`
- Simple Convolutional Neural Network (CNN) architecture
- Easy-to-use prediction script for new images
- Exploratory Data Analysis (EDA) notebook for dataset insights

## Project Structure

```
ASL DETECTION/
│
├── data/
│   ├── raw/         # Original images organized by class
│   └── processed/   # Preprocessed images 
│
├── asl_predict_image.py            # Script to predict ASL from an image
├── asl_model_basic.h5              # Trained model file (after training)
├── requirements.txt                # Python dependencies
└── README.md
```

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/narayan09-boop/asl-detection.git
cd asl-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset

- Download the dataset from `https://www.kaggle.com/datasets/grassknoted/asl-alphabet`
- Place your ASL images in the `data/raw/` directory, organized in subfolders by class (e.g., `A`, `B`, ..., `Z`, `nothing`, `space`, `del`).

## NOTE - Only download the data set if you want to train the model on your own.

### 4. Train the Model

```bash
python src/train.py
```
- The trained model will be saved as `asl_model_basic.h5`.

### 5. Predict on New Images

```bash
python asl_predict_image.py
```
- Enter the path to your image when prompted.

## Example

```
👉 Enter image path to predict: test_images/sample_A.jpg
🧠 Prediction: A (98.45% confidence)
```

## Requirements

- Python 3.7+
- TensorFlow
- NumPy
- Matplotlib
- Pillow

(See `requirements.txt` for full list.)

## License

This project is licensed under the MIT License.

---

**Contributions are
