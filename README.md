# Transformer Model Training with projecte-aina/aguila-7b

This repository contains scripts and resources for training a transformer model using the `projecte-aina/aguila-7b` architecture, enhanced with various training techniques such as QLora, Peft, and EarlyStopping. The model is trained on a Spanish language corpus derived from the RAE Spanish Dictionary.

## Repository Structure

- **Training/**:
  - Contains the main training script to train the model on NVIDIA 2x3090 GPUs.
  - **Data/**: Stores training and validation datasets generated by the training script.
  - **Results/**: Includes the saved model checkpoints after training.

## Corpora

- External corpora used for training can be accessed through the following link:
  - [RAE Spanish Dictionary Corpus](https://github.com/eneko98/RAE-Corpus.git)

## Model Training

The model utilizes the `projecte-aina/aguila-7b` transformer architecture from HuggingFace. Training is performed on NVIDIA 2x3090 GPUs. Key training enhancements include:

- **QLora**: Quantized Layers for Reduced memory.
- **Peft**: Progressive layer freezing for efficiency.
- **EarlyStopping**: To prevent overfitting and optimize training time.

## Model Evaluation

To test the trained model, the `multilingual_testing.py` script in the `Evaluation` folder can be used. The script prompts the model to generate definitions based on input prompts in the following format:

`[BOS] {lang_tag} {word} (POS: {pos}) <definition>`

## Setup and Usage

To set up the training environment and run the training scripts, follow these instructions:

1. **Clone the Repository:**
```
git clone https://github.com/eneko98/Aguila-RAE.git
```
```
cd Aguila-RAE
```

2. **Install Dependencies:**
```
pip install -r requirements.txt
```

3. **Run Training Script:**
```
python rae_training.py
```

4. **Evaluate the Model:**
```
python Evaluation/rae_testing.py
```

5. **Contributing:**
Contributions to this project are welcome. Please fork the repository and submit a pull request to propose changes.