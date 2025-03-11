# Deep Denoising Diffusion Probabilistic Model

This repository contains a refactored implementation of the Denoising Diffusion Probabilistic Model (DDPM) for image generation.

## Project Structure

The project has been simplified to have just two main files:

1. `main_flow.py` - Contains the main execution flow for training and evaluating the model
2. `utils.py` - Contains all the utility functions, model definitions, and helper classes

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Make sure the `score` directory is in your Python path or in the same directory as the code.

## Usage

To train the model, edit `main_flow.py` and set:

```python
train_mode = True
```

To evaluate the model, set:

```python
eval_mode = True
```

Then run:

```bash
python main_flow.py
```

## Configuration

All configuration parameters are defined at the top of `main_flow.py`. You can modify these parameters to change the model architecture, training settings, and evaluation metrics.

## Model Architecture

The model is based on the UNet architecture with time embeddings, as described in the DDPM paper. The diffusion process uses a Gaussian noise schedule.

## Evaluation

The model is evaluated using Inception Score (IS) and Fréchet Inception Distance (FID) metrics.

## License

This code is provided for research purposes only.
