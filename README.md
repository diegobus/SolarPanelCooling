# Solar Panel Cooling Optimization

## Project Overview

This project models and optimizes cooling systems for solar panels to mitigate efficiency losses caused by high operating temperatures. Solar panels experience reduced efficiency as temperature increases, with typical efficiency decreases of 0.45% per °C above reference temperature 25°C. By implementing active cooling strategies, we can maintain higher panel efficiencies while accounting for the energy cost of pumping coolant.

The project has two main components:
1. Data generation: Simulates solar panel behavior under various environmental conditions and calculates optimal flow rates
2. Model training: Uses neural network to predict optimal coolant flow rates based on environmental inputs

### Repository Layout and Key Files

- `config` folder: contains configuration files
  - `config_template.py`: Configuration file for sensitive data like NERL API key and email. Create a `.gitignore` file to hide the file after renaming to `config.py`.
  - `requirements.txt`: List of dependencies required for the project to run

- `model` folder: contains final model weights and evaluation metrics

- `solar_data` folder: contains data for model training
  - `mojave_summer_clear_days.csv`: Solar data for Mojave desert during clear days from NREL's NSRDB (National Solar Radiation Database). Can be generated using `get_solar_data.py`
  - `cooling_data_complete.csv`: Cooling data used to train model generated from cooling experiments using `parallel_generate_model_data.py`

- `src` folder: contains source code for project
  - `get_solar_data.py`: Acquires solar irradiance and temperature data for Mojave desert from NREL's NSRDB (National Solar Radiation Database)
  - `channel_methods.py`: Core physics-based simulation of solar panel cooling
    - Implements heat transfer models for panel and coolant
    - Calculates panel temperatures and efficiency under different conditions
  - `generate_model_data.py`: Single-process data generation
    - Simulates panel behavior across environmental conditions
    - Finds optimal flow rates that maximize efficiency while minimizing pumping costs
  - `parallel_generate_model_data.py`: Multi-process data generation
    - Same functionality as non-parallel version but utilizes all CPU cores
    - Significantly faster for large dataset generation
  - `model.py`: Neural network model for predicting optimal flow rates
    - Preprocesses data with standardization
    - Implements a simple neural network with regularization
    - Evaluates model performance and generates visualizations

## Required Dependencies

Activate a conda environment with Python 3.10 or earlier. The following dependencies are required:
```
pandas>=1.1.0
matplotlib>=3.3.0
sympy>=1.6.0
tensorflow>=2.4.0
scikit-learn>=0.23.0
```

You can install the necessary dependencies in a conda enviroment with:
```
pip install -r requirements.txt
```

## Data Preparation

### Acquiring Solar Data
1. Run `get_solar_data.py` to download solar irradiance and temperature data for Mojave desert from NREL's NSRDB (National Solar Radiation Database). NOTE:
this script depends on NERL API key. Go to the NERL website to request one, then fill out `config_template.py`and rename to `config.py`.
2. Data is saved to `solar_data/` directory. The primary dataset used is `mojave_summer_clear_days.csv` which contains hourly measurements of:
   - `air_temperature`: Ambient air temperature (°C)
   - `ghi`: Global horizontal irradiance (W/m^2)


## Running Instructions

### Data Generation

To generate the training data:
```
python parallel_generate_model_data.py
```
This will simulate solar panel performance with various cooling conditions and find optimal flow rates. The parallel version utilizes multiprocessing to speed up computation.

For a non-parallel version (slower but easier to debug):
```
python generate_model_data.py
```

### Model Training

To train the neural network model:
```
python model.py
```

This will:
1. Load the generated dataset
2. Train a neural network to predict optimal coolant flow rates
3. Evaluate the model using R² score
4. Generate plots for model performance
5. Save the trained model weights to `nn_weights.csv`


### Data Flow

1. Solar data → Data generation scripts → Training datasets
2. Training datasets → Neural network model → Trained weights
3. Trained weights can be used for real-time flow rate prediction

## Ensuring Reproducibility

To reproduce the results:

1. Use the exact datasets provided or follow the data preparation steps precisely
2. Maintain the same random seeds used in code (set to 42 for train-test splits)
3. Run the code with the dependencies at specified versions
4. Use the same hyperparameters for the neural network model

The model training includes early stopping to prevent overfitting, which may cause slight variations in results due to the stochastic nature of neural network training. For full reproducibility, you can set the global random seed for TensorFlow by adding the following to the top of `model.py`:

```python
import tensorflow as tf
import numpy as np
import random

# Set global random seeds
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)
```

## Results and Validation

The model accuracy can be assessed through:
- R² score reported during training
- Visual inspection of the actual vs. predicted plots
- Validation loss curves showing convergence

The current model achieves competitive R² scores, indicating a good fit between predicted and actual optimal flow rates.