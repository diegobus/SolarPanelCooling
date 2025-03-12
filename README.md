# Solar Panel Cooling Optimization

## 1. Project Overview

This project models and optimizes cooling systems for solar panels to mitigate efficiency losses caused by high operating temperatures. Solar panels experience reduced efficiency as temperature increases, with typical efficiency decreases of 0.45% per °C above reference temperature 25°C. By implementing active cooling strategies, we can maintain higher panel efficiencies while accounting for the energy cost of pumping coolant.

The project has two main components:
1. Data generation: Simulates solar panel behavior under various environmental conditions and calculates optimal flow rates
2. Model training: Uses neural network to predict optimal coolant flow rates based on environmental inputs

## 2. Repository Layout and Key Files

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

## 3. Required Dependencies

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

## 4. Data Workflow

### Environmental Data Acquisition
1. Run `get_solar_data.py` to download solar irradiance and temperature data for Mojave desert from NREL's NSRDB (National Solar Radiation Database). NOTE:
this script depends on NERL API key. Go to the NERL website to request one, then fill out `config_template.py`and rename to `config.py`.
2. Data is saved to `solar_data/` directory. The primary dataset used is `mojave_summer_clear_days.csv` which contains hourly measurements of:
   - `air_temperature`: Ambient air temperature (°C)
   - `ghi`: Global horizontal irradiance (W/m^2)

### Cooling Data Generation

To generate the training data, navigate to `src/` and run:
```
python parallel_generate_model_data.py
```
This will simulate solar panel performance with various cooling conditions and find optimal flow rates for each set of enviromental conditions and a cooling fluid initial temperature randomly sampled from normal distribution (μ=25, σ=2.5). The parallel version utilizes multiprocessing to speed up computation.

For a non-parallel version (slower but easier to debug):
```
python generate_model_data.py
```
For parallel and non-parallel versions, you can adjust `num_samples` to only run a few samples from dataset. You can also uncomment print statements to watch each experiment run.

Model training data is saved to `solar_data/cooling_data_complete.csv`.


## 5. Model Training and Evaluation

### Running the Model
To train the neural network model, run:
```bash
python model.py
```

### Training Process

1. **Load Dataset**
   - Reads `large_cooling_data_complete.csv`
   - Features: `Tamb` (ambient temperature), `I` (solar irradiance), `Tcool_in` (coolant inlet temp)
   - Target: `mass_flowrate` (optimal coolant flow rate)

2. **Preprocess Data**
   - Splits into 80% training, 20% testing
   - Standardizes input features and target

3. **Train Neural Network**
   - **Architecture**:
     - Input: 3 features
     - Hidden: 3 neurons, ReLU activation, L2 regularization
     - Dropout: 10% to reduce overfitting
     - Output: Single neuron, linear activation
   - **Training Parameters**:
     - Optimizer: Adam
     - Loss: Mean Squared Error (MSE)
     - Metric: Mean Absolute Error (MAE)
     - Early stopping: Stops if validation loss doesn't improve after 10 epochs

4. **Save Model Weights**
   - Extracts and saves weights to `nn_weights.csv`

5. **Evaluate Performance**
   - Computes **R² score**
   - Clamps negative predictions to zero (physical constraint)
   - Generates:
     - **Actual vs. Predicted Flow Rate Plot**
     - **Training Loss Curve**

### Results

- **Actual vs. Predicted Flow Rate**: Scatter plot comparing predictions to actual values
- **Training Loss Curve**: Tracks training/validation loss over epochs

### Example Output
```bash
Neural network weights saved to nn_weights.csv
Test R^2 Score: 0.9419
```
After training, weights and evaluation plots are saved.

