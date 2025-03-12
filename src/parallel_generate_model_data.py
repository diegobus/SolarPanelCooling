import pandas as pd
import numpy as np
from channel_methods import channel
from sympy import *
import time
import multiprocessing




# Load Mojave dataset
df = pd.read_csv("../solar_data/mojave_summer_clear_days.csv")

# Define constants & parameter ranges
Tcoolant_dist = (25, 2.5)  # Coolant inlet temperature (°C), normal distribution
# will need to redefine methods for volumetric flowrate if not using water
mass_flowrate_range = np.linspace(
    0.0001, 0.01, 50
)  # Test flow rates from 0.0001 kg/s to 0.01 kg/s (assuming water)
PV_EFFICIENCY_REF = 0.20  # 20% efficiency (upper bound) at STC conditions
TEMP_COEFF = -0.0045  # -0.45% per °C efficiency loss
T_REF = 298.15  # Reference temperature in Kelvin


def assess_efficiency_increase(T_panel_no_cooling, T_panel_with_cooling, mass_flowrate):
    """Computes efficiency improvement due to cooling with pump cost penalty."""
    # Compute efficiency without cooling
    eta_no_cooling = PV_EFFICIENCY_REF * (1 + TEMP_COEFF * (T_panel_no_cooling - T_REF))
    
    # Compute efficiency with cooling
    eta_with_cooling = PV_EFFICIENCY_REF * (1 + TEMP_COEFF * (T_panel_with_cooling - T_REF))
    
    # Efficiency gain
    efficiency_gain = eta_with_cooling - eta_no_cooling
    
    # Pump cost penalty (λ is tunable)
    lambda_penalty = 0.012  # Adjust this value based on testing
    pump_cost = lambda_penalty * (mass_flowrate ** 1.5)
    
    # Adjusted efficiency gain
    net_efficiency_gain = efficiency_gain - pump_cost
    
    return net_efficiency_gain


def run_experiment(Tamb, I, Tcoolant_in, mass_flowrate, cooling):
    """Runs a single cooling experiment and returns the temperature profile."""

    # print(
    #     f"Running experiment with Tamb={Tamb}°C, I={I} W/m^2, Tcoolant_in={Tcoolant_in}°C, mass_flowrate={mass_flowrate} kg/s, cooling={'On' if cooling else 'Off'}"
    # )

    # Redefine experimental variables in appropriate units
    Tamb = Tamb + 273.15  # Convert to Kelvin
    Tcoolant_in = Tcoolant_in + 273.15  # Convert to Kelvin
    # irradiance already in W/m^2
    # mass flow rate already in kg/s

    panel_experiment = channel(
        T_ambient=Tamb,
        T_fluid_i=Tcoolant_in,
        intensity=I,  # irradiance
        mass_flow_rate=mass_flowrate,
    )

    # Run the experiment and let hit steady state
    if cooling:
        panel_experiment.cool_and_flow_iter()
        return panel_experiment.T_panel_matrix  # Return panel temperature profile
    else:  # No cooling
        panel_experiment.no_flow_steady_state()
        return panel_experiment.no_flow_panel_temp  # Return panel temperature profile


def find_optimal_flowrate(Tamb, I, Tcoolant_in):
    """Runs cooling simulation at multiple flow rates and finds the optimal one."""
    # print(f"Finding optimal flowrate for Tamb={Tamb}°C, I={I} W/m^2, Tcoolant_in={Tcoolant_in}°C")

    best_flowrate = None  
    best_efficiency_gain = -np.inf  # Initialize to negative infinity

    for mass_flowrate in mass_flowrate_range:
        # print(f"Testing mass_flowrate: {mass_flowrate} kg/s")

        # Simulate no cooling and cooling
        T_panel_no_cooling = run_experiment(Tamb, I, Tcoolant_in, mass_flowrate, cooling=False)
        T_panel_with_cooling = run_experiment(Tamb, I, Tcoolant_in, mass_flowrate, cooling=True)

        # Compute net efficiency gain (accounting for pump energy cost)
        net_efficiency_gain = assess_efficiency_increase(
            T_panel_no_cooling, 
            np.average(T_panel_with_cooling), 
            mass_flowrate
        )

        # print(f"Net efficiency gain at {mass_flowrate:.6f} kg/s: {net_efficiency_gain:.4f}")

        # Select flowrate that maximizes net efficiency gain
        if net_efficiency_gain > best_efficiency_gain:
            best_efficiency_gain = net_efficiency_gain
            best_flowrate = mass_flowrate

    # print(f"Optimal flowrate found: {best_flowrate:.6f} kg/s with net efficiency gain {best_efficiency_gain:.4f}")
    if best_efficiency_gain < 0:
        best_flowrate = None
    return Tamb, I, Tcoolant_in, best_flowrate, best_efficiency_gain, T_panel_no_cooling, np.average(T_panel_with_cooling)


# Generate dataset
num_samples = len(df)  # Use all available environmental data

# Storage for results
data = []

# Limit to the first few rows for testing
num_samples_to_test = num_samples


def process_sample(index, row):
    Tamb = row["air_temperature"]
    I = row["ghi"]
    Tcoolant_in = np.random.normal(
        *Tcoolant_dist
    )  # Randomly sample coolant inlet temp from normal distribution

    # Log progress
    print(
        f"------- Processing sample {index + 1}/{num_samples_to_test} -------\nTamb: {Tamb:.3g}°C, I: {I:.3g} W/m^2, Tcoolant_in: {Tcoolant_in:.3g}°C"
    )
    start_time = time.time()

    result = find_optimal_flowrate(Tamb, I, Tcoolant_in)

    # Indicate sample processing completion
    end_time = time.time()
    print(
        f"*  *  *  Finished processing sample {index + 1}/{num_samples_to_test} in {end_time - start_time:.2f} seconds *  *  *\nbest_flowrate: {result[3]:.6f} kg/s, best_efficiency_gain: {result[4]:.4f}"
    )

    return result


if __name__ == "__main__":
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap(
            process_sample,
            [
                (index, row)
                for index, row in df.iterrows()
                if index < num_samples_to_test
            ],
        )

    # Collect results
    data.extend(results)

    # Convert to DataFrame and save to CSV
    final_df = pd.DataFrame(
        data, columns=["Tamb", "I", "Tcool_in", "mass_flowrate", "eff_gain", "Tp_no_cool", "Tp_cool"]
    )
    final_df.to_csv("../solar_data/cooling_data_complete.csv", index=False)

    print(
        "------- Test dataset generation complete. Saved as cooling_data_complete.csv in solar_data/ -------"
    )
