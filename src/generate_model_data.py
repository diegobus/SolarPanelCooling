import pandas as pd
import numpy as np
from channel_methods import channel

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


def assess_efficiency_increase(T_panel_no_cooling, T_panel_with_cooling):
    """Computes efficiency improvement due to cooling."""

    print(f"T_panel_no_cooling: {T_panel_no_cooling}, T_panel_with_cooling: {T_panel_with_cooling}")
    eta_no_cooling = PV_EFFICIENCY_REF * (1 - TEMP_COEFF * (T_REF - T_panel_no_cooling))
    eta_with_cooling = PV_EFFICIENCY_REF * (
        1 - TEMP_COEFF * (T_REF - T_panel_with_cooling)
    )

    efficiency_gain = eta_with_cooling - eta_no_cooling
    return efficiency_gain


def run_experiment(Tamb, I, Tcoolant_in, mass_flowrate, cooling):
    """Runs a single cooling experiment and returns the temperature profile."""

    # print(f"Running experiment with Tamb={Tamb}°C, I={I} W/m^2, Tcoolant_in={Tcoolant_in}°C, mass_flowrate={mass_flowrate} kg/s, cooling={'On' if cooling else 'Off'}")

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
        panel_experiment.cool_and_flow_iter(1000)
    else:  # No cooling
        panel_experiment.no_flow_steady_state()
        return panel_experiment.no_flow_panel_temp  # Return panel temperature profile

    # Return the temperature profile
    return panel_experiment.T_panel_matrix


def find_optimal_flowrate(Tamb, I, Tcoolant_in):
    """Runs cooling simulation at multiple flow rates and finds the optimal one."""
    best_flowrate = 0
    best_efficiency_gain = 0
    efficiency_threshold = 0.05  # Require at least 5% efficiency gain

    for mass_flowrate in mass_flowrate_range:
        #print(f"Testing mass flowrate: {mass_flowrate} kg/s")

        # simulate no cooling and cooling
        T_panel_no_cooling = run_experiment(
            Tamb, I, Tcoolant_in, mass_flowrate, cooling=False
        )
        T_panel_with_cooling = run_experiment(
            Tamb, I, Tcoolant_in, mass_flowrate, cooling=True
        )

        # for ease, use min temperature of panel
        net_efficiency_gain = assess_efficiency_increase(
            T_panel_no_cooling, np.average(T_panel_with_cooling)
        )  # Compute efficiency improvement

        #print(f"Efficiency gain: {net_efficiency_gain}")

        # Select flowrate that maximizes net efficiency gain
        if net_efficiency_gain > best_efficiency_gain:
            best_efficiency_gain = net_efficiency_gain
            best_flowrate = mass_flowrate

        # # If efficiency decreases, fluid is warming panel, return zero flow
        # if efficiency_gain < 0:
        #     best_flowrate = 0
        #     break
        # # If efficiency gain is less than previous, return previous best flowrate
        # elif efficiency_gain < prev_efficiency_gain:
        #     break
        # # if efficiency increase sufficient, return flowrate
        # elif efficiency_gain >= efficiency_threshold:
        #     best_flowrate = mass_flowrate
        #     break

        # best_flowrate = mass_flowrate
        # prev_efficiency_gain = efficiency_gain

    return best_flowrate


# Generate dataset
num_samples = len(df)  # Use all available environmental data

# Storage for results
data = []

# Limit to the first few rows for testing
num_samples_to_test = 2

for index, row in df.iterrows():
    # for debugging with first few rows
    if index >= num_samples_to_test:
        break

    Tamb = row["air_temperature"]
    I = row["ghi"]
    Tcoolant_in = np.random.normal(
        *Tcoolant_dist
    )  # Randomly sample coolant inlet temp from normal distribution

    # Log progress
    print(f"------- Processing sample {index + 1}/{num_samples_to_test} -------")

    mass_flowrate_optimal = find_optimal_flowrate(Tamb, I, Tcoolant_in)

    data.append([Tamb, I, Tcoolant_in, mass_flowrate_optimal])

# Convert to DataFrame and save to CSV
final_df = pd.DataFrame(
    data, columns=["Tamb", "I", "Tcoolant_in", "mass_flowrate_optimal"]
)
final_df.to_csv("../solar_data/solar_cooling_training_data_test.csv", index=False)

print(
    "------- Test dataset generation complete. Saved as solar_cooling_training_data_test.csv -------"
)