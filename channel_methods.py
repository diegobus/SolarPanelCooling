import numpy as np
import matplotlib.pyplot as plt
from sympy import *
T = symbols("T")


class channel:
    """Constructor for the channel class"""

    def __init__(
        self,
        T_fluid_i=290,
        T_ambient=330,
        T_outside=298,
        panel_dimensions=(1, 0.02, 0.01),
        channel_height=0.02,
        mass_flow_rate=0.001,
        fluid_specific_heat=4186,
        fluid_density=1000,
        h_fluid=70,
        h_amb=10,
        p_k=0.3,
        intensity=1000,
        x_steps=100,
        flow_forward=True,
    ):
        # Constants
        self.sigma = 5.67e-8  # Stefan-Boltzmann constant (W/m^2·K^4)

        # Dimensions (in meters)
        self.panel_length, self.panel_width, self.panel_thickness = panel_dimensions
        self.x_length = self.panel_length / x_steps  # Length of each block (m)
        self.x_steps = x_steps  # Number of steps in the x-direction
        y_length = self.panel_width  # Width of the block (m)
        self.A = self.x_length * y_length  # Surface area of the block (m^2)
        self.block_volume = self.A * channel_height  # Volume of the block (m^3)
        self.A_across = (
            self.x_length * self.panel_thickness
        )  # Area across the block (m^2)
        self.panel_length_array = np.linspace(
            0, self.panel_length, x_steps
        )  # x-coordinates of the blocks
        self.T_fluid_matrix = (
            np.ones(x_steps) * T_fluid_i
        )  # Initial fluid temperature for each block
        self.T_panel_matrix = (
            np.ones(x_steps) * T_ambient
        )  # Initial panel temperature for each block (using ambient temperature as lower bound)

        # Environment parameters
        self.G = intensity  # Solar irradiance (W/m^2)
        self.h = h_amb  # Convective heat transfer coefficient (W/m^2·K)
        self.mass_flow_rate = mass_flow_rate  # Mass flow rate (kg/s)
        self.T_ambient = T_ambient  # Ambient temperature (K)
        self.T_outside = T_ambient  # Temp on non-sun facing side of panel for SS no cooling

        # Panel Properties
        self.p_specific_heat = 700  # Specific heat capacity (J/kg·K)
        self.p_k = p_k  # Thermal conductivity (W/m·K)
        self.p_alpha = 0.85  # Absorptivity
        self.p_epsilon = 0.85  # Emissivity
        self.p_density = 2300  # kg/m^3, assumed to be the density of aluminum
        self.m = (
            self.p_density * self.panel_thickness * self.A
        )  # Mass of the panel per block (kg)

        # Fluid properties
        self.T_fluid_i = T_fluid_i  # Initial fluid temperature (K)
        self.flow_forward = flow_forward  # True if flow is from left to right
        self.h_fluid = h_fluid  # Heat transfer coefficient with cooling fluid (W/m^2·K)
        self.fluid_specific_heat = (
            fluid_specific_heat  # Specific heat capacity of water (J/kg·K)
        )
        self.fluid_speed = mass_flow_rate / (
            fluid_density * y_length * channel_height
        )  # Fluid speed (m/s)
        self.fluid_mass = (
            fluid_density * self.block_volume
        )  # Mass of the cooling fluid (kg)

        # Time parameters
        self.num_timesteps = 20
        self.time_duration = self.fluid_mass / mass_flow_rate  # Total time (seconds)
        self.time_step = self.time_duration / self.num_timesteps  # Time step (seconds)

    def panel_heat_transfer_rate(self, T, T_fluid_current):  # in K/s
        """Calculate the net heat transfer rate (dT/dt) for the panel"""
        absorbed = self.p_alpha * self.G * self.A  # Absorbed solar radiation (W)
        # assume no radiative loss
        radiative_loss = (
            0  # self.p_epsilon * self.sigma * self.A * (self.T_ambient**4 - T**4)
        )
        convective_gain = self.h * self.A * (self.T_ambient - T)
        fluid_cooling_loss = self.h_fluid * self.A * (T - T_fluid_current)
        net_heat_transfer = (
            absorbed + convective_gain - fluid_cooling_loss - radiative_loss
        )
        return net_heat_transfer / (self.m * self.p_specific_heat)

    def fluid_heat_transfer_rate(self, T, T_fluid_current):  # in K/s
        """Calcluate the heat transfer rate to the fluid"""
        heat_transfer_to_fluid = self.h_fluid * self.A * (T - T_fluid_current)
        return heat_transfer_to_fluid / (self.fluid_mass * self.fluid_specific_heat)

    def cool(self, T_initial, T_fluid):
        """Calculate the temperature of the panel and the cooling fluid of one block after self.num_timesteps of heat transfer from the environment and fluid"""
        # Initial conditions
        T = T_initial  # Current temperature (K)
        T_fluid_current = T_fluid  # Current fluid temperature (K)

        # Arrays to store results
        temperatures = [T_initial]
        fluid_temperatures = [T_fluid_current]

        # Numerical solution using Euler's method
        for _ in range(self.num_timesteps):
            dTdt = self.panel_heat_transfer_rate(T, T_fluid_current)
            dT_fluid_dt = self.fluid_heat_transfer_rate(T, T_fluid_current)
            T += dTdt * self.time_step
            T_fluid_current += dT_fluid_dt * self.time_step
        return [T, T_fluid_current]

    def flow(self, flow_T=None):
        """Shift fluid temperature array and update the first element with the initial temperature"""
        if self.flow_forward:
            self.T_fluid_matrix[1:] = self.T_fluid_matrix[:-1]
            self.T_fluid_matrix[0] = self.T_fluid_i
            if flow_T is not None:
                self.T_fluid_matrix[0] = flow_T
        else:
            self.T_fluid_matrix[:-1] = self.T_fluid_matrix[1:]
            self.T_fluid_matrix[-1] = self.T_fluid_i
            if flow_T is not None:
                self.T_fluid_matrix[-1] = flow_T

    def cool_and_flow(self, flow_T=None):
        """Cool and flow the fluid in whole channel for one iteration"""
        for i in range(len(self.T_fluid_matrix)):
            self.T_panel_matrix[i], self.T_fluid_matrix[i] = self.cool(
                self.T_panel_matrix[i], self.T_fluid_matrix[i]
            )
        self.flow(flow_T)

    def cool_and_flow_iter(self, max_iter=1000, tol=0.1):
        """Cool and flow the fluid in whole channel until steady state or max iterations reached
        
        Args:
            max_iter (int): Maximum number of iterations to run
            tol (float): Temperature difference tolerance to determine steady state
            
        Returns:
            int: Number of iterations performed to reach steady state
        """
        # If no cooling, just return intial temp profile (assumes panel same temp as t_amb) but
        # if cooling, let system reach steady state
        prev_temps = self.T_panel_matrix.copy()
        
        for i in range(max_iter):
            self.cool_and_flow()
            
            # Check if steady state reached by comparing temperature changes
            temp_diff = np.abs(self.T_panel_matrix - prev_temps).max()
            if temp_diff < tol:
                return i + 1  # Return number of iterations needed
                
            prev_temps = self.T_panel_matrix.copy()
            
        return max_iter  # Return max_iter if steady state not reached

    def no_flow_steady_state(self):
        """Calculate the steady state temperature of the panel with no flow"""
        # h = 5
        # solve resistance circuit type equation for T of panel
        self.no_flow_panel_temp = solve(
            self.p_alpha * self.G
            + self.sigma * self.p_epsilon * ((self.T_ambient) ** 4 - T**4)
            + self.h * (self.T_ambient - T)
            - self.sigma * self.p_epsilon * (T**4 - (self.T_outside) ** 4)
            - self.h * (T - self.T_outside),
            T,
        )

        # take positive real value from solution
        self.no_flow_panel_temp = [
            i.evalf() for i in self.no_flow_panel_temp if i.is_real and i > 0
        ][0]
        return self.no_flow_panel_temp

    def diffuse(self):
        # Discretization
        alpha = 1.5e-6 * self.time_step / self.x_length**2
        # alpha = D * self.time_step / self.x_length**2 # Stability parameter (alpha = D * dt / dx^2)

        # Create an array to store concentration at each time step
        T_time = np.zeros((self.num_timesteps, self.x_steps))
        T = self.T_fluid_matrix.copy()
        T_time[0, :] = T  # Set initial condition

        # Time stepping loop (FTCS scheme)
        for n in range(1, self.num_timesteps):
            T_new = T.copy()  # Create a new array for the next time step
            for i in range(1, self.x_steps - 1):
                T_new[i] = T[i] + alpha * (
                    T[i + 1] - 2 * T[i] + T[i - 1]
                )  # FTCS update rule
            T = T_new
            T_time[n, :] = T  # Store the updated concentration
        self.T_fluid_matrix = T_time[-1, :]

    def plot(self):
        plt.figure(figsize=(6, 4))
        plt.plot(
            self.panel_length_array, self.T_panel_matrix, label="Surface Temperature"
        )
        plt.plot(
            self.panel_length_array,
            self.T_fluid_matrix,
            label="Cooling Fluid Temperature",
            linestyle="--",
        )
        plt.axhline(
            self.T_ambient, color="r", linestyle="--", label="Ambient Temperature"
        )
        plt.title("Transient Temperature of Solar Panel and Cooling Fluid")
        plt.xlabel("Panel x-Coordinate (m)")
        plt.ylabel("Temperature (K)")
        plt.legend()
        plt.grid()
        plt.show()

    def deltaT(m, c, A, h, dt, thi, tci):
        return float(h * A / m / c * (thi - tci) * dt)

    def diffusion(T_panel1, T_panel2, A, m, k, c, dt, dx):
        T_panel1_new, T_panel2_new = T_panel1.copy(), T_panel2.copy()
        for i in range(len(T_panel1)):
            q = k * A * (T_panel1[i] - T_panel2[i]) * dt / dx
            T_panel1_new[i] -= q / m / c
            T_panel2_new[i] += q / m / c
        return T_panel1_new, T_panel2_new
