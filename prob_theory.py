import streamlit as st
import json
import math
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import time
import random
import tempfile

class DiscreteRandomVariable:
    def __init__(self, values_probs: Dict[float, float]):
        # Checking all keys are floats
        self.values_probs = {float(key): float(value) for key, value in values_probs.items()}
        self._validate_probabilities()
        print(f"DiscreteRandomVariable initialized with {len(values_probs)} values")
    
    def _validate_probabilities(self):
        total_prob = sum(self.values_probs.values())
        if not math.isclose(total_prob, 1.0, rel_tol=1e-9):
            raise ValueError(f"Sum of probabilities must equal 1.0, got: {total_prob}")
        print("Probability validation passed - sum equals 1.0")
    
    def multiply_by_constant(self, constant: float) -> 'DiscreteRandomVariable':
        print(f"Multiplying random variable by constant: {constant}")
        new_values_probs = {value * constant: prob 
                          for value, prob in self.values_probs.items()}
        return DiscreteRandomVariable(new_values_probs)
    
    def __add__(self, other: 'DiscreteRandomVariable') -> 'DiscreteRandomVariable':
        print("Adding two discrete random variables")
        new_values_probs = {}
        
        for value1, prob1 in self.values_probs.items():
            for value2, prob2 in other.values_probs.items():
                new_value = value1 + value2
                new_prob = prob1 * prob2
                
                if new_value in new_values_probs:
                    new_values_probs[new_value] += new_prob
                else:
                    new_values_probs[new_value] = new_prob
        
        return DiscreteRandomVariable(new_values_probs)
    
    def __mul__(self, other: 'DiscreteRandomVariable') -> 'DiscreteRandomVariable':
        print("Multiplying two discrete random variables")
        new_values_probs = {}
        
        for value1, prob1 in self.values_probs.items():
            for value2, prob2 in other.values_probs.items():
                new_value = value1 * value2
                new_prob = prob1 * prob2
                
                if new_value in new_values_probs:
                    new_values_probs[new_value] += new_prob
                else:
                    new_values_probs[new_value] = new_prob
        
        return DiscreteRandomVariable(new_values_probs)
    
    # Mathematical characteristics
    def expected_value(self) -> float:
        return sum(value * prob for value, prob in self.values_probs.items())
    
    def variance(self) -> float:
        mean = self.expected_value()
        return sum(prob * (value - mean) ** 2 for value, prob in self.values_probs.items())
    
    def standard_deviation(self) -> float:
        return math.sqrt(self.variance())
    
    def skewness(self) -> float:
        mean = self.expected_value()
        std = self.standard_deviation()
        if std == 0:
            return 0
        return sum(prob * ((value - mean) / std) ** 3 for value, prob in self.values_probs.items())
    
    def kurtosis(self) -> float:
        mean = self.expected_value()
        std = self.standard_deviation()
        if std == 0:
            return 0
        return sum(prob * ((value - mean) / std) ** 4 for value, prob in self.values_probs.items()) - 3
    
    # Serialization methods
    def to_dict(self) -> Dict:
        serialized = {"values_probs": self.values_probs}
        print("DiscreteRandomVariable serialized to dictionary")
        return serialized
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DiscreteRandomVariable':
        print("DiscreteRandomVariable deserialized from dictionary")
        # Convert all keys and values to float for correct types
        values_probs = {float(key): float(value) for key, value in data["values_probs"].items()}
        return cls(values_probs)
    
    def save_to_file(self, filename: str):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"DiscreteRandomVariable saved to file: {filename}")
    
    @classmethod
    def load_from_file(cls, filename: str) -> 'DiscreteRandomVariable':
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"DiscreteRandomVariable loaded from file: {filename}")
        return cls.from_dict(data)
    
    # Visualization methods
    def get_distribution_law(self) -> List[Tuple[float, float]]:
        """Get distribution law as list of tuples (value, probability)"""
        distribution = list(self.values_probs.items())
        print(f"Distribution law retrieved: {distribution}")
        return distribution
    
    def plot_distribution_law(self):
        """Plot distribution law as stem plot"""
        values = self.get_values()
        probs = self.get_probabilities()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.stem(values, probs, basefmt=" ")
        ax.set_xlabel('Values')
        ax.set_ylabel('Probability')
        ax.set_title('Distribution Law (Stem Plot)')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_polyline(self):
        """Plot polyline connecting probability points"""
        values = self.get_values()
        probs = self.get_probabilities()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot polyline (connecting points with lines)
        ax.plot(values, probs, 'o-', linewidth=2, markersize=8, label='Polyline')
        
        # Also show points clearly
        ax.scatter(values, probs, s=50, color='red', zorder=5)
        
        ax.set_xlabel('Values')
        ax.set_ylabel('Probability')
        ax.set_title('Probability Polyline')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return fig
    
    def plot_distribution_function(self):
        """Plot cumulative distribution function"""
        sorted_items = sorted(self.values_probs.items())
        values = [item[0] for item in sorted_items]
        probabilities = [item[1] for item in sorted_items]
        
        # Create CDF
        cdf_values = []
        cdf_probs = []
        cumulative = 0
        
        for value, prob in sorted_items:
            cdf_values.extend([value, value])
            cdf_probs.extend([cumulative, cumulative + prob])
            cumulative += prob
        
        # Add final point
        cdf_values.append(values[-1] + 1)
        cdf_probs.append(1.0)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.step(cdf_values, cdf_probs, where='post', linewidth=2)
        ax.set_xlabel('Values')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('Cumulative Distribution Function')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        return fig
    
    # Getters
    def get_values(self) -> List[float]:
        return list(self.values_probs.keys())
    
    def get_probabilities(self) -> List[float]:
        return list(self.values_probs.values())

class RandomWalkSimulator:
    def __init__(self):
        self.step_distribution = None
        self.initial_position = 0
        self.n_steps = 10
        self.current_step = 0
        self.current_position = 0
        self.is_running = False
        self.trajectory = []
        
    def set_parameters(self, distribution: DiscreteRandomVariable, initial_pos: float, steps: int):
        self.step_distribution = distribution
        self.initial_position = initial_pos
        self.n_steps = steps
        self.reset()
        
    def reset(self):
        self.current_step = 0
        self.current_position = self.initial_position
        self.is_running = False
        self.trajectory = [self.initial_position]
        
    # Controlling methods
    def start(self):
        self.is_running = True
        
    def stop(self):
        self.is_running = False
        
    # Walking methods
    def perform_step(self):
        if not self.is_running or self.current_step >= self.n_steps:
            return False
            
        # Randomly select step value based on probabilities
        values = self.step_distribution.get_values()
        probabilities = self.step_distribution.get_probabilities()
        step_value = random.choices(values, weights=probabilities, k=1)[0]
        
        # Update position
        self.current_position += step_value
        self.current_step += 1
        self.trajectory.append(self.current_position)
        
        return True
    
    def calculate_final_distribution(self):
        if not self.step_distribution:
            return None
            
        dp = {0: {self.initial_position: 1.0}}
        
        for step in range(1, self.n_steps + 1):
            dp[step] = {}
            for pos, prob in dp[step-1].items():
                for step_val, step_prob in self.step_distribution.values_probs.items():
                    new_pos = pos + step_val
                    new_prob = prob * step_prob
                    if new_pos in dp[step]:
                        dp[step][new_pos] += new_prob
                    else:
                        dp[step][new_pos] = new_prob
        
        return dp[self.n_steps]
    
    def plot_trajectory(self):
        """Plot the trajectory of the random walk"""
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Plot trajectory
        steps = list(range(len(self.trajectory)))
        ax.plot(steps, self.trajectory, 'o-', linewidth=2, markersize=4)
        ax.set_xlabel('Step')
        ax.set_ylabel('Position')
        ax.set_title(f'Random Walk Trajectory (Current: {self.current_position:.2f})')
        ax.grid(True, alpha=0.3)
        
        # Mark initial and current position
        ax.axhline(y=self.initial_position, color='green', linestyle='--', alpha=0.7, label='Initial')
        if len(self.trajectory) > 1:
            ax.axhline(y=self.trajectory[-1], color='red', linestyle='--', alpha=0.7, label='Current')
        
        ax.legend()
        return fig

def format_probability(prob):
    """Format probability in readable form"""
    if prob == 0:
        return "0"
    elif prob < 0.0001:
        return f"{prob:.2e}"  # Scientific notation for very small probabilities
    elif prob < 0.01:
        return f"{prob:.8f}"  # More decimal places for small probabilities
    else:
        return f"{prob:.6f}"  # Standard format

def main():
    st.title("Discrete Random Variable & Random Walk")
    
    # Navigation
    st.sidebar.title("Navigation")
    section = st.sidebar.radio(
        "Go to:",
        ["DSV Overview & Serialization", "Operations", "Visualizations", "Random Walk Simulation"]
    )
    
    # Initialize session state for loaded DSV
    if 'loaded_drv' not in st.session_state:
        st.session_state.loaded_drv = None
    
    # Default example DRV
    if 'example_drv' not in st.session_state:
        st.session_state.example_drv = DiscreteRandomVariable({
            1: 0.1, 2: 0.2, 3: 0.3, 4: 0.25, 5: 0.15
        })
    
    # Initialize random walk simulator
    if 'rw_simulator' not in st.session_state:
        st.session_state.rw_simulator = RandomWalkSimulator()
    
    # Section 1: DSV Overview & Serialization
    if section == "DSV Overview & Serialization":
        st.header("DSV Overview & Serialization")
        
        # Use loaded DSV if available, otherwise use example
        current_drv = st.session_state.loaded_drv if st.session_state.loaded_drv else st.session_state.example_drv
        
        # Current DSV display
        st.subheader("Current Discrete Random Variable")
        distribution = current_drv.get_distribution_law()
        table_data = {
            "Value (x)": [pair[0] for pair in distribution],
            "Probability P(X=x)": [pair[1] for pair in distribution]
        }
        st.table(table_data)
        
        # Mathematical characteristics
        st.subheader("Mathematical Characteristics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Expected Value:** {current_drv.expected_value():.4f}")
            st.write(f"**Variance:** {current_drv.variance():.4f}")
            st.write(f"**Standard Deviation:** {current_drv.standard_deviation():.4f}")
        
        with col2:
            st.write(f"**Skewness:** {current_drv.skewness():.4f}")
            st.write(f"**Kurtosis:** {current_drv.kurtosis():.4f}")
        
        # Serialization section with file dialogs
        st.subheader("Serialization / Deserialization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Save to File")
            
            # File name input with default suggestion
            default_filename = f"dsv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filename = st.text_input("File name:", value=default_filename)
            
            if st.button("Save DSV to File"):
                if filename:
                    try:
                        # Use file downloader for saving
                        json_data = json.dumps(current_drv.to_dict(), indent=2)
                        st.download_button(
                            label="Click to download file",
                            data=json_data,
                            file_name=filename,
                            mime="application/json",
                            key="download_dsv"
                        )
                        st.success("File ready for download!")
                    except Exception as e:
                        st.error(f"Error preparing file: {e}")
                else:
                    st.warning("Please enter a file name")
        
        with col2:
            st.write("### Load from File")
            
            # File uploader for loading
            uploaded_file = st.file_uploader("Choose a JSON file", type=['json'])
            
            if uploaded_file is not None:
                try:
                    # Read the uploaded file
                    data = json.load(uploaded_file)
                    loaded_drv = DiscreteRandomVariable.from_dict(data)
                    
                    st.success(f"File loaded: {uploaded_file.name}")
                    
                    # Show preview of loaded data
                    st.write("**Preview of loaded DSV:**")
                    st.table({
                        "Value": loaded_drv.get_values(),
                        "Probability": loaded_drv.get_probabilities()
                    })
                    
                    # Use session state to store the loaded DSV
                    if st.button("Use This DSV"):
                        st.session_state.loaded_drv = loaded_drv
                        st.success("DSV loaded successfully!")
                        st.rerun()  # Force page rerun to update characteristics immediately
                        
                except Exception as e:
                    st.error(f"Error loading file: {e}")
            
            # Button to reset to default DSV
            if st.session_state.loaded_drv is not None:
                if st.button("Reset to Default DSV"):
                    st.session_state.loaded_drv = None
                    st.success("Reset to default DSV!")
                    st.rerun()
    
    # Section 2: Operations
    elif section == "Operations":
        st.header("Operations with Discrete Random Variables")
        
        # Use loaded DSV if available, otherwise use example
        current_drv = st.session_state.loaded_drv if st.session_state.loaded_drv else st.session_state.example_drv
        
        # Operation selection
        operation = st.selectbox(
            "Select Operation:",
            ["Multiply by Constant", "Add Two DSVs", "Multiply Two DSVs"]
        )
        
        if operation == "Multiply by Constant":
            st.subheader("Multiply DSV by Constant")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Current DSV:**")
                st.table({
                    "Value": current_drv.get_values(),
                    "Probability": current_drv.get_probabilities()
                })
            
            with col2:
                constant = st.number_input("Constant:", value=2.0, step=0.1)
                if st.button("Multiply"):
                    result = current_drv.multiply_by_constant(constant)
                    st.write("**Result:**")
                    st.table({
                        "Value": result.get_values(),
                        "Probability": result.get_probabilities()
                    })
        
        elif operation == "Add Two DSVs":
            st.subheader("Add Two Discrete Random Variables")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**First DSV (Current):**")
                st.table({
                    "Value": current_drv.get_values(),
                    "Probability": current_drv.get_probabilities()
                })
            
            with col2:
                st.write("**Second DSV:**")
                # Create a second DSV for demonstration
                drv2 = DiscreteRandomVariable({0: 0.3, 1: 0.4, 2: 0.3})
                st.table({
                    "Value": drv2.get_values(),
                    "Probability": drv2.get_probabilities()
                })
                
                if st.button("Add DSVs"):
                    result = current_drv + drv2
                    st.write("**Result (First + Second):**")
                    st.table({
                        "Value": result.get_values(),
                        "Probability": result.get_probabilities()
                    })
        
        elif operation == "Multiply Two DSVs":
            st.subheader("Multiply Two Discrete Random Variables")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**First DSV (Current):**")
                st.table({
                    "Value": current_drv.get_values(),
                    "Probability": current_drv.get_probabilities()
                })
            
            with col2:
                st.write("**Second DSV:**")
                # Create a second DSV for demonstration
                drv2 = DiscreteRandomVariable({1: 0.5, 2: 0.5})
                st.table({
                    "Value": drv2.get_values(),
                    "Probability": drv2.get_probabilities()
                })
                
                if st.button("Multiply DSVs"):
                    result = current_drv * drv2
                    st.write("**Result (First x Second):**")
                    st.table({
                        "Value": result.get_values(),
                        "Probability": result.get_probabilities()
                    })
    
    # Section 3: Visualizations
    elif section == "Visualizations":
        st.header("Visualizations")
        
        # Use loaded DSV if available, otherwise use example
        current_drv = st.session_state.loaded_drv if st.session_state.loaded_drv else st.session_state.example_drv
        
        # Visualization selection
        visualization = st.selectbox(
            "Select Visualization:",
            ["Distribution Law Table", "Distribution Law (Stem Plot)", "Probability Polyline", "Cumulative Distribution Function"]
        )
        
        if visualization == "Distribution Law Table":
            st.subheader("Distribution Law Table")
            distribution = current_drv.get_distribution_law()
            table_data = {
                "Value (x)": [pair[0] for pair in distribution],
                "Probability P(X=x)": [pair[1] for pair in distribution]
            }
            st.table(table_data)
            
            # Additional statistics
            st.write("**Summary Statistics:**")
            st.write(f"Total values: {len(distribution)}")
            st.write(f"Value range: {min(current_drv.get_values())} to {max(current_drv.get_values())}")
        
        elif visualization == "Distribution Law (Stem Plot)":
            st.subheader("Distribution Law (Stem Plot)")
            distribution_fig = current_drv.plot_distribution_law()
            st.pyplot(distribution_fig)
            
            st.write("**Interpretation:**")
            st.write("This stem plot shows the probability mass function (PMF) of the discrete random variable.")
            st.write("Each stem represents the probability of a specific value.")
        
        elif visualization == "Probability Polyline":
            st.subheader("Probability Polyline")
            polyline_fig = current_drv.plot_polyline()
            st.pyplot(polyline_fig)
            
            st.write("**Interpretation:**")
            st.write("This polyline connects the probability points with straight lines.")
            st.write("It shows the trend of probabilities across different values.")
        
        elif visualization == "Cumulative Distribution Function":
            st.subheader("Cumulative Distribution Function")
            cdf_fig = current_drv.plot_distribution_function()
            st.pyplot(cdf_fig)
            
            st.write("**Interpretation:**")
            st.write("This graph shows the cumulative distribution function (CDF).")
            st.write("It represents the probability that the random variable takes a value less than or equal to x.")
    
    # Section 4: Random Walk Simulation
    elif section == "Random Walk Simulation":
        st.header("Random Walk Simulation")
    
        st.write("Simulate a random walk process where a point moves according to a discrete random variable distribution.")
        st.write("Each step takes exactly 1 second, and the speed is proportional to the step magnitude.")
    
        # Use loaded DSV if available, otherwise use example
        current_drv = st.session_state.loaded_drv if st.session_state.loaded_drv else st.session_state.example_drv
    
        # Simulation parameters
        col1, col2, col3 = st.columns(3)
    
        with col1:
            initial_position = st.number_input("Initial Position:", value=0.0, step=0.5)
    
        with col2:
            n_steps = st.number_input("Number of Steps (n):", min_value=1, max_value=100, value=10)
    
        with col3:
            # Step distribution selection
            distribution_source = st.radio(
                "Step Distribution:",
                ["Use Current DSV", "Load from File", "Create New"]
            )
    
        # Distribution handling
        step_distribution = None
    
        if distribution_source == "Use Current DSV":
            step_distribution = current_drv
            st.write("**Using Current DSV:**")
            st.table({
                "Step Value": step_distribution.get_values(),
                "Probability": step_distribution.get_probabilities()
            })
    
        elif distribution_source == "Load from File":
            uploaded_file = st.file_uploader("Upload Step Distribution JSON", type=['json'], key="rw_upload")
            if uploaded_file is not None:
                try:
                    data = json.load(uploaded_file)
                    step_distribution = DiscreteRandomVariable.from_dict(data)
                    st.success(f"Distribution loaded from: {uploaded_file.name}")
                    st.table({
                        "Step Value": step_distribution.get_values(),
                        "Probability": step_distribution.get_probabilities()
                    })
                except Exception as e:
                    st.error(f"Error loading distribution: {e}")
    
        elif distribution_source == "Create New":
            st.write("**Create New Distribution:**")
        
            # Input fields
            col1, col2 = st.columns(2)
            with col1:
                values_input = st.text_input("Values (comma separated):", "-2, -1, 1, 2")
            with col2:
                probs_input = st.text_input("Probabilities (comma separated):", "0.25, 0.25, 0.25, 0.25")
        
            # Create button
            if st.button("Create Distribution"):
                try:
                    values = [float(x.strip()) for x in values_input.split(",")]
                    probs = [float(x.strip()) for x in probs_input.split(",")]
                
                    if len(values) != len(probs):
                        st.error("Number of values must match number of probabilities")
                    elif not math.isclose(sum(probs), 1.0, rel_tol=1e-9):
                        st.error(f"Sum of probabilities must equal 1.0 (current sum: {sum(probs):.4f})")
                    else:
                        # Save created distribution
                        st.session_state.created_distribution = DiscreteRandomVariable(dict(zip(values, probs)))
                        st.success("Distribution created successfully!")
                    
                except Exception as e:
                    st.error(f"Error creating distribution: {e}")
        
            # Show created distribution
            if 'created_distribution' in st.session_state:
                step_distribution = st.session_state.created_distribution
                st.write("**Created Distribution:**")
                st.table({
                    "Step Value": step_distribution.get_values(),
                    "Probability": step_distribution.get_probabilities()
                })
            else:
                st.info("Enter values and probabilities above, then click 'Create Distribution'")
    
        # Simulation control
        if step_distribution:
            # Set simulator parameters
            st.session_state.rw_simulator.set_parameters(step_distribution, initial_position, n_steps)
        
            # Control buttons
            col1, col2 = st.columns(2)
        
            with col1:
                if st.button("Start", type="primary"):
                    st.session_state.rw_simulator.start()
        
            with col2:
                if st.button("Stop"):
                    st.session_state.rw_simulator.stop()
        
            # Simulation status
            st.write(f"**Status:** {'Running' if st.session_state.rw_simulator.is_running else 'Stopped'}")
            st.write(f"**Step:** {st.session_state.rw_simulator.current_step} / {n_steps}")
            st.write(f"**Current Position:** {st.session_state.rw_simulator.current_position:.2f}")
        
            # Simulation area
            st.subheader("Simulation")
        
            # Create placeholder for animation
            simulation_placeholder = st.empty()
            status_placeholder = st.empty()
        
            # Run simulation
            if st.session_state.rw_simulator.is_running:
                for step in range(st.session_state.rw_simulator.current_step, n_steps):
                    if not st.session_state.rw_simulator.is_running:
                        break
                
                    # Update status
                    status_placeholder.write(f"**Step {step + 1} in progress...**")
                
                    # Perform step and wait 1 second
                    st.session_state.rw_simulator.perform_step()
                
                    # Update plot
                    fig = st.session_state.rw_simulator.plot_trajectory()
                    simulation_placeholder.pyplot(fig)
                
                    # Wait 1 second
                    time.sleep(1)
            
                # Simulation completed
                if st.session_state.rw_simulator.current_step >= n_steps:
                    st.session_state.rw_simulator.stop()
                    st.success("Simulation completed!")
        
            # Show current trajectory
            if len(st.session_state.rw_simulator.trajectory) > 1:
                fig = st.session_state.rw_simulator.plot_trajectory()
                st.pyplot(fig)
        
            # Calculate and show final distribution
            if st.session_state.rw_simulator.current_step >= n_steps or not st.session_state.rw_simulator.is_running:
                st.subheader("Final Position Distribution")
            
                final_distribution = st.session_state.rw_simulator.calculate_final_distribution()
                if final_distribution:
                    # Sort by position
                    sorted_dist = sorted(final_distribution.items())
                
                    # Display table with smart probability formatting
                    st.write("**All possible final positions and their probabilities:**")
                    table_data = {
                        "Final Position": [pos for pos, prob in sorted_dist],
                        "Probability": [format_probability(prob) for pos, prob in sorted_dist]
                    }
                    st.table(table_data)
            
        else:
            st.warning("Please set up a step distribution first.")

if __name__ == "__main__":
    print("Starting Discrete Random Variable Application...")
    main()
    print("Application execution completed successfully!")