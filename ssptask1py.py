import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define the system transfer function coefficients
numerator = [1]  # Numerator of G(s)
denominator = [1, 3, 5, 1]  # Denominator of G(s)

# PID controller gains
Kp = 10.0  # Proportional gain
Ki = 5.0   # Integral gain
Kd = 2.0   # Derivative gain

# Actuator saturation limits
u_max = 10.0  # Maximum control signal
u_min = -10.0  # Minimum control signal

# Simulation parameters
simulation_time = 20  # Total simulation time in seconds
time_points = 1000  # Number of time points
time = np.linspace(0, simulation_time, time_points)  # Time vector
dt = time[1] - time[0]  # Time step

# Sensor noise parameters
noise_std = 0.01  # Standard deviation of sensor noise

# Step input parameters
step_time = 1.0  # Time when the reference changes
initial_reference = 0.0  # Initial reference value
final_reference = 1.0  # Final reference value

# Initialize variables for the PID controller
previous_error = 0.0  # To store the previous error for the derivative term
integral = 0.0  # To accumulate the integral of the error

# Initial state of the system [y, dy/dt, d^2y/dt^2]
system_state = [0, 0, 0]

# Define the system dynamics (state-space representation)
def system_dynamics(state, t, control_signal):
    y, dy_dt, d2y_dt2 = state
    # Third-order system dynamics: d^3y/dt^3 + 3*d^2y/dt^2 + 5*dy/dt + y = u
    d3y_dt3 = -3 * d2y_dt2 - 5 * dy_dt - y + control_signal
    return [dy_dt, d2y_dt2, d3y_dt3]

# PID controller function
def pid_controller(reference, measured_output, prev_error, integral):
    # Calculate the error
    error = reference - measured_output
    
    # Integral term: accumulate the error over time
    integral += error * dt
    
    # Derivative term: rate of change of the error
    derivative = (error - prev_error) / dt
    
    # Compute the control signal using PID formula
    control_signal = Kp * error + Ki * integral + Kd * derivative
    
    # Update the previous error for the next iteration
    prev_error = error
    
    return control_signal, prev_error, integral

# Lists to store simulation results
output_history = []  # To store the system output
control_signal_history = []  # To store the control signal
reference_history = []  # To store the reference input

# Main simulation loop
for i in range(len(time)):
    # Define the reference input as a step function
    if time[i] < step_time:
        reference = initial_reference
    else:
        reference = final_reference
    
    # Add sensor noise to the measured output
    noisy_output = system_state[0] + np.random.normal(0, noise_std)
    
    # Compute the control signal using the PID controller
    control_signal, previous_error, integral = pid_controller(
        reference, noisy_output, previous_error, integral
    )
    
    # Apply actuator saturation to the control signal
    control_signal = np.clip(control_signal, u_min, u_max)
    
    # Simulate the system using the control signal
    system_state = odeint(system_dynamics, system_state, [0, dt], args=(control_signal,))[-1]
    
    # Save the results for plotting
    output_history.append(system_state[0])
    control_signal_history.append(control_signal)
    reference_history.append(reference)

# Plot the results
plt.figure(figsize=(12, 8))

# Plot the system output and reference input
plt.subplot(2, 1, 1)
plt.plot(time, output_history, label='System Output (y)')
plt.plot(time, reference_history, 'r--', label='Reference Input (r)')
plt.title('System Response to Step Input')
plt.xlabel('Time (s)')
plt.ylabel('Output')
plt.legend()
plt.grid()

# Plot the control signal
plt.subplot(2, 1, 2)
plt.plot(time, control_signal_history, label='Control Signal (u)')
plt.title('Control Signal Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Control Signal')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
