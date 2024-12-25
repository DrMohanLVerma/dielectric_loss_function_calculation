import os
import numpy as np
import matplotlib.pyplot as plt

def find_dielectric_files():
    """
    Automatically detect the real and imaginary dielectric function files in the current directory.

    Returns:
        str: Path to the real dielectric function file.
        str: Path to the imaginary dielectric function file.
    """
    real_file = None
    imag_file = None

    for file in os.listdir():
        if "epsilon_real" in file.lower() and file.endswith(".out"):
            real_file = file
        elif "epsilon_img" in file.lower() and file.endswith(".out"):
            imag_file = file

    if real_file is None or imag_file is None:
        raise FileNotFoundError("Could not detect both real and imaginary dielectric function files.")

    print(f"Detected real dielectric function file: {real_file}")
    print(f"Detected imaginary dielectric function file: {imag_file}")

    return real_file, imag_file

def load_data(file_path):
    """
    Load data from a file.

    Parameters:
        file_path (str): Path to the file containing energy and dielectric values.

    Returns:
        np.ndarray: Array of energies (eV) and dielectric values.
    """
    return np.loadtxt(file_path)

def combine_dielectric_data(real_file, imag_file, output_file):
    """
    Combine real and imaginary dielectric function data into a single file.

    Parameters:
        real_file (str): Path to the file containing the real part of the dielectric function.
        imag_file (str): Path to the file containing the imaginary part of the dielectric function.
        output_file (str): Path to save the combined data.
    """
    # Load real and imaginary data
    real_data = load_data(real_file)
    imag_data = load_data(imag_file)

    # Ensure the energies match in both files
    if not np.allclose(real_data[:, 0], imag_data[:, 0]):
        raise ValueError("Energy values in the real and imaginary files do not match.")

    # Combine data
    combined_data = np.column_stack((real_data[:, 0], real_data[:, 1], imag_data[:, 1]))

    # Save to output file in the same folder
    np.savetxt(output_file, combined_data, fmt="%.6f", 
               header="Energy(eV) RealPart(Epsilon_1) ImaginaryPart(Epsilon_2)")
    print(f"Combined data saved to {output_file}")

def calculate_loss_function(epsilon_real, epsilon_imag):
    """
    Calculate the energy loss function Im[-1 / epsilon].

    Parameters:
        epsilon_real (np.ndarray): Real part of dielectric function.
        epsilon_imag (np.ndarray): Imaginary part of dielectric function.

    Returns:
        np.ndarray: Loss function values.
    """
    epsilon = epsilon_real + 1j * epsilon_imag
    loss_function = -np.imag(1 / epsilon)
    return loss_function

def save_loss_function_data(energies, loss_function, output_file):
    """
    Save the energy loss function data to a .dat file.

    Parameters:
        energies (np.ndarray): Energies (eV).
        loss_function (np.ndarray): Loss function values.
        output_file (str): Path to the output file.
    """
    data = np.column_stack((energies, loss_function))
    np.savetxt(output_file, data, fmt="%.6f", header="Energy(eV) LossFunction")
    print(f"Loss function data saved to {output_file}")

def plot_dielectric_and_loss_function(energies, epsilon_real, epsilon_imag, loss_function):
    """
    Plot the real and imaginary dielectric function, along with the loss function.

    Parameters:
        energies (np.ndarray): Energies (eV).
        epsilon_real (np.ndarray): Real part of dielectric function.
        epsilon_imag (np.ndarray): Imaginary part of dielectric function.
        loss_function (np.ndarray): Loss function values.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the real and imaginary dielectric function
    ax1.plot(energies, epsilon_real, label="Real Part of Dielectric Function (ε₁)", color='blue')
    ax1.plot(energies, epsilon_imag, label="Imaginary Part of Dielectric Function (ε₂)", color='red')
    ax1.set_xlabel("Energy (eV)", fontsize=14)
    ax1.set_ylabel("Dielectric Function", fontsize=14)
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True)

    # Create a second y-axis to plot the loss function
    ax2 = ax1.twinx()
    ax2.plot(energies, loss_function, label="Loss Function Im[-1/ε]", color='green', linestyle='--')
    ax2.set_ylabel("Loss Function", fontsize=14)
    ax2.tick_params(axis='y', labelcolor='green')

    # Adding the legends
    ax1.legend(loc='upper left', fontsize=12)
    ax2.legend(loc='upper right', fontsize=12)

    plt.title("Dielectric Function and Loss Function", fontsize=16)
    plt.show()

def main():
    # Automatically detect the dielectric files
    real_file, imag_file = find_dielectric_files()

    # Output files
    combined_file = "dielectric_function.dat"
    loss_function_file = "loss_function.dat"

    # Combine dielectric function files
    combine_dielectric_data(real_file, imag_file, combined_file)

    # Load combined data
    combined_data = np.loadtxt(combined_file, skiprows=1)
    energies = combined_data[:, 0]
    epsilon_real = combined_data[:, 1]
    epsilon_imag = combined_data[:, 2]

    # Calculate the loss function
    loss_function = calculate_loss_function(epsilon_real, epsilon_imag)

    # Save the loss function data to a file
    save_loss_function_data(energies, loss_function, loss_function_file)

    # Plot the dielectric functions and loss function
    plot_dielectric_and_loss_function(energies, epsilon_real, epsilon_imag, loss_function)

if __name__ == "__main__":
    main()

