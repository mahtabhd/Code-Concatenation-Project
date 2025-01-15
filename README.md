# Code Concatenation in Quantum Error Correction

This repository contains the implementation and simulation of Shor's quantum error-correcting code and its concatenation, developed as part of the "Practical Quantum Computing – CS-C3260" course. The project aims to explore the benefits of concatenating quantum error-correcting codes to achieve lower logical error rates.

## Project Overview

Quantum error correction is essential for mitigating errors in quantum computations. This project focuses on **code concatenation**, where error-correcting codes are layered to enhance error suppression capabilities. The implementation includes:

- Shor's 9-qubit code for single-qubit error correction.
- Concatenation of Shor's code with itself.
- Simulation of logical error rates under different error conditions.

## Features

- **Shor's Code Implementation**:
  - Encodes 1 logical qubit into 9 physical qubits.
  - Uses syndrome measurements and recovery techniques to detect and correct errors.

- **Concatenated Shor Code**:
  - Encodes 1 logical qubit into 81 physical qubits using recursive concatenation.
  - Employs custom logical gates and syndrome measurements for robust error correction.

- **Simulations**:
  - Conducted using Cirq’s pure state simulator.
  - Simulated depolarizing noise with varying physical error rates.
  - Compared logical error rates under two scenarios:
    - Errors occurring only after encoding and before syndrome measurements.
    - Errors occurring throughout the circuit.

## Repository Structure

```plaintext
.
├── code-concatenation-source-code.py  # Python implementation of the project
├── code-concatenation-presentation.pdf  # Project presentation slides
├── README.md  # This file

```
## Installation

To run the code, you will need:

- Python 3.8 or higher
- Required libraries:
  - `cirq`
  - `qualtran`
  - `numpy`
  - `matplotlib`

Install the dependencies using pip:

```bash
pip install cirq numpy matplotlib qualtran
```

## How to Run

1. **Visualize Circuits**:
   The script provides various circuit visualizations using Cirq and Qualtran. For example:
   - The 9-qubit Shor code circuit.
   - The 81-qubit concatenated Shor code circuit.

   These can be displayed using:

   ```bash
   python code-concatenation-source-code.py
   ```

2. **Simulate Logical Error Rates**:
   Simulations are included to analyze logical error rates. The results are plotted as graphs showing the relationship between physical error rates and logical error rates.

## Results

For a more elaborate explanation of the project and an analysis of the results refer to the "code-concatenation-presentation.pdf" file.

## Acknowledgments

This project was completed under the supervision of Professor Alexandru Paler as part of the "Practical Quantum Computing – CS-C3260" course.
  
