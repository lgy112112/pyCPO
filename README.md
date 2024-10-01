# pyCPO: Python Implementation of the Crested Porcupine Optimizer

---

## Introduction

Welcome to `pyCPO`, a Python library that implements the Crested Porcupine Optimizer (CPO) algorithm. This repository provides a comprehensive implementation of the CPO algorithm for solving complex optimization problems. The CPO algorithm, originally developed in MATLAB, has been ported to Python with detailed documentation and example use cases.

---

## Paper and Original MATLAB Code

The Crested Porcupine Optimizer algorithm is introduced in the following paper:

- [Crested Porcupine Optimizer: A New Nature-Inspired Algorithm for Optimization Problems](https://www.sciencedirect.com/science/article/abs/pii/S0950705123010067)

The original MATLAB code for the CPO algorithm can be found here:

- [MATLAB Code for Crested Porcupine Optimizer](https://drive.matlab.com/sharing/24c48ec7-bfd5-4c22-9805-42b7c394c691/)

---

## Algorithm Explanation
![Hystrix_cristata_Hardwicke](https://github.com/lgy112112/pyCPO/assets/144128974/59a4d77d-7337-460e-8d8d-fd7838ce2889)

The Crested Porcupine Optimizer (CPO) is a nature-inspired optimization algorithm that mimics the defense mechanism and social behavior of crested porcupines. The algorithm operates by simulating a population of search agents (porcupines) that move through the search space to find the optimal solution. The main steps of the algorithm are:

1. **Initialization**: Randomly initialize the positions of the search agents within the bounds of the search space.
2. **Fitness Evaluation**: Evaluate the fitness of each search agent based on the objective function.
3. **Position Update**: Update the positions of the search agents using a combination of random perturbations and information exchange between agents.
4. **Selection**: Retain the best positions found so far and update the global best solution.
5. **Convergence Check**: Repeat the position update and selection steps until the maximum number of iterations is reached or convergence criteria are met.

---

## Installation

To install `pyCPO`, you need to install simply torch and NumPy which is too common to inform you guys so just forget this installation sorry🤣:

```bash
git clone https://github.com/lgy112112/pyCPO.git
```

---

## Code Highlight

Here is a highlighted code example showing how to use the `pyCPO`. Let's take a snippet of CPO in `pyCPO_example.ipynb` as an example:

```python
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from IPython.display import clear_output

# Define the device to use (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Step 1: Importing Libraries and Setting Up Device

1. **Import Libraries**:

   - `numpy` for numerical operations.
   - `tqdm` for progress bars.
   - `matplotlib.pyplot` for plotting.
   - `torch`, `torch.nn`, `torch.optim` for building and training neural networks.
   - `IPython.display.clear_output` for clearing the output in Jupyter notebooks.
2. **Set Up Device**:

   - `device` is set to use GPU if available, otherwise, it falls back to CPU.

```python
def initialization(SearchAgents_no, dim, ub, lb):
    """
    Initialize the positions of search agents within the given bounds.
  
    Parameters:
    SearchAgents_no (int): Number of search agents.
    dim (int): Dimension of the search space.
    ub (list or array): Upper bounds of the search space.
    lb (list or array): Lower bounds of the search space.
  
    Returns:
    Positions (array): Initialized positions of search agents.
    """
    Boundary_no = len(ub)
    Positions = np.zeros((SearchAgents_no, dim))
  
    if Boundary_no == 1:
        # If all dimensions have the same bounds
        Positions = np.random.rand(SearchAgents_no, dim) * (ub - lb) + lb
    else:
        # If each dimension has different bounds
        for i in range(dim):
            ub_i = ub[i]
            lb_i = lb[i]
            Positions[:, i] = np.random.rand(SearchAgents_no) * (ub_i - lb_i) + lb_i
    return Positions
```

### Step 2: Initialization Function

1. **Parameters**:

   - `SearchAgents_no`: Number of search agents.
   - `dim`: Dimension of the search space.
   - `ub`, `lb`: Upper and lower bounds of the search space.
2. **Initialize Positions**:

   - Creates an array `Positions` to hold the positions of search agents.
   - If all dimensions have the same bounds, positions are initialized uniformly within `[lb, ub]`.
   - If bounds vary across dimensions, each dimension is initialized separately within its respective bounds.

```python
def CPO(Pop_size, Tmax, lb, ub, dim, fobj):
    """
    Execute the Crested Porcupine Optimizer algorithm.
  
    Parameters:
    Pop_size (int): Population size (number of search agents).
    Tmax (int): Maximum number of iterations.
    lb (list or array): Lower bounds of the search space.
    ub (list or array): Upper bounds of the search space.
    dim (int): Dimension of the search space.
    fobj (function): Objective function to be minimized.
  
    Returns:
    Gb_Fit (float): Best fitness value found.
    Gb_Sol (array): Best solution found.
    Conv_curve (array): Convergence curve of the best fitness value over iterations.
    """
    Conv_curve = np.zeros(Tmax)  # Array to store the best fitness value at each iteration
    ub = np.array(ub)
    lb = np.array(lb)
  
    X = initialization(Pop_size, dim, ub, lb)  # Initialize the positions of search agents
    t = 0  # Iteration counter
  
    # Evaluate the fitness of initial positions
    fitness = np.array([fobj(X[i, :]) for i in range(Pop_size)])
    Gb_Fit = np.min(fitness)  # Best fitness value
    Gb_Sol = X[np.argmin(fitness), :]  # Best solution
  
    Xp = X.copy()  # Copy of the current positions
```

### Step 3: CPO Function (Part 1)

1. **Parameters**:

   - `Pop_size`: Population size.
   - `Tmax`: Maximum number of iterations.
   - `lb`, `ub`: Lower and upper bounds of the search space.
   - `dim`: Dimension of the search space.
   - `fobj`: Objective function to be minimized.
2. **Initialization**:

   - `Conv_curve`: Array to store the best fitness value at each iteration.
   - Convert `ub` and `lb` to numpy arrays.
   - `X`: Initialize positions of search agents using the `initialization` function.
   - `t`: Iteration counter.
3. **Evaluate Initial Fitness**:

   - Compute fitness for initial positions.
   - `Gb_Fit`: Best fitness value found.
   - `Gb_Sol`: Best solution found.
   - `Xp`: Copy of the current positions.

```python
    # Optimization loop
    with tqdm(total=Tmax, desc='CPO Optimization', unit='iter') as pbar:
        while t < Tmax:
            for i in range(Pop_size):
                U1 = np.random.rand(dim) > np.random.rand()
                if np.random.rand() < np.random.rand():
                    if np.random.rand() < np.random.rand():
                        y = (X[i, :] + X[np.random.randint(Pop_size), :]) / 2
                        X[i, :] = X[i, :] + (np.random.randn() * abs(2 * np.random.rand() * Gb_Sol - y))
                    else:
                        y = (X[i, :] + X[np.random.randint(Pop_size), :]) / 2
                        X[i, :] = (U1 * X[i, :]) + ((1 - U1) * (y + np.random.rand() * (X[np.random.randint(Pop_size), :] - X[np.random.randint(Pop_size), :])))
                else:
                    Yt = 2 * np.random.rand() * (1 - t / Tmax) ** (t / Tmax)
                    U2 = (np.random.rand(dim) < 0.5) * 2 - 1
                    S = np.random.rand() * U2
                    if np.random.rand() < 0.8:
                        St = np.exp(fitness[i] / (np.sum(fitness) + np.finfo(float).eps))
                        S = S * Yt * St
                        X[i, :] = (1 - U1) * X[i, :] + U1 * (X[np.random.randint(Pop_size), :] + St * (X[np.random.randint(Pop_size), :] - X[np.random.randint(Pop_size), :]) - S)
                    else:
                        Mt = np.exp(fitness[i] / (np.sum(fitness) + np.finfo(float).eps))
                        vt = X[i, :]
                        Vtp = X[np.random.randint(Pop_size), :]
                        Ft = np.random.rand(dim) * (Mt * (-vt + Vtp))
                        S = S * Yt * Ft
                        X[i, :] = Gb_Sol + 0.2 * (1 - np.random.rand()) + np.random.rand() * (U2 * Gb_Sol - X[i, :]) - S
              
                # Ensure the positions are within the bounds
                X[i, :] = np.clip(X[i, :], lb, ub)
                nF = fobj(X[i, :])  # Evaluate the fitness of the new position
                if fitness[i] < nF:
                    X[i, :] = Xp[i, :]
                else:
                    Xp[i, :] = X[i, :]
                    fitness[i] = nF
                    if fitness[i] <= Gb_Fit:
                        Gb_Sol = X[i, :]
                        Gb_Fit = fitness[i]
                      
            t += 1
            Conv_curve[t - 1] = Gb_Fit  # Update the convergence curve
            pbar.set_postfix({'Best Fitness': Gb_Fit, 'Iteration': t})
            pbar.update(1)
        pbar.close()
  
    return Gb_Fit, Gb_Sol, Conv_curve
```

### Step 4: CPO Function (Part 2)

1. **Optimization Loop**:

   - Iterate until the maximum number of iterations (`Tmax`) is reached.
   - For each search agent, update its position based on random perturbations and the best-known solution.
2. **Position Update**:

   - `U1`, `U2`: Random vectors for updating positions.
   - Update positions using various strategies (e.g., perturbation, attraction to the best solution).
3. **Fitness Evaluation**:

   - Evaluate the fitness of the new positions.
   - Update the best-known solution if a better solution is found.
4. **Convergence Curve**:

   - Update the convergence curve with the best fitness value at each iteration.
5. **Return Results**:

   - Return the best fitness value, the best solution, and the convergence curve.

```python
# Example usage of the CPO algorithm
# Define the objective function for hyperparameter optimization of a neural network
def fobj(params):
    lstm_units = int(params[0])

    conv_filters_1 = int(params[1])
    conv_filters_2 = int(params[2])
    conv_filters_3 = int(params[3])
    conv_filters_4 = int(params[4])
    conv_filters_5 =int(params[5])

    fc_units_1 = int(params[6])
    fc_units_2 = int(params[7])
    learning_rate = params[8]

    conv_filters = [conv_filters_1, conv_filters_2, conv_filters_3, conv_filters_4, conv_filters_5]
    fc_units = [fc_units_1, fc_units_2]

    # Define the model with the given hyperparameters
    model = CNN1DModelWithLSTMAndAttention(input_size, num_classes, lstm_units, conv_filters, fc_units).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        train_bar = tqdm(train_loader, desc=f'Optimization Epoch {epoch+1}/{num_epochs} (Training)', unit='batch')
        for X_batch, y_batch in train_bar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            X_batch = X_batch.unsqueeze(1)  # Add channel dimension
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_bar.set_postfix({'Loss': loss.item()})
        train_bar.close()

    model.eval()
    test_loss = 0
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc='Validation (Testing)', unit='batch')
        for X_batch, y_batch in test_bar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            X_batch = X_batch.unsqueeze(1)  # Add channel dimension
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item()
            test_bar.set_postfix({'Loss': loss.item()})
        test_bar.close()

    return test_loss / len(test_loader)
```

### Step 5: Define the Objective Function

1. **Objective Function**:

   - `fobj` is defined to handle hyperparameter optimization for a neural network model.
   - Extract hyperparameters from the input parameters.
2. **Model Definition**:

   - Define a neural network model (`CNN1DModelWithLSTMAndAttention`) with the given hyperparameters.
   - Use `nn.MSELoss` as the loss function and `optim.Adam` as the optimizer.
3. **Training Loop**:

   - Train the model for a specified number of epochs.
   - Use `tqdm` to display a progress bar for the training loop.
4. **Validation Loop**:

   - Evaluate the model on the validation set and compute the test loss.

```python
    # CPO algorithm parameters
    Pop_size = 10
    Tmax = 10
    dim = 9  # Number of hyperparameters to optimize
    lb = [10, 16, 16, 16, 16, 16, 50, 50, 0.00001]  # Lower bounds of hyperparameters
    ub = [512, 128, 128, 128, 128, 128, 2048, 1024, 0.001]  # Upper bounds of hyperparameters

    # Use CPO to optimize hyperparameters
    Best_fit, Best_sol, Conv_curve = CPO(Pop_size, Tmax, lb, ub, dim, fobj)
    clear_output(wait=True)
    print(f"Best parameters found: {Best_sol}")

    # Plot the convergence curve
    plt.figure()
    plt.plot(Conv_curve)
    plt.title('Convergence Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness')
    plt.grid()
    plt.show()
```

### Step 6: Run the CPO Algorithm

1. **CPO Parameters**:

   - Define the parameters for the CPO algorithm, including population size, maximum iterations, dimension of the search space, and bounds for the hyperparameters.
2. **Run CPO**:

   - Call the `CPO` function with the defined parameters and the objective function.
3. **Output Results**:

   - Print the best hyperparameters found.
   - Plot the convergence curve to visualize the optimization process.

---

## Show Me the Outcome
![263346b1-4842-45a6-845f-b1e6cd98c94e](https://github.com/lgy112112/pyCPO/assets/144128974/0e355986-acfa-4784-bb56-415787249ca3)

The graph above compares the output of the Original model and the Crested Porcupine Optimizer (CPO) model. The latter one outrun the former one. To compare with digits, the CPO model has a better performance of 0.90 R2 Score, while the Original model has a R2 Score of 0.85.
