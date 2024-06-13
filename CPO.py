import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

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

# Example usage of the CPO algorithm
if __name__ == "__main__":
    # Define the objective function
    def fobj(params):
        # Your objective function implementation
        return np.sum(params**2)  # Example: Sum of squares function

    # CPO algorithm parameters
    Pop_size = 10
    Tmax = 100
    dim = 5  # Dimension of the search space
    lb = [-10, -10, -10, -10, -10]  # Lower bounds of the search space
    ub = [10, 10, 10, 10, 10]  # Upper bounds of the search space

    # Run the CPO algorithm
    Best_fit, Best_sol, Conv_curve = CPO(Pop_size, Tmax, lb, ub, dim, fobj)

    # Print the best solution and fitness
    print(f"Best Solution: {Best_sol}")
    print(f"Best Fitness: {Best_fit}")

    # Plot the convergence curve
    plt.figure()
    plt.plot(Conv_curve)
    plt.title('Convergence Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness')
    plt.grid()
    plt.show()
