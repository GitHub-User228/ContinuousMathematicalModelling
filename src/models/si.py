import numpy as np



class SI:
    """
    Implementation of Susceptible-Infected model 
    with 1-order approximation of time derivatives

    Attributes:
    - b (float): Transmission rate
    - N (int): Population size
    """

    def __init__(self,
                 b: float, 
                 N: int):
        """
        Initialize SI model with 1-order approximation

        Parameters:
        - b (float): Transmission rate
        - N (int): Population size
        """
        
        self.b = b
        self.N = N

    
    def step(self, 
             s1: float, 
             x1: float,
             dt: float):
        """
        Function to calculate next values of S, X based on ones at the last step.

        Parameters:
        - s1 (float): Number of susceptible people at the last step
        - x1 (float): Number of total infected people at the last step
        - dt (float): Time step size

        Returns:
        - s2 (float): Number of susceptible people at the current step
        - x2 (float): Number of total infected people at the current step
        """

        s2 = s1 + dt * (-self.b*s1*x1/self.N)
        x2 = x1 + dt * (self.b*s1*x1/self.N)

        return s2, x2


    def run(self, 
            n_steps: int, 
            s0: float, 
            x0: float,
            dt: float):
        """
        Function to run a model for desired number of steps

        Parameters:
        - n_steps (int): Number of steps
        - s0 (float): Initial number of susceptible people
        - x0 (float): Initial number of total infected people
        - dt (float): Time step size

        Returns:
        - S (np.ndarray): Number of susceptible people at each time step
        - X (np.ndarray): Number of infected people at each time step
        """

        S, X = [s0], [x0]

        for step in range(n_steps-1):

            s, x = self.step(S[-1], X[-1], dt)
            S.append(s)
            X.append(x)

        S = np.array(S)
        X = np.array(X)

        return S, X