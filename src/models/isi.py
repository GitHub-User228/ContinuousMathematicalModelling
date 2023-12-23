import numpy as np



class ISI:
    """
    Implementation of Improved Susceptible-Infected model 
    with 1-order approximation of time derivatives

    Attributes:
    - b0 (float): Initial trend value of b
    - db (float): Long term trend value of b
    - k (float): Decreasing rate of trend value of b
    - bss (float): Amplitude of short term fluctuations of b
    - bls (float): Amplitude of long term fluctuations of b
    - Tss (float): Period of short term fluctuations of b
    - Tls (float): Period of long term fluctuations of b
    - wss (float): Phase shift of short term fluctuations of b
    - wls (float): Phase shift of long term fluctuations of b
    - N (int): Population size
    """

    def __init__(self,
                 b0: float, 
                 db: float, 
                 k: float,
                 bss: float, 
                 bls: float, 
                 Tss: float, 
                 Tls: float, 
                 wss: float, 
                 wls: float, 
                 N: int):
        """
        Initialize ISI model with 1-order approximation

        Parameters:
        - b0 (float): Initial trend value of b
        - db (float): Long term trend value of b
        - k (float): Decreasing rate of trend value of b
        - bss (float): Amplitude of short term fluctuations of b
        - bls (float): Amplitude of long term fluctuations of b
        - Tss (float): Period of short term fluctuations of b
        - Tls (float): Period of long term fluctuations of b
        - wss (float): Phase shift of short term fluctuations of b
        - wls (float): Phase shift of long term fluctuations of b
        - N (int): Population size
        """
        
        self.b0 = b0
        self.db = db
        self.k = k
        self.bss = bss
        self.bls = bls
        self.Tss = Tss
        self.Tls = Tls
        self.wss = wss
        self.wls = wls
        self.N = N


    def transmission_rate_approximation(self, t: np.ndarray) -> np.ndarray:
        """
        Calculates an approximation of transmission rate at desired time points

        Parameters:
        - t (np.ndarray): Time points

        Returns:
        - b (np.ndarray): Transmission rates
        """
        
        trend = self.db + (self.b0 - self.db)/(1 + self.k * t)
        short_term_fluctuation = self.bss * np.sin(t/self.Tss + self.wss)**2
        long_term_fluctuation = self.bls * np.sin(t/self.Tls + self.wls)**2
        b = trend + short_term_fluctuation + long_term_fluctuation
        
        return b

    
    def step(self, 
             s1: float, 
             x1: float,
             b1: float,
             dt: float):
        """
        Function to calculate next values of S, X based on ones at the last step.

        Parameters:
        - s1 (float): Number of susceptible people at the last step
        - x1 (float): Number of total infected people at the last step
        - b1 (float): Transmission rate at the last step
        - dt (float): Time step size

        Returns:
        - s2 (float): Number of susceptible people at the current step
        - x2 (float): Number of total infected people at the current step
        """

        s2 = s1 + dt * (-b1*s1*x1/self.N)
        x2 = x1 + dt * (b1*s1*x1/self.N)

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
        - b (np.ndarray): Approximation of transmission rate at each time step
        """

        S, X = [s0], [x0]
        
        b = self.transmission_rate_approximation(np.arange(n_steps))

        for step in range(n_steps-1):

            s, x, = self.step(S[-1], X[-1], b[step], dt)
            S.append(s)
            X.append(x)

        S = np.array(S)
        X = np.array(X)

        return S, X, b