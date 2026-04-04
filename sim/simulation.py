import numpy as np
import pandas as pd
import scipy.stats as stats

class DataGen:
    def __init__(self,
                 alpha_r, alpha_m, alpha_l,
                 sigma_m, sigma_l,
                 rho,
                 mu,
                 beginning_r, beginning_m, beginning_l,
                 seed, dt = 1):
        
        self.alpha_r = alpha_r
        self.alpha_m = alpha_m
        self.alpha_l = alpha_l
        self.sigma_m = sigma_m
        self.sigma_l = sigma_l
        self.rho = rho
        self.mu = mu
        self.beginning_r = beginning_r
        self.beginning_m = beginning_m
        self.beginning_l = beginning_l
        self.seed = seed
        self.dt = dt
    
    def generatePath(self, sampleSize):
    
        brownianOne = stats.norm.rvs(size = sampleSize, random_state=self.seed) * np.sqrt(self.dt)
        brownianTwo = stats.norm.rvs(size = sampleSize, random_state=self.seed+1) * np.sqrt(self.dt)

        shortRatePath, midRatePath, longRatePath = np.zeros(sampleSize), np.zeros(sampleSize), np.zeros(sampleSize)

        shortRatePath[0] = self.beginning_r 
        midRatePath[0] = self.beginning_m
        longRatePath[0] = self.beginning_l

        for t in range(1, sampleSize):
            dr = self.alpha_r * (midRatePath[t-1] - shortRatePath[t-1]) * self.dt
            dR = self.alpha_m * (longRatePath[t-1] - midRatePath[t-1]) * self.dt + self.sigma_m * (self.rho * brownianOne[t-1] + np.sqrt(1 - self.rho**2) * brownianTwo[t-1])
            dL = self.alpha_l * (self.mu - longRatePath[t-1]) * self.dt + self.sigma_l * brownianOne[t-1]

            shortRatePath[t] = shortRatePath[t-1] + dr
            midRatePath[t] = midRatePath[t-1] + dR
            longRatePath[t] = longRatePath[t-1] + dL

        return np.vstack([shortRatePath, midRatePath, longRatePath]).T

