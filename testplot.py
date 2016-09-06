import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot():
    test_data = pd.read_csv('sentiment_output', delimiter='\t')
    test_data.plot()

def main():
    plot()

if __name__ == '__main__':
    main()
    
