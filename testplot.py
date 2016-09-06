import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot():
    test_data = pd.read_csv('sentiment_output', delimiter='\t')
    print("hello")
    print(test_data)
    plt.savefig('test.png')
    plt.plot(test_data)
def main():
    plot()

if __name__ == '__main__':
    main()
    
