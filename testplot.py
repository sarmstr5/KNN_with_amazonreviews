import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot():
    test_data = pd.read_csv('sentiment_output', delimiter='\t')
    plt.plot(test_data)
    #plt.savefig('test.png')
    plt.show()
def main():
    plot()

if __name__ == '__main__':
    main()
    
