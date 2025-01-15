#%%
import matplotlib.pyplot as plt
import numpy as np

# Create x values
x = np.linspace(0.1, 10, 100)  # Avoid x=0 since log(0) is undefined

# Calculate y values using natural log
y = np.log(x)

# Create the plot
plt.plot(x, y)
plt.title('Natural Logarithm Function')
plt.xlabel('x')
plt.ylabel('ln(x)')
plt.grid(True)

#%%