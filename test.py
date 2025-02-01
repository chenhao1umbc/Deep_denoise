#%%
import matplotlib.pyplot as plt
import numpy as np

# Create x values from 0.1 to 10
x = np.linspace(0.1, 10, 1000)  # More points for smoother curve

# Calculate y values using natural log
y = np.log(x)

# Create an aesthetically pleasing plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, linewidth=2, color='#2E86C1')  # Thicker line with nice blue color

# Customize the plot
plt.title('Natural Logarithm Function', fontsize=14, pad=15)
plt.xlabel('x', fontsize=12)
plt.ylabel('ln(x)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Add some style
plt.style.use('seaborn-v0_8')
plt.tight_layout()

# Add x and y axis lines
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

#%%``