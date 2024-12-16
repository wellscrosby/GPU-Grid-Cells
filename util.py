import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class GridCell(nn.Module):
    def __init__(self, kernel_size):
        super(GridCell, self).__init__()
        self.conv_stationary = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False, padding_mode="circular")  # Set padding to 0
        self.conv_velocity = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False, padding_mode="circular")  # Set padding to 0
        
    def forward(self, x, direction_kernel, mask):
        self.conv_velocity.weight.data = direction_kernel
         
        velocity_result = self.conv_velocity(x)

        x = self.conv_stationary(x)
        
        x[0, 0, mask] = velocity_result[0, 0, mask]
            
        return x 

def gaussian(x, mu, sigma):
    # return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    return (1 / (sigma * (2 * np.pi)**0.5)) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def center_surround(distance, width):
    factor = (3 * np.pi) / 2
    return np.cos((distance / width) * factor) * gaussian((distance / width) * factor, 0, 2.9)
    
def dist_from_center(i, j, center, deviantion_x, deviantion_y):
    return ((i - center + deviantion_x) ** 2 + (j - center + deviantion_y) ** 2) ** 0.5

# janky but works, can certainly be improved upon
def make_filter(kernel_size, deviation_y, deviation_x):
    x = torch.zeros(1, 1, kernel_size, kernel_size)
    center = (kernel_size - 1) / 2
        
    count = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            dist = dist_from_center(i, j, center, deviation_y, deviation_x)
            if dist <= center:
                x[0, 0, i, j] = center_surround(dist, center)
                count +=1
            else:
                x[0, 0, i, j] = 0
                
    mean = torch.sum(x) / count
    
    x = torch.where(x == 0, x, x - mean)
           
    return x * 5 # Multiply by 5 to increase the intensity of the filter

def plot_agreement_sum(is_agreed, sums):
    # Create a new figure
    plt.figure(figsize=(10, 5))

    # Convert sums to a numpy array for easier plotting
    sums_np = np.array([sum_value.cpu().numpy() for sum_value in sums])

    # Plot the sums as a connected line plot
    plt.plot(sums_np, color='blue', label='Sum Value')

    # Highlight the area in green if is_agreed is true
    is_agreed_np = np.array(is_agreed)
    plt.fill_between(range(len(sums_np)), sums_np, where=is_agreed_np, color='green', alpha=0.3, label='Agreed')

    # Set y-axis limits
    plt.ylim(sums_np.min() - 100, sums_np.max() + 100)

    # Add labels and title
    plt.xlabel('Frame (time)')
    plt.ylabel('Activation Sum')
    plt.title('Sum Values with Agreement Status')
    plt.legend()

    # Show the plot
    plt.show()
    
def update_grid(grids, model, mask, direction_kernel, should_input, input_points, power, iters_per_frame, random_factor, device):
    with torch.no_grad():
        grid_size = grids.shape[-1]
        
        for _ in range(iters_per_frame):
            random_deviation = torch.rand(1, 1, grid_size, grid_size).to(device) * random_factor
            grids = model(grids, direction_kernel, mask) + random_deviation
            grids = (grids - grids.min()) / (grids.max() - grids.min())

            grids = grids**power

            if should_input:
                for deviation_x, deviation_y in input_points:
                    deviation_x = int(deviation_x)
                    deviation_y = int(deviation_y)
                    grids[0, 0, (deviation_y) % grid_size, (deviation_x) % grid_size] = 1
                    grids[0, 0, (deviation_y+1) % grid_size, (deviation_x) % grid_size] = 0.9
                    grids[0, 0, (deviation_y) % grid_size, (deviation_x+1) % grid_size] = 0.9
                    grids[0, 0, (deviation_y-1) % grid_size, (deviation_x) % grid_size] = 0.9
                    grids[0, 0, (deviation_y) % grid_size, (deviation_x-1) % grid_size] = 0.9
                    grids[0, 0, (deviation_y+1) % grid_size, (deviation_x+1) % grid_size] = 0.8
                    grids[0, 0, (deviation_y-1) % grid_size, (deviation_x-1) % grid_size] = 0.8
                    grids[0, 0, (deviation_y+1) % grid_size, (deviation_x-1) % grid_size] = 0.8
                    grids[0, 0, (deviation_y-1) % grid_size, (deviation_x+1) % grid_size] = 0.8
            
        current_sum = grids.sum()
    return grids, current_sum