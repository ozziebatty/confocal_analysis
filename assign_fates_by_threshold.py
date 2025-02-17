dynamic_channel_thresholds = [0.15, 0.05, 0, -0.215]
channel_stdev_thresholds = [-7, -2, 1, -1] #threshold above which is classed as positive(more negative = more generous)

#Load in stitched_image
#Load in normalised_cell_fluorescence
#Load in characterised_cells

def find_average_z_slice_of_each_label(segmented_image):
    print("Calculating z positions...")
    z_slice_averages = np.zeros(total_cells, dtype=[('running_z_total', int), ('z_stack_count', int), ('average_z', float)])

    for z in range(total_z):
        z_slice_segmented_image = segmented_image[z]
        
        for label in np.unique(z_slice_segmented_image):
            if label == 0:  # Skip background
                continue
            z_slice_averages[label]['running_z_total'] += z
            z_slice_averages[label]['z_stack_count'] += 1

    z_slice_averages = z_slice_averages[1:]

    for label in range(len(z_slice_averages)):
        z_slice_averages[label]['average_z'] = z_slice_averages[label]['running_z_total'] / z_slice_averages[label]['z_stack_count']
      
    return z_slice_averages

def plot_channel_intensities_to_z(cell_fluorescence, z_slice_average):
    colours = ['red', 'green', 'blue', 'orange']
    print("Plotting channel intensities...")
    cell_fluorescence = cell_fluorescence[1:] #Cut background

    channel_fluorescence = cell_fluorescence['channels']
    average_z = z_slice_average['average_z']
    
    plt.figure(figsize=(10, 6))

    for channel in range(0, 1):
        fluorescence = channel_fluorescence[:, channel]
        plt.scatter(average_z, fluorescence, color=colours[channel-1], s=10, label=f'Channel {channel+1}')

        # Calculate the line of best fit
        coefficients = np.polyfit(average_z, channel_fluorescence[:, channel], 4)
        polynomial = np.poly1d(coefficients)
        best_fit_line = polynomial(average_z)

        # Calculate the standard deviation of the best fit line
        std_best_fit = np.std(best_fit_line)
        threshold = best_fit_line + std_best_fit * dynamic_channel_thresholds[channel]

        # Plot the threshold
        plt.plot(average_z, threshold, label=f'Threshold - Channel {channel}')
        
        # Find outliers
        outliers = (fluorescence > threshold)
        
        # Plot outliers
        plt.scatter(average_z[outliers], fluorescence[outliers], color='black', s=2, marker='x')

    # Customize the plot
    plt.xlabel('Z Slice')
    plt.ylabel('Normalised Channel Intensity')
    plt.legend(title='Channels')
    plt.grid(True)

    # Display the plot
    plt.tight_layout()
    plt.show()


def assign_fates_by_threshold(cell_fluorescence, z_slice_average, characterised_cells):
    print("Assigning cell fates...")    

    cell_fluorescence = cell_fluorescence[1:] #Cut background
    channel_fluorescence = cell_fluorescence['channels']
    average_z = z_slice_average['average_z']

    total_Bra, total_Sox1, total_Bra_Sox1, total_unlabelled = 0, 0, 0, 0

    for cell in range(1, total_cells):
        characterised_cells[cell]['fate'] = 'unlabelled' #Set to unlabelled by default
        total_unlabelled += 1
    
    for channel in range(0, 3):
        fluorescence = channel_fluorescence[:, channel]

        # Calculate the line of best fit
        coefficients = np.polyfit(average_z, channel_fluorescence[:, channel], 4)
        polynomial = np.poly1d(coefficients)
        best_fit_line = polynomial(average_z)

        # Calculate the standard deviation of the best fit line
        std_best_fit = np.std(best_fit_line)
        threshold = best_fit_line + dynamic_channel_thresholds[channel]

        
        # Find outliers
        for cell in range(1, len(fluorescence)):
            if fluorescence[cell] > threshold[cell]:
                if channel == 1:
                    characterised_cells[(cell+1)]['fate'] = 'Bra'
                    total_Bra += 1
                    total_unlabelled -= 1
                elif channel == 2:
                    if characterised_cells[(cell+1)]['fate'] == 'Bra':
                        characterised_cells[(cell+1)]['fate'] = 'Bra_Sox1'
                        total_Bra_Sox1 += 1
                        total_Bra -= 1                        
                    else:
                        characterised_cells[(cell+1)]['fate'] = 'Sox1'
                        total_Sox1 += 1
                        total_unlabelled -= 1

    print("Total Bra+ : ", total_Bra)
    print("Total Sox1+ : ", total_Sox1)
    print("Total Bra+ Sox1+ : ", total_Bra_Sox1)
    print("Total unlabelled : ", total_unlabelled)    


z_slice_average = find_average_z_slice_of_each_label(segmented_image)

plot_channel_intensities_to_z(normalised_cell_fluorescence, z_slice_average)

assign_fates_by_threshold(normalised_cell_fluorescence, z_slice_average, characterised_cells)
