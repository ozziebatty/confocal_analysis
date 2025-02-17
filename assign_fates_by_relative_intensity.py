def assign_fates_by_relative_intensity(cell_fluorescence, z_slice_average, characterised_cells):
    print("Assigning cell fates...")    

    cell_fluorescence = cell_fluorescence[1:] #Cut background
    channel_fluorescence = cell_fluorescence['channels']
    average_z = z_slice_average['average_z']

    total_Bra, total_Sox1, total_Bra_Sox1, total_unlabelled = 0, 0, 0, 0

    for cell in range(1, total_cells):
        characterised_cells[cell]['fate'] = 'unlabelled' #Set to unlabelled by default
        total_unlabelled += 1

    channel_1_over_2 = channel_fluorescence[:, 1] / channel_fluorescence[:, 2]

    plt.scatter(average_z, channel_1_over_2, s=2, label=f'Channel 1 relative to Channel 2')
    
    
    for channel in range(1, 3):
        fluorescence = channel_fluorescence[:, channel]

        # Calculate the line of best fit
        coefficients = np.polyfit(average_z, channel_fluorescence[:, channel], 4)
        polynomial = np.poly1d(coefficients)
        best_fit_line = polynomial(average_z)

        # Calculate the standard deviation of the best fit line
        std_best_fit = np.std(best_fit_line)
        threshold = best_fit_line + std_best_fit * dynamic_channel_thresholds[channel]

        
        # Find outliers
        for cell in range(1, len(fluorescence)):
            if fluorescence[cell] > threshold[cell]:
                if channel == 1:
                    characterised_cells[(cell+1)]['fate'] = 'Bra'
                    total_Bra += 1
                    total_unlabelled -= 1
                elif channel == 0:
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
