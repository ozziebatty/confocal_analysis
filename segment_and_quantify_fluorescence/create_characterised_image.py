#Load segmented_image
#Load characterised_cells



# Define representative colours for each range
display_colours = {
    'red': (255, 0, 0),
    'blue': (0, 0, 255),
    'green': (0, 255, 0),
    'purple': (128, 0, 128),
    'grey': (100, 100, 100)
}

def create_characterised_image(characterised_cells):
    print("Creating characterised image...")

    characterised_image = np.zeros((*segmented_image.shape, 3), dtype=np.uint8)


    for cell, data, in characterised_cells.items():
        fate = data['fate']
        if fate == 'Bra_Sox1':
            colour = display_colours['purple']
        elif fate == 'Bra':
            colour = display_colours['red']
        elif fate == 'Sox1':
            colour = display_colours['green']
        else:
            colour = display_colours['grey']

        characterised_image[segmented_image == cell] = colour

    # Visualize with Napari
    print("Visualizing results with Napari...")
    viewer = napari.Viewer()
    viewer.add_image(characterised_image)
    viewer.add_image(segmented_image, name='Segmentation Masks')
    napari.run()

    return characterised_image