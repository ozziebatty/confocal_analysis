// Define the cropping dimensions
z_start = 30;
z_end = 100;
y_start = 125;
y_end = 325;
x_start = 125;
x_end = 325;

// Set the cropping region
run("Select All");
makeRectangle(x_start, y_start, x_end - x_start, y_end - y_start);
run("Crop");

// Crop along Z dimension
setSlice(z_start);
run("Make Substack...", "start=" + z_start + " end=" + z_end);