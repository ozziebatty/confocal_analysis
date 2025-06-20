#%%
print("Importing")
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons, TextBox
from matplotlib.patches import Rectangle, Polygon
from matplotlib.path import Path
import random
from matplotlib.widgets import PolygonSelector
import matplotlib.colors as mcolors
import json
import os
from datetime import datetime
from tkinter import simpledialog
from PyQt5.QtWidgets import QApplication, QInputDialog
import sys

matplotlib.use('TkAgg')  # or 'Qt5Agg'

#%%
# Load data
print("Loading data")
file_path = os.path.normpath(r"Y:\Room225_SharedFolder\Leica_Stellaris5_data\Gastruloids\oskar\analysis\SBSO_OPP_NM_two_analysis\replicate_2\replicate_2_characterised_cells.csv")
gates_directory = os.path.normpath(r"Y:\Room225_SharedFolder\Leica_Stellaris5_data\Gastruloids\oskar\analysis\SBSO_OPP_NM_two_analysis\replicate_2")
df = pd.read_csv(file_path)

columns = ['cell_number','pixel_count','z_position','channel_0','channel_1','channel_2','channel_3','channel_4']
channels = ['channel_0', 'channel_1', 'channel_2', 'channel_3', 'channel_4']

#%%
class InteractiveScatterPlot:
    def __init__(self, dataframe, columns, channels, save_dir=gates_directory):
        print("Initializing InteractiveScatterPlot")
        self.df = dataframe
        self.channels = channels
        self.columns = columns
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Default settings
        self.x_channel = channels[0]
        self.y_channel = channels[3]
        self.display_prop = 1.0  # 100%
        self.x_log = False
        self.y_log = False
        self.epsilon = 1.0  # For log scaling
        
        # Population definition
        self.input_population = "All"  # Can be "All" or gate name
        self.current_population_mask = None
        
        # Gating variables
        self.polygon_selector = None
        self.current_vertices = []
        self.gates = {}  # Store gates by channel combination
        self.gate_counter = 1
        
        # Text input variables for gate naming
        self.naming_mode = False
        self.gate_name_textbox = None
        self.gate_name_figure = None
        
        self.clear_gates()

        # Load existing gates
        self.load_gates()
        
        # Calculate initial ranges
        self.update_ranges()
        
        # Create the plot
        self.setup_plot()
        self.update_plot()
        
    def get_channel_key(self):
        """Get unique key for current channel combination"""
        print("Getting channel key")
        return f"{self.x_channel}_{self.y_channel}_{self.x_log}_{self.y_log}"
    
    def get_current_gates(self):
        """Get gates for current channel combination"""
        print("Getting current gates")
        key = self.get_channel_key()
        return self.gates.get(key, {})
    
    def update_ranges(self):
        """Update ranges based on current channels and log scale"""
        print("Updating ranges")
        # Get current population data
        current_data = self.get_current_population_data()
        
        if len(current_data) == 0:
            # Handle empty data case
            self.x_min, self.x_max = 0, 1
            self.y_min, self.y_max = 0, 1
            return
        
        x_data = current_data[self.x_channel]
        y_data = current_data[self.y_channel]
        
        if self.x_log:
            x_data = np.log10(x_data + self.epsilon)
        if self.y_log:
            y_data = np.log10(y_data + self.epsilon)
            
        # Check for valid data ranges
        if len(x_data) > 0 and not np.all(np.isnan(x_data)):
            data_x_min, data_x_max = np.nanmin(x_data), np.nanmax(x_data)
            x_range = data_x_max - data_x_min
            padding = x_range * 0.1 if x_range > 0 else 1
            self.x_min = data_x_min - padding
            self.x_max = data_x_max + padding
        else:
            self.x_min, self.x_max = 0, 1
            
        if len(y_data) > 0 and not np.all(np.isnan(y_data)):
            data_y_min, data_y_max = np.nanmin(y_data), np.nanmax(y_data)
            y_range = data_y_max - data_y_min
            padding = y_range * 0.1 if y_range > 0 else 1
            self.y_min = data_y_min - padding
            self.y_max = data_y_max + padding
        else:
            self.y_min, self.y_max = 0, 1
        
        # Ensure ranges are valid
        if self.x_min == self.x_max:
            self.x_min -= 0.5
            self.x_max += 0.5
        if self.y_min == self.y_max:
            self.y_min -= 0.5
            self.y_max += 0.5
    
    def get_current_population_data(self):
        """Get data for current input population"""
        print("Getting current population data")
        if self.input_population == "All":
            return self.df
        else:
            # First check current channel combination
            current_key = self.get_channel_key()
            if current_key in self.gates and self.input_population in self.gates[current_key]:
                gate_info = self.gates[current_key][self.input_population]
                mask = self.calculate_gate_mask(gate_info, current_key)
                return self.df[mask]
            
            # Then check "All" key for logic gates
            if "All" in self.gates and self.input_population in self.gates["All"]:
                gate_info = self.gates["All"][self.input_population]
                if gate_info['type'] == 'logic':
                    mask = self.evaluate_logic_expression_for_data(gate_info['expression'], self.df)
                    if mask is not None:
                        return self.df[mask]
            
            # Fallback: search all keys
            for key, gates_dict in self.gates.items():
                if self.input_population in gates_dict:
                    gate_info = gates_dict[self.input_population]
                    if gate_info['type'] == 'logic':
                        mask = self.evaluate_logic_expression_for_data(gate_info['expression'], self.df)
                        if mask is not None:
                            return self.df[mask]
                    else:
                        mask = self.calculate_gate_mask(gate_info, key)
                        return self.df[mask]
            
            return self.df  # Fallback to all if gate not found
        
    def setup_plot(self):
        print("Setting up plot")
        # Create figure with smaller height
        self.fig = plt.figure(figsize=(14, 7))
        
        #LEFT BOTTOM WIDTH HEIGHT

        # Main scatter plot - larger
        self.ax_main = plt.axes([0.08, 0.35, 0.55, 0.6])
        
       # Control panel layout - top right - CHANGED: Extended axis boxes down even more, shifted everything down
        # Channel selection dropdowns - CHANGED: Extended downward by extra 50%
        self.ax_x_dropdown = plt.axes([0.68, 0.68, 0.12, 0.225])  # Extended from 0.15 to 0.225 height (50% more)
        self.ax_y_dropdown = plt.axes([0.82, 0.68, 0.12, 0.225])  # Extended from 0.15 to 0.225 height (50% more)

        # Log scale dropdowns - CHANGED: Shifted down more
        self.ax_x_log_dropdown = plt.axes([0.68, 0.58, 0.12, 0.07])  # Moved down from 0.65
        self.ax_y_log_dropdown = plt.axes([0.82, 0.58, 0.12, 0.07])  # Moved down from 0.65

        # Display proportion and population - CHANGED: Shifted down more
        self.ax_prop_dropdown = plt.axes([0.68, 0.48, 0.12, 0.07])  # Moved down from 0.55
        self.ax_pop_dropdown = plt.axes([0.82, 0.3, 0.12, 0.25])  # Moved down from 0.37

        # Gate information display - CHANGED: Shifted down more
        self.ax_gate_info = plt.axes([0.68, 0.02, 0.26, 0.25])  # Moved down from 0.1, reduced height to fit
        
        # Range sliders - bottom left
        self.ax_x_min = plt.axes([0.08, 0.25, 0.25, 0.03])
        self.ax_x_max = plt.axes([0.08, 0.2, 0.25, 0.03])
        self.ax_y_min = plt.axes([0.08, 0.15, 0.25, 0.03])
        self.ax_y_max = plt.axes([0.08, 0.1, 0.25, 0.03])
        
        # Gate control buttons - bottom right, only two buttons
        button_width = 0.08
        button_height = 0.04
        button_spacing = 0.01
        start_x = 0.4
        button_y = 0.05
        
        self.ax_gate_btn = plt.axes([start_x, button_y, button_width, button_height])
        self.ax_save_gate = plt.axes([start_x + button_width + button_spacing, button_y, button_width, button_height])
        self.ax_logic_gate = plt.axes([start_x + 2*(button_width + button_spacing), button_y, button_width, button_height])
        self.ax_clear_all = plt.axes([start_x + 3*(button_width + button_spacing), button_y, button_width, button_height])
        
        # Create radio button controls
        self.radio_x = RadioButtons(self.ax_x_dropdown, columns)
        self.radio_y = RadioButtons(self.ax_y_dropdown, columns)
        self.radio_x.set_active(columns.index(self.x_channel))
        self.radio_y.set_active(columns.index(self.y_channel))
        
        # Create log scale controls
        self.radio_x_log = RadioButtons(self.ax_x_log_dropdown, ['Linear', 'Log'])
        self.radio_y_log = RadioButtons(self.ax_y_log_dropdown, ['Linear', 'Log'])
        self.radio_x_log.set_active(1 if self.x_log else 0)
        self.radio_y_log.set_active(1 if self.y_log else 0)
        
        # Create display proportion control
        prop_labels = ['0.1%', '1%', '10%', '100%']
        self.radio_prop = RadioButtons(self.ax_prop_dropdown, prop_labels)
        self.radio_prop.set_active(3)  # Default to 100%
        # Make font smaller for radio buttons
        for label in self.radio_prop.labels:
            label.set_fontsize(7)
        
        # Create population dropdown
        self.update_population_dropdown()
        
        # Create sliders
        self.slider_x_min = Slider(self.ax_x_min, 'X Min', 
                                self.x_min, self.x_max, valinit=self.x_min)
        self.slider_x_max = Slider(self.ax_x_max, 'X Max', 
                                self.x_min, self.x_max, valinit=self.x_max)
        self.slider_y_min = Slider(self.ax_y_min, 'Y Min', 
                                self.y_min, self.y_max, valinit=self.y_min)
        self.slider_y_max = Slider(self.ax_y_max, 'Y Max', 
                                self.y_min, self.y_max, valinit=self.y_max)
        
        # Gate buttons - reduced to 4 buttons
        self.button_gate = Button(self.ax_gate_btn, 'Draw Gate')
        self.button_save = Button(self.ax_save_gate, 'Save Gate')
        self.button_logic = Button(self.ax_logic_gate, 'Logic Gate')
        self.button_clear_all = Button(self.ax_clear_all, 'Clear All Gates')
        
        print("before connecting events")
        
        # Connect events
        self.radio_x.on_clicked(self.on_x_channel_change)
        self.radio_y.on_clicked(self.on_y_channel_change)
        self.radio_x_log.on_clicked(self.on_x_log_change)
        self.radio_y_log.on_clicked(self.on_y_log_change)
        self.radio_prop.on_clicked(self.on_prop_change)
        self.radio_population.on_clicked(self.on_population_change)
        self.slider_x_min.on_changed(self.on_range_change)
        self.slider_x_max.on_changed(self.on_range_change)
        self.slider_y_min.on_changed(self.on_range_change)
        self.slider_y_max.on_changed(self.on_range_change)
        self.button_gate.on_clicked(self.start_gating)
        self.button_save.on_clicked(self.save_gate)
        self.button_logic.on_clicked(self.create_logic_gate)
        self.button_clear_all.on_clicked(self.clear_all_gates)
        
        print("after connecting events")
        
        # Connect click event for gate info panel
        self.fig.canvas.mpl_connect('button_press_event', self.on_gate_info_click)
        
        # Add window close event to reset naming mode
        self.fig.canvas.mpl_connect('close_event', self.on_main_window_close)
        
        # Add compact labels
        self.ax_x_dropdown.set_title('X Axis', fontsize=8, weight='bold')
        self.ax_y_dropdown.set_title('Y Axis', fontsize=8, weight='bold')
        self.ax_x_log_dropdown.set_title('X Scale', fontsize=8, weight='bold')
        self.ax_y_log_dropdown.set_title('Y Scale', fontsize=8, weight='bold')
        self.ax_prop_dropdown.set_title('Display', fontsize=8, weight='bold')
        self.ax_pop_dropdown.set_title('Population', fontsize=8, weight='bold')
        self.ax_gate_info.set_title('Gates', fontsize=9, weight='bold')

        self.update_gate_info_display()
        print("setup complete")

    def update_population_dropdown(self):
        """Update the population dropdown with available gates"""
        print("Updating population dropdown")
        
        # Get all available gates
        all_gate_names = set()
        for gates_dict in self.gates.values():
            all_gate_names.update(gates_dict.keys())
        
        # Create dropdown options
        population_options = ['All'] + sorted(list(all_gate_names))
        
        # CHANGED: Increased limit since box is now taller
        max_options = 15  # Increased from 8 to 15 options
        if len(population_options) > max_options:
            # Keep 'All' and most recent gates
            population_options = population_options[:max_options-1] + ['...']
        
        # Clear existing dropdown
        self.ax_pop_dropdown.clear()
        
        # Create new radio buttons for population selection
        self.radio_population = RadioButtons(self.ax_pop_dropdown, population_options)
        
        # Make font smaller for population radio buttons
        for label in self.radio_population.labels:
            label.set_fontsize(5)  # CHANGED: Even smaller font to fit more options
        
        # Set active selection
        try:
            current_index = population_options.index(self.input_population)
            self.radio_population.set_active(current_index)
        except ValueError:
            self.radio_population.set_active(0)  # Default to 'All'
            self.input_population = 'All'
        
        # Reconnect the event handler
        self.radio_population.on_clicked(self.on_population_change)
        
        # Add title back
        self.ax_pop_dropdown.set_title('Population', fontsize=9, weight='bold')

    def show_gate_edit_menu(self, gate_name):
        """Show gate edit menu"""
        if self.naming_mode:
            return
        
        self.naming_mode = True
        
        # FIXED: Create edit dialog with proper TkAgg handling
        self.gate_name_figure = plt.figure(figsize=(6, 4))
        self.gate_name_figure.suptitle(f'Edit Gate: {gate_name}', fontsize=14, weight='bold')
        
        # FIXED: Store references to prevent garbage collection
        self.edit_dialog_widgets = {}
        
        def on_edit_close(event):
            self.naming_mode = False
        self.gate_name_figure.canvas.mpl_connect('close_event', on_edit_close)
        
        # Rename section
        ax_rename_text = plt.axes([0.1, 0.6, 0.8, 0.15])
        self.edit_dialog_widgets['textbox_rename'] = TextBox(ax_rename_text, 'New Name: ', initial=gate_name)
        
        # Buttons
        ax_rename = plt.axes([0.1, 0.4, 0.25, 0.15])
        ax_delete = plt.axes([0.4, 0.4, 0.25, 0.15])
        ax_cancel = plt.axes([0.7, 0.4, 0.25, 0.15])
        
        self.edit_dialog_widgets['button_rename'] = Button(ax_rename, 'Rename')
        self.edit_dialog_widgets['button_delete'] = Button(ax_delete, 'Delete')
        self.edit_dialog_widgets['button_cancel'] = Button(ax_cancel, 'Cancel')
        
        def on_rename_clicked(event):
            print("Rename button clicked")
            try:
                new_name = self.edit_dialog_widgets['textbox_rename'].text.strip().replace(' ', '_')
                if new_name and new_name != gate_name:
                    plt.close(self.gate_name_figure)
                    self.naming_mode = False
                    self.rename_gate(gate_name, new_name)
                else:
                    print("Invalid or unchanged name")
            except Exception as e:
                print(f"Error in rename: {e}")
        
        def on_delete_clicked(event):
            print("Delete button clicked")
            try:
                plt.close(self.gate_name_figure)
                self.naming_mode = False
                self.delete_gate(gate_name)
            except Exception as e:
                print(f"Error in delete: {e}")
        
        def on_cancel_edit(event):
            print("Cancel edit clicked")
            try:
                plt.close(self.gate_name_figure)
                self.naming_mode = False
            except Exception as e:
                print(f"Error in cancel: {e}")
        
        self.edit_dialog_widgets['button_rename'].on_clicked(on_rename_clicked)
        self.edit_dialog_widgets['button_delete'].on_clicked(on_delete_clicked)
        self.edit_dialog_widgets['button_cancel'].on_clicked(on_cancel_edit)
        
        # FIXED: Force the dialog to be interactive
        plt.show(block=False)
        self.gate_name_figure.canvas.draw()
        self.gate_name_figure.canvas.flush_events()
        
        # FIXED: Bring window to front
        try:
            self.gate_name_figure.canvas.manager.window.wm_attributes('-topmost', 1)
            self.gate_name_figure.canvas.manager.window.wm_attributes('-topmost', 0)
        except:
            pass
                
    def rename_gate(self, old_name, new_name):
        """Rename a gate"""
        for key in self.gates:
            if old_name in self.gates[key]:
                self.gates[key][new_name] = self.gates[key].pop(old_name)
                break
        
        # Update input population if it was using the renamed gate
        if self.input_population == old_name:
            self.input_population = new_name
        
        self.save_gates()
        self.update_population_dropdown()
        self.update_gate_info_display()
        self.update_plot()
        print(f"Gate renamed from '{old_name}' to '{new_name}'")

    def delete_gate(self, gate_name):
        """Delete a gate"""
        for key in self.gates:
            if gate_name in self.gates[key]:
                del self.gates[key][gate_name]
                break
        
        # Reset input population if it was using the deleted gate
        if self.input_population == gate_name:
            self.input_population = "All"
        
        self.save_gates()
        self.update_population_dropdown()
        self.update_gate_info_display()
        self.update_plot()
        print(f"Gate '{gate_name}' deleted")
        
    def get_plot_data(self, data):
        print("Getting plot data")
        """Apply log transformation if needed"""
        x_data = data[self.x_channel].copy()
        y_data = data[self.y_channel].copy()
        
        if self.x_log:
            x_data = np.log10(x_data + self.epsilon)
        if self.y_log:
            y_data = np.log10(y_data + self.epsilon)
            
        return x_data, y_data
        
    def sample_data(self):
        print("Sampling data")
        """Sample data based on display proportion from current population"""
        current_data = self.get_current_population_data()
        n_total = len(current_data)
        n_sample = int(n_total * self.display_prop)
        
        if n_sample >= n_total:
            return current_data
        else:
            random.seed(42)
            indices = random.sample(range(n_total), n_sample)
            return current_data.iloc[indices]
    
    def update_plot(self):
        print("Updating plot")
        """Update the scatter plot"""
        self.ax_main.clear()
        
        # Get sampled data
        plot_data = self.sample_data()
        
        # Get plot coordinates
        x_plot, y_plot = self.get_plot_data(plot_data)
        
        # Create scatter plot
        scatter = self.ax_main.scatter(x_plot, y_plot, alpha=0.6, s=1)
        
        # Set labels
        x_label = f'log10({self.x_channel})' if self.x_log else self.x_channel
        y_label = f'log10({self.y_channel})' if self.y_log else self.y_channel
        
        self.ax_main.set_xlabel(x_label, fontsize=12)
        self.ax_main.set_ylabel(y_label, fontsize=12)
        
        # Display current population above the plot
        pop_display = f"Population: {self.input_population}"
        plot_title = f'{y_label} vs {x_label}\n{pop_display}\nShowing {len(plot_data):,} points ({self.display_prop*100:.1f}%)'
        
        self.ax_main.set_title(plot_title, fontsize=11, weight='bold')
        
        # Set ranges
        self.ax_main.set_xlim(self.slider_x_min.val, self.slider_x_max.val)
        self.ax_main.set_ylim(self.slider_y_min.val, self.slider_y_max.val)
        
        # Add grid
        self.ax_main.grid(True, alpha=0.3)
        
        # Draw existing gates
        self.draw_gates()
        
        # Add point count info
        visible_x = (x_plot >= self.slider_x_min.val) & (x_plot <= self.slider_x_max.val)
        visible_y = (y_plot >= self.slider_y_min.val) & (y_plot <= self.slider_y_max.val)
        visible_count = np.sum(visible_x & visible_y)
        
        self.ax_main.text(0.02, 0.98, f'Visible: {visible_count:,}', 
                        transform=self.ax_main.transAxes, 
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Update gate info display
        self.update_gate_info_display()
        
        # Refresh the plot
        self.fig.canvas.draw()
        
    def draw_gates(self):
        print("Drawing gates")
        """Draw all existing gates on the plot - CHANGED: Remove gate drawing when log scale changes"""
        colors = list(mcolors.TABLEAU_COLORS.values())
        gate_count = 0
        
        # Draw gates from all channel combinations, not just current
        for key, gates_dict in self.gates.items():
            for gate_name, gate_info in gates_dict.items():
                if gate_info['type'] == 'polygon':
                    # CHANGED: Only show gates if they match EXACT log scale settings
                    if (gate_info['x_channel'] == self.x_channel and 
                        gate_info['y_channel'] == self.y_channel and
                        gate_info['x_log'] == self.x_log and
                        gate_info['y_log'] == self.y_log):
                        
                        # Use vertices as stored (they're in the correct coordinate system)
                        plot_vertices = gate_info['vertices']
                        
                        # Draw polygon
                        polygon = Polygon(plot_vertices, fill=False, 
                                        edgecolor=colors[gate_count % len(colors)], 
                                        linewidth=2, alpha=0.8)
                        self.ax_main.add_patch(polygon)
                        
                        # Add label
                        center_x = np.mean([v[0] for v in plot_vertices])
                        center_y = np.mean([v[1] for v in plot_vertices])
                        self.ax_main.text(center_x, center_y, gate_name, 
                                        ha='center', va='center', 
                                        bbox=dict(boxstyle='round,pad=0.3', 
                                                facecolor='white', alpha=0.8),
                                        fontsize=8, weight='bold')
                
                elif gate_info['type'] == 'logic':
                    # Show logic gates as text labels in corner
                    x_pos = self.ax_main.get_xlim()[0] + 0.02 * (self.ax_main.get_xlim()[1] - self.ax_main.get_xlim()[0])
                    y_pos = self.ax_main.get_ylim()[1] - 0.05 * (gate_count + 1) * (self.ax_main.get_ylim()[1] - self.ax_main.get_ylim()[0])
                    
                    self.ax_main.text(x_pos, y_pos, f"{gate_name} (Logic)", 
                                    ha='left', va='top', 
                                    bbox=dict(boxstyle='round,pad=0.3', 
                                            facecolor='lightgreen', alpha=0.8),
                                    fontsize=8, weight='bold')
                
                gate_count += 1
    
    def start_gating(self, event):
        print("Starting gating")
        """Start polygon selection for gating"""
        if self.polygon_selector is not None:
            self.polygon_selector.disconnect_events()
        
        def onselect(verts):
            self.current_vertices = list(verts)
            print(f"Polygon selected with {len(verts)} vertices: {verts}")
            # Note: Removed auto-save functionality - now requires manual completion
        
        try:
            self.polygon_selector = PolygonSelector(self.ax_main, onselect, useblit=True)
            print("PolygonSelector created successfully")
            print("Draw a polygon gate on the plot. Click 'Save Gate' when finished.")
        except Exception as e:
            print(f"ERROR creating PolygonSelector: {e}")
            import traceback
            traceback.print_exc()
   
    def save_gate(self, event):
        print("Saving gate")
        """Save the current polygon as a named gate with user input"""
        if len(self.current_vertices) < 3:
            print("Please draw a polygon gate first (at least 3 points)")
            return
        
        if self.naming_mode:
            print("Already in naming mode")
            return
        
        self.naming_mode = True
        
        # FIXED: Create figure with proper event processing for TkAgg
        self.gate_name_figure = plt.figure(figsize=(6, 3))
        self.gate_name_figure.suptitle('Enter Gate Name', fontsize=14, weight='bold')
        
        # FIXED: Store references to prevent garbage collection
        self.gate_dialog_widgets = {}
        
        def on_gate_name_close(event):
            print("Gate name window closed")
            self.naming_mode = False
            if hasattr(self, 'gate_save_completed') and not self.gate_save_completed:
                self.current_vertices = []
                if self.polygon_selector is not None:
                    self.polygon_selector.disconnect_events()
                    self.polygon_selector = None
        
        self.gate_name_figure.canvas.mpl_connect('close_event', on_gate_name_close)
        
        # Create text input area
        ax_text = plt.axes([0.2, 0.5, 0.6, 0.15])
        self.gate_dialog_widgets['textbox'] = TextBox(ax_text, 'Gate Name: ', initial=f"Gate_{self.gate_counter}")
        
        # Create OK and Cancel buttons
        ax_ok = plt.axes([0.3, 0.2, 0.15, 0.15])
        ax_cancel = plt.axes([0.55, 0.2, 0.15, 0.15])
        
        self.gate_dialog_widgets['button_ok'] = Button(ax_ok, 'OK')
        self.gate_dialog_widgets['button_cancel'] = Button(ax_cancel, 'Cancel')
        
        self.gate_save_completed = False
        
        def on_ok_clicked(event):
            print("OK button clicked - processing...")
            try:
                gate_name = self.gate_dialog_widgets['textbox'].text.strip()
                if not gate_name:
                    gate_name = f"Gate_{self.gate_counter}"
                
                print(f"Gate name: {gate_name}")
                
                # Check if gate name already exists
                all_gate_names = set()
                for gates_dict in self.gates.values():
                    all_gate_names.update(gates_dict.keys())
                
                if gate_name in all_gate_names:
                    print(f"Warning: Gate '{gate_name}' already exists. Overwriting...")
                
                self.gate_save_completed = True
                
                # Close dialog first
                plt.close(self.gate_name_figure)
                self.naming_mode = False
                
                # Then save gate
                self.complete_gate_save(gate_name)
                
            except Exception as e:
                print(f"Error in on_ok_clicked: {e}")
                import traceback
                traceback.print_exc()
        
        def on_cancel_clicked(event):
            print("Cancel button clicked - processing...")
            try:
                self.gate_save_completed = False
                
                # Clear current vertices
                self.current_vertices = []
                if self.polygon_selector is not None:
                    self.polygon_selector.disconnect_events()
                    self.polygon_selector = None
                
                plt.close(self.gate_name_figure)
                self.naming_mode = False
                print("Gate creation cancelled")
                
            except Exception as e:
                print(f"Error in on_cancel_clicked: {e}")
                import traceback
                traceback.print_exc()
        
        # FIXED: Connect events and force event processing
        self.gate_dialog_widgets['button_ok'].on_clicked(on_ok_clicked)
        self.gate_dialog_widgets['button_cancel'].on_clicked(on_cancel_clicked)
        
        # FIXED: Force the dialog to be interactive
        plt.show(block=False)  # Non-blocking show
        self.gate_name_figure.canvas.draw()
        self.gate_name_figure.canvas.flush_events()
        
        # FIXED: Bring window to front (TkAgg specific)
        try:
            self.gate_name_figure.canvas.manager.window.wm_attributes('-topmost', 1)
            self.gate_name_figure.canvas.manager.window.wm_attributes('-topmost', 0)
        except:
            pass
        
    def on_main_window_close(self, event):
        """Reset naming mode when main window is closed"""
        self.naming_mode = False
        
    def create_logic_gate(self, event):
        print("Creating logic gate")
        
        # Get available gates
        all_gates = set()
        for gates_dict in self.gates.values():
            all_gates.update(gates_dict.keys())

        if len(all_gates) < 1:
            print("Need at least one existing gate to create logic gate")
            return

        if self.naming_mode:
            print("Already in naming mode")
            return
        
        self.naming_mode = True
        gate_list = list(sorted(all_gates))
        
        # FIXED: Create logic gate dialog with proper TkAgg handling
        self.gate_name_figure = plt.figure(figsize=(10, 6))
        self.gate_name_figure.suptitle('Create Logic Gate', fontsize=14, weight='bold')
        
        # FIXED: Store references to prevent garbage collection
        self.logic_dialog_widgets = {}
        
        def on_logic_gate_close(event):
            self.naming_mode = False
        self.gate_name_figure.canvas.mpl_connect('close_event', on_logic_gate_close)
        
        # Variables to store selections - initialize with defaults
        selected_gate1 = [gate_list[0]]
        selected_operation = ['AND']
        selected_gate2 = [gate_list[0] if len(gate_list) > 0 else None]
        
        # Gate 1 selection
        ax_gate1 = plt.axes([0.1, 0.7, 0.35, 0.2])
        ax_gate1.set_title('Select First Gate', fontsize=10, weight='bold')
        self.logic_dialog_widgets['radio_gate1'] = RadioButtons(ax_gate1, gate_list)
        self.logic_dialog_widgets['radio_gate1'].set_active(0)
        
        def on_gate1_select(label):
            selected_gate1[0] = label
        self.logic_dialog_widgets['radio_gate1'].on_clicked(on_gate1_select)
        
        # Operation selection
        operations = ['AND', 'OR', 'AND NOT', 'OR NOT', 'NOT']
        ax_operation = plt.axes([0.55, 0.7, 0.35, 0.2])
        ax_operation.set_title('Select Operation', fontsize=10, weight='bold')
        self.logic_dialog_widgets['radio_operation'] = RadioButtons(ax_operation, operations)
        self.logic_dialog_widgets['radio_operation'].set_active(0)
        
        def on_operation_select(label):
            selected_operation[0] = label
        self.logic_dialog_widgets['radio_operation'].on_clicked(on_operation_select)
        
        # Gate 2 selection (for AND/OR operations)
        ax_gate2 = plt.axes([0.1, 0.45, 0.35, 0.2])
        ax_gate2.set_title('Select Second Gate (if needed)', fontsize=10, weight='bold')
        self.logic_dialog_widgets['radio_gate2'] = RadioButtons(ax_gate2, gate_list)
        self.logic_dialog_widgets['radio_gate2'].set_active(0)
        
        def on_gate2_select(label):
            selected_gate2[0] = label
        self.logic_dialog_widgets['radio_gate2'].on_clicked(on_gate2_select)
        
        # Gate name input
        ax_name = plt.axes([0.1, 0.3, 0.8, 0.08])
        self.logic_dialog_widgets['textbox_name'] = TextBox(ax_name, 'Logic Gate Name: ', initial=f"Logic_{self.gate_counter}")
        
        # Buttons
        ax_ok = plt.axes([0.3, 0.1, 0.15, 0.1])
        ax_cancel = plt.axes([0.55, 0.1, 0.15, 0.1])
        
        self.logic_dialog_widgets['button_ok'] = Button(ax_ok, 'Create')
        self.logic_dialog_widgets['button_cancel'] = Button(ax_cancel, 'Cancel')
        
        def on_create_clicked(event):
            print("Create logic gate clicked")
            try:
                gate1 = selected_gate1[0]
                operation = selected_operation[0]
                
                # Fixed logic expression creation
                if operation == 'NOT':
                    logic_expr = f"NOT {gate1}"
                elif operation == 'AND NOT':
                    gate2 = selected_gate2[0]
                    logic_expr = f"{gate1} AND NOT {gate2}"
                elif operation == 'OR NOT':
                    gate2 = selected_gate2[0]
                    logic_expr = f"{gate1} OR NOT {gate2}"
                else:  # AND or OR
                    gate2 = selected_gate2[0]
                    logic_expr = f"{gate1} {operation} {gate2}"
                
                gate_name = self.logic_dialog_widgets['textbox_name'].text.strip()
                if not gate_name:
                    gate_name = f"Logic_{self.gate_counter}"
                
                # Handle spaces in gate names by replacing with underscores
                gate_name = gate_name.replace(' ', '_')
                
                # Check if gate name already exists
                all_existing_gates = set()
                for gates_dict in self.gates.values():
                    all_existing_gates.update(gates_dict.keys())
                
                if gate_name in all_existing_gates:
                    print(f"Gate name '{gate_name}' already exists. Please choose a different name.")
                    return
                
                plt.close(self.gate_name_figure)
                self.naming_mode = False
                
                # Create the logic gate
                self.create_logical_gate(gate_name, logic_expr)
                
                # Update population dropdown
                self.update_population_dropdown()
                
                print(f"Logic gate '{gate_name}' created: {logic_expr}")
                
            except Exception as e:
                print(f"Error creating logic gate: {e}")
                import traceback
                traceback.print_exc()
        
        def on_cancel_logic(event):
            print("Cancel logic gate clicked")
            try:
                plt.close(self.gate_name_figure)
                self.naming_mode = False
                print("Logic gate creation cancelled")
            except Exception as e:
                print(f"Error cancelling: {e}")
        
        self.logic_dialog_widgets['button_ok'].on_clicked(on_create_clicked)
        self.logic_dialog_widgets['button_cancel'].on_clicked(on_cancel_logic)
        
        # FIXED: Force the dialog to be interactive
        plt.show(block=False)
        self.gate_name_figure.canvas.draw()
        self.gate_name_figure.canvas.flush_events()
        
        # FIXED: Bring window to front
        try:
            self.gate_name_figure.canvas.manager.window.wm_attributes('-topmost', 1)
            self.gate_name_figure.canvas.manager.window.wm_attributes('-topmost', 0)
        except:
            pass

    def create_logical_gate(self, gate_name, logic_expr):
        print("Creating logical gate")
        """Create a gate based on logical expression"""
        try:
            # Test the logic expression with current data
            test_mask = self.evaluate_logic_expression_for_data(logic_expr, self.df)
            
            if test_mask is not None:
                print("Storing logic gate")
                # Store logical gate in "All" key so it's accessible from any channel combination
                key = "All"  # Use a universal key for logic gates
                if key not in self.gates:
                    self.gates[key] = {}
                
                self.gates[key][gate_name] = {
                    'type': 'logic',
                    'expression': logic_expr,
                    'x_axis': None,  # Logic gates aren't tied to specific channels
                    'y_axis': None,
                    'x_log': None,
                    'y_log': None,
                    'created': datetime.now().isoformat()
                }
                
                # CHANGED: Increment gate counter for logic gates too
                self.gate_counter += 1
                
                self.save_gates()
                self.save_gate_cells(gate_name, test_mask)
                
                points_inside = np.sum(test_mask)
                total_points = len(test_mask)
                percentage = 100 * points_inside / total_points if total_points > 0 else 0
                
                print(f"Logic gate '{gate_name}' created: {logic_expr}")
                print(f"Points inside gate: {points_inside} / {total_points} ({percentage:.2f}%)")
                
                # Update displays
                self.update_population_dropdown()
                self.update_gate_info_display()
                self.update_plot()
                
            else:
                print("Failed to create logic gate - invalid expression")
                
        except Exception as e:
            print(f"Error creating logic gate: {e}")
    
    def evaluate_logic_expression_for_data(self, expr, data):
        """Evaluate logical expression for specific data"""
        print(f"Evaluating logic expression for data: {expr}")
        
        # Replace gate names with their masks
        tokens = expr.replace('(', ' ( ').replace(')', ' ) ').split()
            
        # Create masks for each gate mentioned
        gate_masks = {}
        for token in tokens:
            if token not in ['AND', 'OR', 'NOT', '(', ')']:
                # This should be a gate name
                mask = self.get_gate_mask_for_data(token, data)
                if mask is not None:
                    gate_masks[token] = mask
                else:
                    print(f"Gate '{token}' not found")
                    return np.zeros(len(data), dtype=bool)
        
        # Build evaluation string
        eval_expr = expr
        for gate_name in gate_masks.keys():
            eval_expr = eval_expr.replace(gate_name, f"gate_masks['{gate_name}']")
        eval_expr = eval_expr.replace('AND', '&').replace('OR', '|').replace('NOT', '~')
        
        # Evaluate
        try:
            result_mask = eval(eval_expr)
            print(f"Logic expression result: {np.sum(result_mask)} points")
            return result_mask
        except Exception as e:
            print(f"Error evaluating expression: {e}")
            return np.zeros(len(data), dtype=bool)
            print(f"Error evaluating expression: {e}")
            return None
    
    def get_gate_mask_for_data(self, gate_name, data):
        print("Getting gate mask for data")
        """Get mask for a specific gate applied to given data"""
        # Find the gate across all channel combinations
        for key, gates_dict in self.gates.items():
            if gate_name in gates_dict:
                gate_info = gates_dict[gate_name]
                
                if gate_info['type'] == 'polygon':
                    # Calculate mask for polygon gate
                    x_data = data[gate_info['x_channel']].values
                    y_data = data[gate_info['y_channel']].values
                    
                    # Apply log transform if needed
                    if gate_info['x_log']:
                        x_data = np.log10(x_data + self.epsilon)
                    if gate_info['y_log']:
                        y_data = np.log10(y_data + self.epsilon)
                    
                    # Create path from vertices
                    vertices = gate_info['vertices']
                    if gate_info['x_log'] or gate_info['y_log']:
                        # Transform vertices if needed
                        vertices = []
                        for x, y in gate_info['vertices']:
                            plot_x = np.log10(x + self.epsilon) if gate_info['x_log'] else x
                            plot_y = np.log10(y + self.epsilon) if gate_info['y_log'] else y
                            vertices.append([plot_x, plot_y])
                    
                    path = Path(vertices)
                    points = np.column_stack([x_data, y_data])
                    
                    # Handle NaN values
                    valid_points = ~(np.isnan(x_data) | np.isnan(y_data))
                    mask = np.zeros(len(data), dtype=bool)
                    if np.any(valid_points):
                        mask[valid_points] = path.contains_points(points[valid_points])
                    
                    return mask
                
                elif gate_info['type'] == 'logic':
                    # Recursively evaluate logic gate
                    return self.evaluate_logic_expression_for_data(gate_info['expression'], data)
        
        return np.zeros(len(data), dtype=bool)
    
    def calculate_gate_mask(self, gate_info, key):
        print( "Calculating gate mask")
        """Calculate mask for a gate"""
        if gate_info['type'] == 'polygon':
            x_data = self.df[gate_info['x_channel']]
            y_data = self.df[gate_info['y_channel']]
            
            if gate_info['x_log']:
                x_data = np.log10(x_data + self.epsilon)
            if gate_info['y_log']:
                y_data = np.log10(y_data + self.epsilon)
            
            # Create appropriate vertices for testing
            test_vertices = gate_info['vertices']
            if gate_info['x_log'] or gate_info['y_log']:
                test_vertices = []
                for x, y in gate_info['vertices']:
                    test_x = np.log10(x + self.epsilon) if gate_info['x_log'] else x
                    test_y = np.log10(y + self.epsilon) if gate_info['y_log'] else y
                    test_vertices.append([test_x, test_y])
            
            path = Path(test_vertices)
            points = np.column_stack([x_data, y_data])
            return path.contains_points(points)
        
        return np.zeros(len(self.df), dtype=bool)
    
    def calculate_and_save_gate_stats(self, gate_name, key):
        print("Calculating and saving gate stats")
        """Calculate statistics and save cells for a gate"""
        gate_info = self.gates[key][gate_name]
        mask = self.calculate_gate_mask(gate_info, key)
        
        # Save gate cell information
        self.save_gate_cells(gate_name, mask)
        
        # Print statistics
        points_inside = np.sum(mask)
        total_points = len(mask)
        percentage = 100 * points_inside / total_points if total_points > 0 else 0
        
        print(f"\nGate '{gate_name}' statistics:")
        print(f"Points inside gate: {points_inside} / {total_points} ({percentage:.2f}%)")
        
        if gate_info['type'] == 'polygon':
            print("Coordinates (original scale):")
            for i, (x, y) in enumerate(gate_info['vertices']):
                print(f"  Point {i+1}: ({x:.3f}, {y:.3f})")
    
    def save_gate_cells(self, gate_name, mask):
        print("Saving gate cells")
        """Save cell indices that fall within a gate"""
        cell_data = {
            'gate_name': gate_name,
            'cell_indices': self.df.index[mask].tolist(),
            'cell_count': int(np.sum(mask)),
            'total_cells': len(self.df),
            'percentage': float(100 * np.sum(mask) / len(self.df)),
            'timestamp': datetime.now().isoformat()
        }
        
        filename = os.path.join(self.save_dir, f"cells_{gate_name}.json")
        with open(filename, 'w') as f:
            json.dump(cell_data, f, indent=2)
        
        print(f"Cell data saved to {filename}")
    
    def save_gates(self):
        print("Saving gates")
        """Save all gates to file"""
        filename = os.path.join(self.save_dir, "gates_definition.json")
        with open(filename, 'w') as f:
            json.dump(self.gates, f, indent=2)
        print(f"Gates saved to {filename}")
    
    def complete_gate_save(self, gate_name):
        """Complete the gate saving process with the given name"""
        print(f"Starting complete_gate_save for: {gate_name}")
        print(f"Current vertices: {self.current_vertices}")
        print(f"X channel: {self.x_channel}, Y channel: {self.y_channel}")
        print(f"X log: {self.x_log}, Y log: {self.y_log}")
        
        # Handle spaces in gate names by replacing with underscores
        gate_name = gate_name.replace(' ', '_')
        
        self.gate_counter += 1
        
        # Check if we have vertices
        if not self.current_vertices:
            print("ERROR: No vertices to save!")
            return
        
        # CHANGED: Add error handling for coordinate transformation
        original_vertices = []
        plot_vertices = []
        
        try:
            for plot_x, plot_y in self.current_vertices:
                # Store plot coordinates (for current display)
                plot_vertices.append([plot_x, plot_y])
                
                # Convert to original coordinates with error handling
                try:
                    if self.x_log:
                        if plot_x <= 0:  # Check for invalid log values
                            print(f"WARNING: Invalid log value for x: {plot_x}")
                            orig_x = plot_x  # Use as-is if invalid
                        else:
                            orig_x = 10**(plot_x) - self.epsilon
                    else:
                        orig_x = plot_x
                    
                    if self.y_log:
                        if plot_y <= 0:  # Check for invalid log values
                            print(f"WARNING: Invalid log value for y: {plot_y}")
                            orig_y = plot_y  # Use as-is if invalid
                        else:
                            orig_y = 10**(plot_y) - self.epsilon
                    else:
                        orig_y = plot_y
                    
                    original_vertices.append([orig_x, orig_y])
                    print(f"Converted ({plot_x}, {plot_y}) -> ({orig_x}, {orig_y})")
                    
                except Exception as coord_error:
                    print(f"ERROR converting coordinates ({plot_x}, {plot_y}): {coord_error}")
                    # Fallback: use plot coordinates as original
                    original_vertices.append([plot_x, plot_y])
            
            print(f"Final original vertices: {original_vertices}")
            
        except Exception as e:
            print(f"ERROR in coordinate conversion: {e}")
            return
        
        # Store the gate
        key = self.get_channel_key()
        if key not in self.gates:
            self.gates[key] = {}
        
        try:
            self.gates[key][gate_name] = {
                'type': 'polygon',
                'vertices': original_vertices,  # Always store in original coordinates
                'x_channel': self.x_channel,
                'y_channel': self.y_channel,
                'x_log': self.x_log,
                'y_log': self.y_log,
                'created': datetime.now().isoformat()
            }
            
            print(f"Gate stored successfully: {gate_name}")
            
            # Save gates and calculate statistics
            self.save_gates()
            self.calculate_and_save_gate_stats(gate_name, key)
            
            # Update population dropdown
            self.update_population_dropdown()
            
            # Clear current selection
            self.current_vertices = []
            if self.polygon_selector is not None:
                self.polygon_selector.disconnect_events()
                self.polygon_selector = None
            
            # Update plot
            self.update_plot()
            print(f"Gate '{gate_name}' saved successfully")
            
        except Exception as e:
            print(f"ERROR saving gate: {e}")
            import traceback
            traceback.print_exc()

    def load_gates(self):
        print("Loading gates")
        """Load gates from file"""
        filename = os.path.join(self.save_dir, "gates_definition.json")
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    self.gates = json.load(f)
                
                # Update gate counter to avoid conflicts
                max_gate_num = 0
                for gates_dict in self.gates.values():
                    for gate_name in gates_dict.keys():
                        if gate_name.startswith('Gate_'):
                            try:
                                num = int(gate_name.split('_')[1])
                                max_gate_num = max(max_gate_num, num)
                            except (IndexError, ValueError):
                                pass
                        elif gate_name.startswith('Logic_'):
                            try:
                                num = int(gate_name.split('_')[1])
                                max_gate_num = max(max_gate_num, num)
                            except (IndexError, ValueError):
                                pass
                
                self.gate_counter = max_gate_num + 1
                print(f"Loaded {sum(len(gates_dict) for gates_dict in self.gates.values())} gates")
            except Exception as e:
                print(f"Error loading gates: {e}")
                self.gates = {}
        else:
            print("No existing gates file found")
    
    def update_gate_info_display(self):
        """Update the gate information display panel - simplified version"""
        print("Updating gate info display")
        self.ax_gate_info.clear()
        self.ax_gate_info.set_title('Gates', fontsize=9, weight='bold')
        
        # Get all gates
        all_gates_info = []
        for key, gates_dict in self.gates.items():
            for gate_name, gate_info in gates_dict.items():
                # Calculate statistics for this gate
                try:
                    if gate_info['type'] == 'logic':
                        mask = self.evaluate_logic_expression_for_data(gate_info['expression'], self.df)
                    else:
                        mask = self.calculate_gate_mask(gate_info, key)
                    
                    if mask is not None:
                        points_inside = np.sum(mask)
                        total_points = len(self.df)
                        percentage = 100 * points_inside / total_points if total_points > 0 else 0
                        
                        all_gates_info.append({
                            'name': gate_name,
                            'type': gate_info['type'],
                            'points': points_inside,
                            'percentage': percentage,
                            'key': key,
                            'gate_info': gate_info
                        })
                    else:
                        print(f"Failed to calculate mask for gate {gate_name}")
                except Exception as e:
                    print(f"Error calculating stats for gate {gate_name}: {e}")
                    continue
        
        if not all_gates_info:
            self.ax_gate_info.text(0.5, 0.5, 'No gates created yet', ha='center', va='center', 
                                transform=self.ax_gate_info.transAxes, fontsize=10)
            self.ax_gate_info.set_xlim(0, 1)
            self.ax_gate_info.set_ylim(0, 1)
            self.ax_gate_info.axis('off')
            return
        
        # CHANGED: Squeezed spacing for more gates
        y_pos = 0.98
        row_height = 0.08  # Reduced from 0.1 to squeeze more gates
        
        for i, gate_info in enumerate(all_gates_info):
            if y_pos < 0.05:
                # Add "..." if there are more gates
                self.ax_gate_info.text(0.5, y_pos, f'... and {len(all_gates_info) - i} more gates', 
                                    ha='center', va='top', fontsize=7, style='italic',
                                    transform=self.ax_gate_info.transAxes)
                break
            
            # Gate name
            display_name = gate_info['name']
            color = 'lightblue' if gate_info['type'] == 'polygon' else 'lightgreen'
            
            # Gate name (smaller box to make room for buttons)
            self.ax_gate_info.text(0.02, y_pos, display_name, 
                                ha='left', va='top', fontsize=7, weight='bold',  # CHANGED: Smaller font
                                bbox=dict(boxstyle='round,pad=0.1', facecolor=color, alpha=0.8),  # CHANGED: Less padding
                                transform=self.ax_gate_info.transAxes)
            
            # Buttons on same line, properly spaced and smaller
            button_y = y_pos
            self.ax_gate_info.text(0.42, button_y, 'Data', 
                                ha='center', va='top', fontsize=5, color='green', weight='bold',  # CHANGED: Smaller font
                                bbox=dict(boxstyle='round,pad=0.05', facecolor='white', alpha=0.8),  # CHANGED: Less padding
                                transform=self.ax_gate_info.transAxes,
                                picker=True, gid=f"data_{gate_info['name']}")
            
            self.ax_gate_info.text(0.62, button_y, 'Edit', 
                                ha='center', va='top', fontsize=5, color='blue', weight='bold',  # CHANGED: Smaller font
                                bbox=dict(boxstyle='round,pad=0.05', facecolor='white', alpha=0.8),  # CHANGED: Less padding
                                transform=self.ax_gate_info.transAxes,
                                picker=True, gid=f"edit_{gate_info['name']}")
            
            self.ax_gate_info.text(0.82, button_y, 'Del', 
                                ha='center', va='top', fontsize=5, color='red', weight='bold',  # CHANGED: Smaller font
                                bbox=dict(boxstyle='round,pad=0.05', facecolor='white', alpha=0.8),  # CHANGED: Less padding
                                transform=self.ax_gate_info.transAxes,
                                picker=True, gid=f"delete_{gate_info['name']}")
            
            y_pos -= row_height
        
        self.ax_gate_info.set_xlim(0, 1)
        self.ax_gate_info.set_ylim(0, 1)
        self.ax_gate_info.axis('off')

    def clear_gates(self, event=None):
        print("Clearing gates")
        """Clear all gates for current channel combination"""
        key = self.get_channel_key()
        if key in self.gates:
            del self.gates[key]
            self.save_gates()
            
            # Update population dropdown
            self.update_population_dropdown()
            
            # Reset input population to "All" if current selection no longer exists
            self.input_population = "All"
            
            # Update plot and ranges
            self.update_ranges()
            self.update_sliders()
            self.update_plot()
            print("All gates cleared for current channel combination")
        else:
            print("No gates to clear for current channel combination")
    
    def clear_all_gates(self, event=None):
        print("Clearing all gates")
        """Clear all gates across all channel combinations"""
        self.gates = {}
        self.gate_counter = 1  # CHANGED: Reset gate counter to 1
        self.save_gates()
        
        # Reset input population to "All"
        self.input_population = "All"
        
        # Update population dropdown
        self.update_population_dropdown()
        
        # Update plot and ranges
        self.update_ranges()
        self.update_sliders()
        self.update_plot()
        self.update_gate_info_display()
        print("All gates cleared")
    
    def on_x_channel_change(self, label):
        print("Changing X channel")
        self.x_channel = label
        self.update_ranges()
        self.update_sliders()
        self.update_plot()
    
    def on_y_channel_change(self, label):
        print("Changing Y channel")
        self.y_channel = label
        self.update_ranges()
        self.update_sliders()
        self.update_plot()
    
    def on_prop_change(self, label):
        print("Changing display proportion")
        prop_map = {'0.1%': 0.001, '1%': 0.01, '10%': 0.1, '100%': 1.0}
        self.display_prop = prop_map[label]
        self.update_plot()
    
    def on_x_log_change(self, label):
        print("Changing X log scale")
        self.x_log = (label == 'Log')
        self.update_ranges()
        self.update_sliders()
        self.update_plot()

    def on_y_log_change(self, label):
        print("Changing Y log scale")
        self.y_log = (label == 'Log')
        self.update_ranges()
        self.update_sliders()
        self.update_plot()
    
    def on_population_change(self, label):
        print("Changing population")
        self.input_population = label
        self.update_ranges()
        self.update_sliders()
        self.update_plot()
    
    def on_range_change(self, val):
        print("Changing range slider")
        """Handle range slider changes"""
        self.update_plot()
    
    def update_sliders(self):
        print("Updating sliders")
        """Update slider ranges when channels or log scale change"""
        # Update X sliders
        self.slider_x_min.valmin, self.slider_x_min.valmax = self.x_min, self.x_max
        self.slider_x_max.valmin, self.slider_x_max.valmax = self.x_min, self.x_max
        self.slider_x_min.set_val(self.x_min)
        self.slider_x_max.set_val(self.x_max)
        
        # Update Y sliders
        self.slider_y_min.valmin, self.slider_y_min.valmax = self.y_min, self.y_max
        self.slider_y_max.valmin, self.slider_y_max.valmax = self.y_min, self.y_max
        self.slider_y_min.set_val(self.y_min)
        self.slider_y_max.set_val(self.y_max)

    def on_gate_info_click(self, event):
        """Handle clicks on gate info panel"""
        if event.inaxes == self.ax_gate_info:
            # Check if click is on a text element with a gid
            for child in self.ax_gate_info.get_children():
                if hasattr(child, 'get_gid') and hasattr(child, 'contains') and child.contains(event)[0]:
                    gid = child.get_gid()
                    if gid and ('edit_' in gid or 'delete_' in gid or 'data_' in gid):  # CHANGED: Added data button
                        if gid.startswith('edit_'):
                            gate_name = gid.replace('edit_', '')
                            self.show_gate_edit_menu(gate_name)
                            return
                        elif gid.startswith('delete_'):
                            gate_name = gid.replace('delete_', '')
                            self.delete_gate(gate_name)
                            return
                        elif gid.startswith('data_'):  # CHANGED: Added data button handler
                            gate_name = gid.replace('data_', '')
                            self.show_gate_data(gate_name)
                            return
    
    def show(self):
        print("Showing plot")
        """Display the interactive plot"""
        plt.show()
    
    def get_gates(self):
        print("Getting all gates")
        """Return all created gates"""
        return self.gates.copy()

    def show_gate_data(self, gate_name):
        """Show gate data analysis window"""
        if self.naming_mode:
            return
        
        self.naming_mode = True
        
        # Find the gate
        gate_info = None
        gate_key = None
        for key, gates_dict in self.gates.items():
            if gate_name in gates_dict:
                gate_info = gates_dict[gate_name]
                gate_key = key
                break
        
        if gate_info is None:
            print(f"Gate '{gate_name}' not found")
            self.naming_mode = False
            return
        
        # Calculate gate mask
        try:
            if gate_info['type'] == 'logic':
                mask = self.evaluate_logic_expression_for_data(gate_info['expression'], self.df)
            else:
                mask = self.calculate_gate_mask(gate_info, gate_key)
            
            if mask is None:
                print(f"Failed to calculate mask for gate '{gate_name}'")
                self.naming_mode = False
                return
            
            # Get data for cells in this gate
            gate_data = self.df[mask]
            
        except Exception as e:
            print(f"Error calculating gate data: {e}")
            self.naming_mode = False
            return
        
        # FIXED: Create data display window with proper TkAgg handling
        self.gate_data_figure = plt.figure(figsize=(10, 6))
        self.gate_data_figure.suptitle(f'Data Analysis: {gate_name}', fontsize=14, weight='bold')
        
        # FIXED: Store reference to prevent garbage collection
        self.data_dialog_widgets = {}
        
        def on_data_close(event):
            self.naming_mode = False
        self.gate_data_figure.canvas.mpl_connect('close_event', on_data_close)
        
        # Create text display area
        ax_data = plt.axes([0.1, 0.15, 0.8, 0.75])
        ax_data.axis('off')
        
        # Calculate statistics
        points_inside = np.sum(mask)
        total_points = len(self.df)
        percentage = 100 * points_inside / total_points if total_points > 0 else 0
        
        # Build data text
        data_text = f"Gate: {gate_name}\n"
        data_text += f"Type: {gate_info['type'].title()}\n"
        data_text += f"Points: {points_inside:,} / {total_points:,} ({percentage:.2f}%)\n\n"
        
        if gate_info['type'] == 'polygon':
            data_text += f"Axes: {gate_info['x_channel']} vs {gate_info['y_channel']}\n"
            data_text += f"Log Scale: X={gate_info['x_log']}, Y={gate_info['y_log']}\n"
            data_text += f"Vertices ({len(gate_info['vertices'])}):\n"
            for i, (x, y) in enumerate(gate_info['vertices']):
                data_text += f"  {i+1}: ({x:.3f}, {y:.3f})\n"
        elif gate_info['type'] == 'logic':
            data_text += f"Expression: {gate_info['expression']}\n"
        
        data_text += "\nChannel Statistics (Mean ± Std):\n"
        
        # Calculate channel statistics
        channel_stats = {}
        for channel in self.channels:
            if len(gate_data) > 0:
                channel_data = gate_data[channel]
                mean_val = np.mean(channel_data)
                std_val = np.std(channel_data)
                channel_stats[channel] = {'mean': float(mean_val), 'std': float(std_val)}
                data_text += f"  {channel}: {mean_val:.3f} ± {std_val:.3f}\n"
            else:
                channel_stats[channel] = {'mean': 0.0, 'std': 0.0}
                data_text += f"  {channel}: No data\n"
        
        # Save data analysis to gate info
        gate_info['data_analysis'] = {
            'points_inside': int(points_inside),
            'total_points': int(total_points),
            'percentage': float(percentage),
            'channel_stats': channel_stats,
            'last_updated': datetime.now().isoformat()
        }
        
        # Save updated gate info
        self.save_gates()
        
        # Display the text
        ax_data.text(0.05, 0.95, data_text, transform=ax_data.transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        # Add close button
        ax_close = plt.axes([0.45, 0.02, 0.1, 0.08])
        self.data_dialog_widgets['button_close'] = Button(ax_close, 'Close')
        
        def on_close_data(event):
            print("Close data window clicked")
            try:
                plt.close(self.gate_data_figure)
                self.naming_mode = False
            except Exception as e:
                print(f"Error closing data window: {e}")
        
        self.data_dialog_widgets['button_close'].on_clicked(on_close_data)
        
        # FIXED: Force the dialog to be interactive
        plt.show(block=False)
        self.gate_data_figure.canvas.draw()
        self.gate_data_figure.canvas.flush_events()
        
        # FIXED: Bring window to front
        try:
            self.gate_data_figure.canvas.manager.window.wm_attributes('-topmost', 1)
            self.gate_data_figure.canvas.manager.window.wm_attributes('-topmost', 0)
        except:
            pass

#%%
# Create and show the interactive plot
interactive_plot = InteractiveScatterPlot(df, columns, channels)
interactive_plot.show()

# %%
