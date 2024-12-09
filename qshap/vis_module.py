import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import BoundedIntText, Layout, interactive_output, VBox, HBox
from IPython.display import display, clear_output

class vis:

    @staticmethod
    def rsq(x, color_map_name="Blues", horizontal=False, model_rsq=True, max_feature=10, cutoff=0, title="Shapley R²", xtitle="Feature index", ytitle="R²", rotation=0, label=None, decimal=3, save_name=None):
        """
        Visualize shapley rsq
        
        Parameters:
        -x: 1 dim array, but we recommend array from shapley rsq OvO ...
        -color_map_name: Color map allows to you take a variety color map from matplotlib to customize your visualization color.
        You can try to use "Pastel1", "Pasterl2", "PuBu", "Reds", "Greens" ...
        -horizontal: horizontal plot or not
        -model_rsq: display model rsq or not
        -max_feature: maximum number of features to show
        -cutoff: only value greater than or equal to the cutoff will be displayed
        -xtitle: xtitle, note that it's reversed for horizontal 
        -ytitle: ytitle, note that it's reversed for horizontal 
        -rotation: rotation of the tick
        -label: label for the features
        -title: plot title
        -decimal: decimals to show 
        -save_name: the name of the file if you want to save. None as default without saving.
        """
        
        # Calculate the sum of x
        x_sum = np.sum(x)
        x_len = np.size(x)
        cutoff_feature = np.sum(x >= cutoff)
        show_len = min(x_len, max_feature, cutoff_feature)

        # Sort x and keep track of the indices
        indices = np.argsort(x)[::-1]  # Indices of sorted elements (in descending order)
            
        sorted_x = x[indices]          # Sorted x
        indices = indices[:show_len]
        if label is not None:
            if len(x)!=len(label):
            #error: the length of label and and x must match 
                raise ValueError("The length of the label and x must match")
            else:
                sorted_label = label[indices]
                sorted_label = sorted_label[:show_len]
            
        sorted_x = sorted_x[:show_len]

        # Creating a color map based on the sorted values
        # Normalize the sorted_x values to the range [0, 1]
        normalized_x = (sorted_x - min(sorted_x)) / (max(sorted_x) - min(sorted_x))

        # Get the colormap
        cmap = plt.get_cmap(color_map_name)

        # Generate colors from the colormap
        colors = cmap(normalized_x)
        
        # Define an offset for the text to ensure it doesn't touch the bars
        text_offset = max(sorted_x) * 0.02  # 2% of the max value as the offset
        
        if not horizontal:
            # Create the bar chart
            bars = plt.bar(range(len(sorted_x)), sorted_x, color=colors)

            # Label the x-ticks with the original indices
            if label is not None:
                plt.xticks(range(len(sorted_x)), sorted_label, rotation=rotation)
            else:
                plt.xticks(range(len(sorted_x)), indices, rotation=rotation)
            
            # Adding the text on top of the bars
            for bar in bars:
                height = bar.get_height()  # Get the height of the bar
                # Place the text at the top of the bar, slightly above
                plt.text(bar.get_x() + bar.get_width() / 2, height + text_offset, f'{height:.{decimal}f}', ha='center', va='bottom')
            
            plt.ylim(0, max(sorted_x) * 1.1)

            if model_rsq:
                # Add text for the sum of values
                plt.text(len(sorted_x) - 1, max(sorted_x), f'Model R²: {round(x_sum, 3)}', 
                    horizontalalignment='right', verticalalignment='top', fontsize=12)
                
            plt.xlabel(xtitle)
            plt.ylabel(ytitle)
            
        else:
            bars = plt.barh(range(len(sorted_x)), sorted_x, color=colors)
            
            # Label the y-ticks with the original indices
            if label is not None:
                plt.yticks(range(len(sorted_x)), sorted_label, rotation=rotation)
            else:
                plt.yticks(range(len(sorted_x)), indices, rotation=rotation)
            plt.gca().invert_yaxis()
            
            # Adding the text beside the bars
            for bar in bars:
                width = bar.get_width()  # Get the width of the bar (since it's horizontal)
                # Place the text to the right of the bar, slightly beyond its end
                plt.text(width + text_offset, bar.get_y() + bar.get_height() / 2, f'{width:.{decimal}f}', 
                        ha='left', va='center')
                
            plt.xlim(0, max(sorted_x) * 1.2)
            
            if model_rsq:
                plt.text(max(sorted_x), len(sorted_x) - 1,f'Model R²: {round(x_sum, 3)}', 
                horizontalalignment='right', verticalalignment='bottom', fontsize=12)
                
            plt.xlabel(ytitle)
            plt.ylabel(xtitle)

        # Add title
        plt.title(title)

        
        if save_name is not None:
            name = save_name + ".pdf"
            plt.savefig(name, bbox_inches='tight')

        plt.show()
        
        plt.close()
    
    # vis.rsq(rsq_res)
    # # Change color
    # vis.rsq(rsq_res, color_map_name="Pastel2")

    # import numpy as np

    # # Generate feature names using list comprehension and format them
    # feature_names = np.array([f"feature{i}" for i in range(1, rsq_res.shape[0]+1)])

    # # Give it a name and rotate
    # vis.rsq(rsq_res, color_map_name="Pastel2", label=feature_names, rotation=45)

    # # Give a horizontal plot, hide model rsq, change the number of features to show
    # vis.rsq(rsq_res, color_map_name="PuBu", horizontal=True, model_rsq=False, max_feature=15, save_name="rsq_eg")
    
    @staticmethod
    def loss(loss, save_ind=None, save_prefix="Shapley loss sample", title="Shapley Loss: Sample", color_map_name="Blues", model_rsq=False, decimal=0, xtitle="Feature Index", ytitle="Loss"):
        """
        Visualize the loss function for each sample
        
        Parameters:
        -loss: multidimensonal loss matrix. usually n*p
        -save_ind: index of sample you want to save, default None.
        -save_prefix: prefix if you want to save
        other parameters inherite from vis_rsq
        """
        def sample_loss(i):
            plt.cla()  # Clear the current axes
            if 0 <= i < loss.shape[0]:  # Check if i is within the valid range
                vis.rsq(loss[i], title=title + " " + str(i), color_map_name=color_map_name, model_rsq=model_rsq, decimal=decimal, xtitle=xtitle, ytitle=ytitle)
        
        # # Slider for quick navigation
        # i_slider = widgets.IntSlider(
        #     value=0,
        #     min=0,
        #     max=x.shape[1]-1,
        #     step=1,
        #     description='Slider:',
        #     continuous_update=False
        # )

        # Text input for precise entry
        i_text = widgets.BoundedIntText(
            value=0,
            min=0,
            max=loss.shape[0]-1,
            step=1,
            description='Sample Index',
            style={'description_width': 'initial'},  # This ensures the description is not cut off
            layout=Layout(width='150px'), 
            continuous_update=True
        )

        # Link the slider and text input to keep them in sync
        # widgets.jslink((i_slider, 'value'), (i_text, 'value'))

        # # Display both the slider and the text input
        # ui = widgets.VBox([i_slider, i_text])
        ui = widgets.VBox([i_text])

        # Use interactive_output to update the plot, linking both controls
        out = widgets.interactive_output(sample_loss, {'i': i_text})

        if save_ind is not None:
            save_name = save_prefix + " " + str(save_ind)
            vis.rsq(loss[save_ind], title=title + " " + str(save_ind), color_map_name=color_map_name, model_rsq=model_rsq, decimal=decimal, xtitle=xtitle, ytitle=ytitle, 
                save_name=save_name)
        else:
            display(ui, out)
        
    # vis.loss(loss)

    # # Find a lovely plot and save it, say for the 5-th sample
    # vis.loss(loss, save_ind=10)
            
    @staticmethod
    def elbow(x, xtitle="Feature Number", ytitle="Explained Variance",
                max_comp=10, title='Explained Variance by Top Features', marker='o', linestyle='--'):
        """
        Construct elbow plot for top features.
        
        Parameters:
        -x: Shapley R squared
        -max_comp: maximum number of components in the plot
        
        Return:
        An elbow plot and the top indices in x that have the highest variance explained
        """
        # Ensure max_comp is not greater than the length of x
        max_comp = int(min(max_comp, len(x)))
        
        # Get indices of the sorted variances in descending order
        indices_sorted_variances = np.argsort(x)[::-1]
        
        # Select the indices corresponding to the highest "max_comp" variances
        selected_indices = indices_sorted_variances[:max_comp]
        
        # Calculate cumulative explained variance for the selected components
        sorted_variances = x[selected_indices]
        #cumulative_explained_variance = np.cumsum(sorted_variances / np.sum(x))
        
        
        plt.plot(range(1, max_comp + 1), sorted_variances, marker=marker, linestyle=linestyle)
        plt.title(title)
        plt.xlabel(xtitle)
        plt.ylabel(ytitle)
        
        # Setting integer ticks on the x-axis
        plt.xticks(range(1, max_comp + 1), range(1, max_comp + 1))
        plt.show()
        
        plt.close()
        
        return selected_indices
    
        # Call the elbow_plot_indices function
         #indices_for_max_comp = vis_elbow(rsq_res, max_comp = 15)
         #indices_for_max_comp

    @staticmethod
    def cumu(x, xtitle="Feature Number", ytitle="Cumulative Explained Variance", title='Cumulative Explained Variance by Top Features',
             max_comp=10, save_name=None):
        """
        Construct cumulative explained variance plot for top features.

        Parameters:
        -x:  Shapley R squared

        Return:
        A cumulative explained variance plot
        """

        # Step 1: Calculate the total variance explained (R²) as the sum of explained variances
        r_squared = x.sum()  # Total R²

        max_comp = int(min(max_comp, len(x)))

        # Step 2: Sort variance explained in descending order
        sorted_indices = np.argsort(-x)  # Sort in descending order
        explained_variance_sorted = x[sorted_indices][:max_comp]

        # Step 3: Calculate cumulative variance
        cumulative_variance = np.cumsum(explained_variance_sorted)

        plt.axhline(y=r_squared, color='red', linestyle='--', label=f'Model R² = {r_squared:.2f}')
        plt.plot(
            np.arange(1, len(explained_variance_sorted) + 1),  # X-axis: Number of components
            cumulative_variance,                        # Y-axis: Cumulative variance
            marker='o',                                 # Marker style
            linestyle='-',                              # Line style
            label='Cumulative Variance'
        )

        # Add labels and title
        plt.xticks(ticks=np.arange(1, len(explained_variance_sorted) + 1))  # Ensure integer ticks for components
        plt.xlabel('Number of Top Features')  # X-axis label
        plt.ylabel('Cumulative R²')           # Y-axis label
        plt.title('Cumulative Variance Explained by Components')  # Title

        # Add legend
        plt.legend()

        if save_name is not None:
            name = save_name + ".pdf"
            plt.savefig(name, bbox_inches='tight')

        # Show plot (grid is disabled by default)
        plt.show()

        plt.close()
    
    @staticmethod
    def gcorr(x, color_map_name="Blues", horizontal=False, max_feature=10, cutoff=0, title="Generalized Correlation of Features to the Outcome", xtitle="Feature index", ytitle="Generalized Correlation", rotation=0, label=None, decimal=3, save_name=None):
        """
        Visualize Generalized Shapley correlation
        
        Parameters:
        -x: 1 dim array, but we recommend array from shapley rsq OvO ...
        -color_map_name: Color map allows to you take a variety color map from matplotlib to customize your visualization color.
        You can try to use "Pastel1", "Pasterl2", "PuBu", "Reds", "Greens" ...
        -horizontal: horizontal plot or not
        -max_feature: maximum number of features to show
        -cutoff: only value greater than or equal to the cutoff will be displayed
        -xtitle: xtitle, note that it's reversed for horizontal 
        -ytitle: ytitle, note that it's reversed for horizontal 
        -rotation: rotation of the tick
        -label: label for the features
        -title: plot title
        -decimal: decimals to show 
        -save_name: the name of the file if you want to save. None as default without saving.
        """
        vis.rsq(np.sqrt(x), color_map_name=color_map_name, horizontal=horizontal, model_rsq=False, max_feature=max_feature, cutoff=cutoff, title=title, xtitle=xtitle, ytitle=ytitle, rotation=0, label=None, decimal=3, save_name=None)



import subprocess

# Run the Streamlit app using subprocess
def run_streamlit_app():
    try:
        # Run the Streamlit app using the generated app.py
        subprocess.run(["streamlit", "run", "qshap/vis_llm.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
    except FileNotFoundError:
        print("Streamlit is not installed or app.py not found.")

# Run the Streamlit app
if __name__ == "__main__":
    run_streamlit_app()  