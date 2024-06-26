import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import BoundedIntText, Layout, interactive_output, VBox, HBox
from IPython.display import display, clear_output


def vis_rsq(x, color_map_name="Blues", horizontal=False, model_rsq=True, max_feature=10, title="Shapley R²", xtitle="Feature index", ytitle="R²", decimal=3, save_name=None):
    """
    Visualize Shapley R²
    
    Parameters:
    -x: 1 dim array, but we recommend array from Shapley rsq OvO ...
    -color_map_name: Color map allows to you take a variety color map from matplotlib to customize your visualization color.
    You can try to use "Pastel1", "Pasterl2", "PuBu", "Reds", "Greens" ...
    -horizontal: horizontal plot or not
    -model_rsq: display model rsq or not
    -max_feature: maximum number of features to show
    -xtitle: xtitle, note that it's reversed for horizontal 
    -ytitle: ytitle, note that it's reversed for horizontal 
    -title: plot title
    -decimal: decimals to show 
    -save_name: the name of the file if you want to save. None as default without saving.
    """
    
    # Calculate the sum of x
    x_sum = np.sum(x)
    x_len = np.size(x)
    show_len = min(x_len, max_feature)

    # Sort x and keep track of the indices
    indices = np.argsort(x)[::-1]  # Indices of sorted elements (in descending order)
        
    sorted_x = x[indices]          # Sorted x
    indices = indices[:show_len]
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
        plt.xticks(range(len(sorted_x)), indices)
         
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
        plt.yticks(range(len(sorted_x)), indices)
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
    
# vis_rsq(rsq_res)
# # Change color
# vis_rsq(rsq_res, color_map_name="Pastel2")
# # Give a horizontal plot, hide model rsq, change the number of features to show
# vis_rsq(rsq_res, color_map_name="PuBu", horizontal=True, model_rsq=False, max_feature=15, save_name="rsq_eg")
    
def vis_loss(loss, save_ind=None, save_prefix="Treeshap loss sample", title="Treeshap Loss", color_map_name="Blues", model_rsq=False, decimal=0, xtitle="Feature Index", ytitle="Loss"):
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
            vis_rsq(loss[i], title=title, color_map_name=color_map_name, model_rsq=model_rsq, decimal=decimal, xtitle=xtitle, ytitle=ytitle)
    
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
        vis_rsq(loss[save_ind], title=title, color_map_name=color_map_name, model_rsq=model_rsq, decimal=decimal, xtitle=xtitle, ytitle=ytitle, 
            save_name=save_name)
    else:
        display(ui, out)
    
# vis_loss(loss)

# # Find a lovely plot and save it, say for the 5-th sample
# vis_loss(loss, save_ind=10)
