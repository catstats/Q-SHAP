import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import PercentFormatter
from ipywidgets import BoundedIntText, Layout, interactive_output, VBox, HBox
from IPython.display import display, clear_output

class vis:
    """Static plotting helpers for global and observation-level Q-SHAP output."""

    @staticmethod
    def rsq(x, color_map_name="Blues", horizontal=False, model_rsq=True, max_feature=10, cutoff=0, title="Shapley R²", xtitle="Feature index", ytitle="R²", rotation=0, label=None, decimal=3, save_name=None):
        """Plot ranked feature R-squared contributions as a bar chart.
        
        Parameters
        ----------
        x : numpy.ndarray of shape (n_features,)
            Global feature contribution vector.
        color_map_name : str, optional
            Matplotlib color map name. Default: ``'Blues'``.
        horizontal : bool, optional
            Draw horizontal rather than vertical bars. Default: ``False``.
        model_rsq : bool, optional
            Annotate the sum of the displayed input vector as model R-squared. Default: ``True``.
        max_feature : int, optional
            Maximum number of ranked features to display. Default: ``10``.
        cutoff : float, optional
            Include values greater than or equal to this cutoff. Default: ``0``.
        title : str or None, optional
            Plot title; heatmap None selects a quantity-specific title. Default: ``'Shapley R²'``.
        xtitle : str, optional
            Horizontal axis label (bar plots swap axes when horizontal). Default: ``'Feature index'``.
        ytitle : str, optional
            Vertical axis label (bar plots swap axes when horizontal). Default: ``'R²'``.
        rotation : float, optional
            Feature tick-label rotation in degrees. Default: ``0``.
        label : numpy.ndarray or None, optional
            Feature labels in original input order, one per feature. Default: ``None``.
        decimal : int, optional
            Number of decimal places for bar annotations. Default: ``3``.
        save_name : str or None, optional
            Output file prefix; .pdf is appended. None skips saving. Default: ``None``.

        Returns
        -------
        None
            Displays the chart and closes it; optionally writes a PDF.

        Notes
        -----
        The cutoff and feature limit must leave at least one value to display.
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
        """Inspect raw loss contributions for one observation at a time.
        
        Parameters
        ----------
        loss : numpy.ndarray of shape (n_samples, n_features)
            Raw observation-level loss contributions.
        save_ind : int or None, optional
            Zero-based observation to save. None displays a notebook row selector. Default: ``None``.
        save_prefix : str, optional
            Saved PDF prefix, followed by the selected observation index. Default: ``'Shapley loss sample'``.
        title : str or None, optional
            Plot title; heatmap None selects a quantity-specific title. Default: ``'Shapley Loss: Sample'``.
        color_map_name : str, optional
            Matplotlib color map name. Default: ``'Blues'``.
        model_rsq : bool, optional
            Annotate the sum of the displayed input vector as model R-squared. Default: ``False``.
        decimal : int, optional
            Number of decimal places for bar annotations. Default: ``0``.
        xtitle : str, optional
            Horizontal axis label (bar plots swap axes when horizontal). Default: ``'Feature Index'``.
        ytitle : str, optional
            Vertical axis label (bar plots swap axes when horizontal). Default: ``'Loss'``.

        Returns
        -------
        None
            Displays an ipywidgets selector in notebooks or saves a selected row as PDF.
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
    def heatmap(
        result,
        quantity=None,
        feature_names=None,
        observation_ids=None,
        samples=None,
        n_show=40,
        global_importance=None,
        title=None,
        xtitle="Feature",
        ytitle="Observations",
        legend_title=None,
        low_color="#2166AC",
        mid_color="white",
        high_color="#B2182B",
        rotation=45,
        save_name=None,
        show=True,
    ):
        """Plot observation-level contributions with a signed row-total column.

        Parameters
        ----------
        result : local result, dict, or numpy.ndarray
            Result of gazer.rsq(local=True), its fields as a dict, or an n-by-p matrix.
        quantity : {"local_rsq", "loss"} or None, optional
            Matrix field to show. None prefers local_rsq when present, otherwise loss. Default: ``None``.
        feature_names : array-like or None, optional
            Labels matching matrix columns. None generates Feature 0, Feature 1, etc. Default: ``None``.
        observation_ids : array-like or None, optional
            Identifiers matching rows. None uses string versions of zero-based indices. Default: ``None``.
        samples : array-like or None, optional
            Explicit zero-based integer row indices or string observation IDs. None selects extremes of row totals. Default: ``None``.
        n_show : int, optional
            Maximum rows selected automatically, split between high and low totals. Default: ``40``.
        global_importance : array-like or None, optional
            Column ordering scores, sorted descending. Defaults to result.rsq or matrix column sums. Default: ``None``.
        title : str or None, optional
            Plot title; heatmap None selects a quantity-specific title. Default: ``None``.
        xtitle : str, optional
            Horizontal axis label (bar plots swap axes when horizontal). Default: ``'Feature'``.
        ytitle : str, optional
            Vertical axis label (bar plots swap axes when horizontal). Default: ``'Observations'``.
        legend_title : str or None, optional
            Color bar title. None selects a quantity-specific label. Default: ``None``.
        low_color : color, optional
            Matplotlib color for negative values. Default: ``'#2166AC'``.
        mid_color : color, optional
            Matplotlib color at zero. Default: ``'white'``.
        high_color : color, optional
            Matplotlib color for positive values. Default: ``'#B2182B'``.
        rotation : float, optional
            Feature tick-label rotation in degrees. Default: ``45``.
        save_name : str or None, optional
            Output file prefix; .pdf is appended. None skips saving. Default: ``None``.
        show : bool, optional
            Call pyplot.show() when True; False supports scripts and caller-managed display. Default: ``True``.

        Returns
        -------
        matplotlib.figure.Figure
            Heatmap figure; the caller may save or close it.

        Raises
        ------
        ValueError
            If matrix dimensions, labels, quantity, selection, or ordering scores are invalid.

        Notes
        -----
        Columns follow descending global importance. Automatic row selection uses
        both extremes of signed row totals. Values decompose global R-squared;
        they are not separate per-observation R-squared values.
        ``loss_heatmap`` is an alias of this method.
        """

        def get_field(obj, name):
            if isinstance(obj, dict):
                return obj.get(name)
            return getattr(obj, name, None)

        local_rsq = get_field(result, "local_rsq")
        raw_loss = get_field(result, "loss")
        is_result = local_rsq is not None or raw_loss is not None

        if quantity is None:
            quantity = "local_rsq" if local_rsq is not None else "loss"
        if quantity not in {"local_rsq", "loss"}:
            raise ValueError("quantity must be 'local_rsq' or 'loss'")

        if is_result:
            values = get_field(result, quantity)
            if values is None:
                raise ValueError(f"The result does not contain {quantity!r}")
            if global_importance is None:
                global_importance = get_field(result, "rsq")
        else:
            values = result

        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("The heatmap values must be a two-dimensional matrix")
        n_observations, n_features = values.shape
        if n_observations < 1 or n_features < 1:
            raise ValueError("The heatmap requires at least one observation and one feature")

        if feature_names is None:
            feature_names = np.asarray(
                [f"Feature {index}" for index in range(n_features)],
                dtype=object,
            )
        else:
            feature_names = np.asarray(feature_names, dtype=object)
        if feature_names.ndim != 1 or len(feature_names) != n_features:
            raise ValueError("feature_names must match the number of matrix columns")

        if observation_ids is None:
            observation_ids = np.asarray(
                [str(index) for index in range(n_observations)],
                dtype=object,
            )
        else:
            observation_ids = np.asarray(observation_ids, dtype=object)
        if observation_ids.ndim != 1 or len(observation_ids) != n_observations:
            raise ValueError("observation_ids must match the number of matrix rows")
        observation_labels = np.asarray(
            [str(identifier) for identifier in observation_ids],
            dtype=object,
        )

        if global_importance is None:
            column_sums = np.nansum(values, axis=0)
            global_importance = (
                column_sums if quantity == "local_rsq" else -column_sums
            )
        global_importance = np.asarray(global_importance, dtype=np.float64)
        if global_importance.ndim != 1 or len(global_importance) != n_features:
            raise ValueError(
                "global_importance must match the number of matrix columns"
            )
        feature_order = np.argsort(-global_importance, kind="stable")

        observation_total = np.nansum(values, axis=1)
        if samples is None:
            if isinstance(n_show, (bool, np.bool_)) or not isinstance(
                n_show, (int, np.integer)
            ) or n_show < 1:
                raise ValueError("n_show must be a positive integer")
            n_select = min(int(n_show), n_observations)
            n_high = (n_select + 1) // 2
            n_low = n_select // 2
            high_order = np.argsort(-observation_total, kind="stable")
            low_order = np.argsort(observation_total, kind="stable")

            selected = list(high_order[:n_high])
            selected_set = set(selected)
            selected.extend(
                index for index in low_order if index not in selected_set
            )
            selected = selected[:n_select]

            if len(selected) < n_select:
                selected_set = set(selected)
                absolute_order = np.argsort(
                    -np.abs(observation_total), kind="stable"
                )
                selected.extend(
                    index
                    for index in absolute_order
                    if index not in selected_set
                )
                selected = selected[:n_select]

            selected = np.asarray(selected, dtype=np.int64)
            selected = selected[
                np.argsort(-observation_total[selected], kind="stable")
            ]
        else:
            requested = np.atleast_1d(samples)
            if requested.size < 1:
                raise ValueError("samples must contain at least one observation")
            if np.issubdtype(requested.dtype, np.integer):
                selected = requested.astype(np.int64, copy=False)
                if np.any(selected < 0) or np.any(selected >= n_observations):
                    raise ValueError("samples contains an invalid observation index")
            else:
                id_to_index = {}
                for index, identifier in enumerate(observation_labels):
                    if identifier in id_to_index:
                        raise ValueError(
                            "observation_ids must be unique when samples uses IDs"
                        )
                    id_to_index[identifier] = index
                try:
                    selected = np.asarray(
                        [id_to_index[str(identifier)] for identifier in requested],
                        dtype=np.int64,
                    )
                except KeyError as error:
                    raise ValueError(
                        f"Unknown observation ID: {error.args[0]}"
                    ) from None

            selected = np.asarray(
                list(dict.fromkeys(selected.tolist())), dtype=np.int64
            )

        selected_values = values[selected][:, feature_order]
        selected_totals = observation_total[selected]
        selected_labels = observation_labels[selected]
        ordered_feature_names = feature_names[feature_order]

        finite_values = np.concatenate(
            (
                selected_values[np.isfinite(selected_values)],
                selected_totals[np.isfinite(selected_totals)],
            )
        )
        color_limit = (
            float(np.max(np.abs(finite_values))) if finite_values.size else 1.0
        )
        if not np.isfinite(color_limit) or color_limit == 0:
            color_limit = 1.0

        cmap = LinearSegmentedColormap.from_list(
            "qshap_diverging", [low_color, mid_color, high_color]
        )
        norm = TwoSlopeNorm(
            vmin=-color_limit,
            vcenter=0.0,
            vmax=color_limit,
        )
        figure_width = max(7.0, min(14.0, 3.5 + 0.55 * n_features))
        figure_height = max(4.5, min(12.0, 2.6 + 0.22 * len(selected)))
        figure = plt.figure(
            figsize=(figure_width, figure_height), constrained_layout=True
        )
        grid = figure.add_gridspec(
            1,
            2,
            width_ratios=[max(n_features, 1), 0.75],
            wspace=0.06,
        )
        heatmap_axis = figure.add_subplot(grid[0, 0])
        total_axis = figure.add_subplot(grid[0, 1], sharey=heatmap_axis)

        image = heatmap_axis.imshow(
            selected_values,
            aspect="auto",
            interpolation="nearest",
            origin="upper",
            cmap=cmap,
            norm=norm,
        )
        total_axis.imshow(
            selected_totals[:, np.newaxis],
            aspect="auto",
            interpolation="nearest",
            origin="upper",
            cmap=cmap,
            norm=norm,
        )

        heatmap_axis.set_xticks(np.arange(n_features))
        heatmap_axis.set_xticklabels(
            ordered_feature_names,
            rotation=rotation,
            ha="right" if rotation else "center",
        )
        heatmap_axis.set_yticks(np.arange(len(selected)))
        heatmap_axis.set_yticklabels(selected_labels)
        heatmap_axis.set_xlabel(xtitle)
        heatmap_axis.set_ylabel(ytitle)
        heatmap_axis.tick_params(axis="y", length=0)

        total_axis.set_xticks([0])
        total_axis.set_xticklabels(["Total"], rotation=rotation)
        total_axis.tick_params(axis="y", left=False, labelleft=False)
        total_axis.spines["left"].set_color("0.55")
        total_axis.spines["left"].set_linewidth(1.0)

        for axis in (heatmap_axis, total_axis):
            axis.grid(False)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)

        if title is None:
            title = (
                "Observation-level contributions to the global R² decomposition"
                if quantity == "local_rsq"
                else "Observation-level loss contributions"
            )
        if legend_title is None:
            legend_title = (
                "Local R² contribution"
                if quantity == "local_rsq"
                else "Loss contribution"
            )
        figure.suptitle(title, fontweight="bold")
        colorbar = figure.colorbar(
            image,
            ax=[heatmap_axis, total_axis],
            fraction=0.035,
            pad=0.03,
        )
        colorbar.set_label(legend_title)
        if quantity == "local_rsq":
            colorbar.ax.yaxis.set_major_formatter(
                PercentFormatter(xmax=1.0, decimals=2)
            )

        if save_name is not None:
            figure.savefig(f"{save_name}.pdf", bbox_inches="tight")
        if show:
            plt.show()

        return figure

    loss_heatmap = heatmap
            
    @staticmethod
    def elbow(x, xtitle="Feature Number", ytitle="Explained Variance",
                max_comp=10, title='Explained Variance by Top Features', marker='o', linestyle='--'):
        """Plot the contributions of the highest-ranked features.
        
        Parameters
        ----------
        x : numpy.ndarray of shape (n_features,)
            Global feature contribution vector.
        xtitle : str, optional
            Horizontal axis label (bar plots swap axes when horizontal). Default: ``'Feature Number'``.
        ytitle : str, optional
            Vertical axis label (bar plots swap axes when horizontal). Default: ``'Explained Variance'``.
        max_comp : int, optional
            Maximum number of ranked features in the curve. Default: ``10``.
        title : str or None, optional
            Plot title; heatmap None selects a quantity-specific title. Default: ``'Explained Variance by Top Features'``.
        marker : str, optional
            Matplotlib marker style for the elbow curve. Default: ``'o'``.
        linestyle : str, optional
            Matplotlib line style for the elbow curve. Default: ``'--'``.
        
        Returns
        -------
        numpy.ndarray of integers
            Zero-based indices of the displayed features, in descending contribution order.
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
        """Plot cumulative contributions of the highest-ranked features.

        Parameters
        ----------
        x : numpy.ndarray of shape (n_features,)
            Global feature contribution vector.
        xtitle : str, optional
            Horizontal axis label (bar plots swap axes when horizontal). Default: ``'Feature Number'``.
        ytitle : str, optional
            Vertical axis label (bar plots swap axes when horizontal). Default: ``'Cumulative Explained Variance'``.
        title : str or None, optional
            Plot title; heatmap None selects a quantity-specific title. Default: ``'Cumulative Explained Variance by Top Features'``.
        max_comp : int, optional
            Maximum number of ranked features in the curve. Default: ``10``.
        save_name : str or None, optional
            Output file prefix; .pdf is appended. None skips saving. Default: ``None``.

        Returns
        -------
        None
            Displays and closes the cumulative chart; optionally writes a PDF.
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
        plt.xlabel(xtitle)  # X-axis label
        plt.ylabel(ytitle)           # Y-axis label
        plt.title(title)  # Title

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
        """Plot square roots of global feature R-squared contributions.
        
        Parameters
        ----------
        x : numpy.ndarray of shape (n_features,)
            Global feature contribution vector.
        color_map_name : str, optional
            Matplotlib color map name. Default: ``'Blues'``.
        horizontal : bool, optional
            Draw horizontal rather than vertical bars. Default: ``False``.
        max_feature : int, optional
            Maximum number of ranked features to display. Default: ``10``.
        cutoff : float, optional
            Include values greater than or equal to this cutoff. Default: ``0``.
        title : str or None, optional
            Plot title; heatmap None selects a quantity-specific title. Default: ``'Generalized Correlation of Features to the Outcome'``.
        xtitle : str, optional
            Horizontal axis label (bar plots swap axes when horizontal). Default: ``'Feature index'``.
        ytitle : str, optional
            Vertical axis label (bar plots swap axes when horizontal). Default: ``'Generalized Correlation'``.
        rotation : float, optional
            Feature tick-label rotation in degrees. Default: ``0``.
        label : numpy.ndarray or None, optional
            Feature labels in original input order, one per feature. Default: ``None``.
        decimal : int, optional
            Number of decimal places for bar annotations. Default: ``3``.
        save_name : str or None, optional
            Output file prefix; .pdf is appended. None skips saving. Default: ``None``.

        Returns
        -------
        None
            Displays and closes the chart; optionally writes a PDF. Negative input values produce NaN.

        Notes
        -----
        The cutoff and feature limit must leave at least one value to display.
        """
        vis.rsq(
            np.sqrt(x),
            color_map_name=color_map_name,
            horizontal=horizontal,
            model_rsq=False,
            max_feature=max_feature,
            cutoff=cutoff,
            title=title,
            xtitle=xtitle,
            ytitle=ytitle,
            rotation=rotation,
            label=label,
            decimal=decimal,
            save_name=save_name,
        )



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
