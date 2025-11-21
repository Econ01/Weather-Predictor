# This module is responsible for loading and formatting the raw data obtained from: https://www.ecad.eu/dailydata/index.php

import pandas as pd
import numpy as np

# Define file mapping for easy referencing
# (Element, File Path, Header Rows to Skip, Column Name)
FILE_MAPPINGS = [
    ("TG", "./data/TG_SOUID121044.txt", 20, "   TG"), # Mean Temperature
    ("TN", "./data/TN_SOUID121045.txt", 20, "   TN"), # Min Temperature
    ("TX", "./data/TX_SOUID121046.txt", 20, "   TX"), # Max Temperature
    ("RR", "./data/RR_SOUID121042.txt", 20, "   RR"), # Precipitation
    ("PP", "./data/PP_SOUID121041.txt", 20, "   PP"), # Sea Level Pressure
    ("SS", "./data/SS_SOUID121040.txt", 20, "   SS"), # Sunshine Duration
    ("HU", "./data/HU_SOUID121047.txt", 20, "   HU"), # Humidity
    ("FG", "./data/FG_SOUID121048.txt", 20, "   FG"), # Wind Speed
    ("FX", "./data/FX_SOUID121049.txt", 20, "   FX"), # Wind Gust
    ("CC", "./data/CC_SOUID121039.txt", 20, "   CC"), # Cloud Cover
    ("SD", "./data/SD_SOUID121043.txt", 20, "   SD"), # Snow Depth
    ("QQ", "./data/QQ_SOUID210447.txt", 20, "   QQ"), # Global Radiation
]

def ReadAndMerge():
    print("Reading and Formatting the data...")

    # Create empty list to hold each individual DataFrame
    all_dfs = []

    for element, file_path, skip, col_name in FILE_MAPPINGS:
        # Load the specific file
        df = pd.read_csv(file_path,
                         skiprows=skip,
                         usecols=["    DATE", col_name], # Only load DATE and DATA COLUMN
                         parse_dates= ["    DATE"], # Parse DATA values correctly
                         na_values=["-9999"] # Remove unusable data by converting them to NaN
                        )
        
        # Clean column names
        df.rename(columns={col_name: element, "    DATE": "DATE"}, inplace=True)

        # Set DATA as index to merge correctly
        df.set_index("DATE", inplace=True)

        all_dfs.append(df)

    # Set the DATA as the index for easy merging
    master_df = pd.concat(all_dfs, axis=1)

    print("Read and Merged")

    # Sort by date and return
    return master_df.sort_index()

def CleanData(master_df):
    # Fills and drops NaN values to create a clean and a complete dataset
    print("Cleaning Data (Fill/Drop NaN)...")

    # IMPORTANT: Remove data beyond last real observation date
    # 2025 data only goes up to Sept 30, 2024 (rest is padding/NaN)
    LAST_REAL_DATE = pd.Timestamp('2025-09-30')

    # Truncate at last real date
    master_df = master_df[master_df.index <= LAST_REAL_DATE]

    # Fill missing values using the value from the previous day
    # (only for gaps within real data, not future padding)
    master_df.ffill(inplace=True)

    # After ffill, there might be NaNs left at the very start (e.g., 1931-1957)
    # We simply drop those data
    master_df.dropna(inplace=True)

    print(f"Cleaning Complete (data up to {master_df.index.max()})")

    return master_df

def PlotHeatmap(df, show=True, save=True, plot_3d=True):
    # Calculates and plots a correlation heatmap for the raw weather variables

    print("Generating Correlation Heatmap...")
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from mpl_toolkits.mplot3d import Axes3D # noqa: F401 (This is required for 'projection=3d')
        from matplotlib import cm
    except ImportError:
        print("\n---")
        print("ERROR: `matplotlib` and `seaborn` are required for this step.")
        print("Please run: pip install matplotlib seaborn")
        print("---\n")
        return

    # Calculate the correlation matrix
    # Note: PP and QQ are excluded due to data quality issues
    corr_matrix = df[['TG', 'TN', 'TX', 'RR', 'SS', 'HU', 'FG', 'FX', 'CC', 'SD']].corr()

    # 2D Plot
    fig_2d, ax_2d = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr_matrix, 
        annot=True,       # Show the correlation numbers
        fmt='.2f',        # Format numbers to 2 decimal places
        cmap='coolwarm',  # Use a blue-to-red color map
        vmin=-1,          # Set min value of color bar
        vmax=1,           # Set max value of color bar
        ax=ax_2d          # Specify the axes
    )
    ax_2d.set_title('2D Correlation Heatmap of Weather Variables')
    fig_2d.tight_layout()

    if save:
        # Save the figure
        fig_2d.savefig('./figures/correlation_heatmap_2d.png', dpi=800)
        print("2D Correlation heatmap saved to './figures/correlation_heatmap_2d.png'")
    
    if show:
        plt.show()
    else:
        plt.close(fig_2d) # Close the 2D figure if not showing

    if plot_3d:
        print("Generating 3D Correlation Plot...")
        
        fig_3d = plt.figure(figsize=(14, 12))
        ax_3d = fig_3d.add_subplot(111, projection='3d')

        n_vars = len(corr_matrix.columns)
        
        # Create a meshgrid for x, y positions
        # (x_pos, y_pos) = (0,0), (0,1), (0,2)... (1,0), (1,1)...
        x_pos, y_pos = np.meshgrid(np.arange(n_vars), np.arange(n_vars))
        x_pos = x_pos.flatten()
        y_pos = y_pos.flatten()
        
        # z is the height (correlation value)
        z_values = corr_matrix.values.flatten()
        
        # Set the diagonal (where x==y) to 0 so it doesn't 
        # ruin the Z-axis scale.
        z_values[x_pos == y_pos] = 0
        
        # z_bottom is where the bars start (at the 0 plane)
        z_bottom = np.zeros_like(z_values)
        
        # dx, dy are the width/depth of bars
        dx = dy = 0.8 * np.ones_like(z_values)
        
        # Map values to 'coolwarm' colormap
        norm = plt.Normalize(vmin=-1, vmax=1)
        cmap = cm.coolwarm
        colors = cmap(norm(z_values))

        # Plot the 3D bars
        ax_3d.bar3d(x_pos, y_pos, z_bottom, dx, dy, z_values, color=colors, shade=True)
        
        # Set Ticks and Labels
        ticks = np.arange(n_vars) + 0.4 # Center ticks on 0.8 width bars
        ax_3d.set_xticks(ticks)
        ax_3d.set_yticks(ticks)
        ax_3d.set_xticklabels(corr_matrix.columns, rotation=45, ha='left')
        ax_3d.set_yticklabels(corr_matrix.columns, rotation=-45, ha='right')
        ax_3d.set_zlabel('Correlation')
        ax_3d.set_title('3D Correlation Plot (Diagonal Removed)')
        # Set Z-limit to -1 and 1
        ax_3d.set_zlim(-1, 1)

        # Add a color bar
        mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
        mappable.set_array(z_values)
        
        fig_3d.colorbar(mappable, ax=ax_3d, shrink=0.6, aspect=10, label='Correlation')

        if save:
            fig_3d.savefig('./figures/correlation_heatmap_3d.png', dpi=800)
            print("3D Correlation plot saved to './figures/correlation_heatmap_3d.png'")
        
        if show:
            plt.show()
        else:
            plt.close(fig_3d)

def PlotHistograms(df, show=True, save=True):
    # Generates and saves histograms for all available variables
    print("Generating histograms...")
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: `matplotlib` is required for this step.")
        return

    # Create a mapping of variable codes to descriptive names with units
    variable_names = {
        'TG': 'Mean Temperature (TG) [0.1°C]',
        'TN': 'Min Temperature (TN) [0.1°C]',
        'TX': 'Max Temperature (TX) [0.1°C]',
        'RR': 'Precipitation (RR) [0.1mm]',
        'PP': 'Sea Level Pressure (PP) [0.1hPa]',
        'SS': 'Sunshine Duration (SS) [0.1h]',
        'HU': 'Humidity (HU) [%]',
        'FG': 'Wind Speed (FG) [0.1m/s]',
        'FX': 'Wind Gust (FX) [0.1m/s]',
        'CC': 'Cloud Cover (CC) [oktas]',
        'SD': 'Snow Depth (SD) [cm]',
        'QQ': 'Global Radiation (QQ) [W/m²]',
        'IS_RAIN': 'Precipitation Indicator (IS_RAIN) [0/1]',
        'DAY_SIN': 'Day of Year Sin (DAY_SIN) [-1 to 1]',
        'DAY_COS': 'Day of Year Cos (DAY_COS) [-1 to 1]'
    }

    # Get all numeric columns from the dataframe
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        print("  Skipping histograms: No numeric variables found.")
        return

    # Calculate number of variables to plot
    n_vars = len(numeric_df.columns)

    # Calculate appropriate grid layout (aim for roughly square layout)
    n_cols = int(np.ceil(np.sqrt(n_vars)))
    n_rows = int(np.ceil(n_vars / n_cols))

    # Adjust figure size based on number of subplots
    fig_width = max(15, n_cols * 4)
    fig_height = max(10, n_rows * 3)

    # Create a temporary dataframe with renamed columns for better labels
    df_renamed = numeric_df.rename(columns=variable_names)

    # Configure the figure
    axes = df_renamed.hist(bins=50, figsize=(fig_width, fig_height), layout=(n_rows, n_cols))
    fig = axes.flat[0].get_figure()

    fig.suptitle(f'Histograms (Distributions) of All Variables ({n_vars} variables)', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    if save:
        fig.savefig('./figures/variable_histograms.png', dpi=800)
        print(f"Histograms saved to './figures/variable_histograms.png' ({n_vars} variables)")

    if show:
        plt.show()
    else:
        plt.close(fig) # Close the figure if not showing


def PlotTimeSeries(df, show=True, save=True):
    # Generates and saves time series plots for all available variables
    print("Generating time series plots...")
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: `matplotlib` is required for this step.")
        return

    # Create a mapping of variable codes to descriptive names with units
    variable_names = {
        'TG': 'Mean Temperature (TG) [0.1°C]',
        'TN': 'Min Temperature (TN) [0.1°C]',
        'TX': 'Max Temperature (TX) [0.1°C]',
        'RR': 'Precipitation (RR) [0.1mm]',
        'PP': 'Sea Level Pressure (PP) [0.1hPa]',
        'SS': 'Sunshine Duration (SS) [0.1h]',
        'HU': 'Humidity (HU) [%]',
        'FG': 'Wind Speed (FG) [0.1m/s]',
        'FX': 'Wind Gust (FX) [0.1m/s]',
        'CC': 'Cloud Cover (CC) [oktas]',
        'SD': 'Snow Depth (SD) [cm]',
        'QQ': 'Global Radiation (QQ) [W/m²]',
        'IS_RAIN': 'Precipitation Indicator (IS_RAIN) [0/1]',
        'DAY_SIN': 'Day of Year Sin (DAY_SIN) [-1 to 1]',
        'DAY_COS': 'Day of Year Cos (DAY_COS) [-1 to 1]'
    }

    # Get all numeric columns from the dataframe
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        print("  Skipping time series plots: No numeric variables found.")
        return

    # Calculate number of variables to plot
    n_vars = len(numeric_df.columns)

    # Calculate appropriate grid layout (aim for roughly square layout)
    n_cols = int(np.ceil(np.sqrt(n_vars)))
    n_rows = int(np.ceil(n_vars / n_cols))

    # Adjust figure size based on number of subplots
    fig_width = max(18, n_cols * 5)
    fig_height = max(12, n_rows * 3)

    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

    # Flatten axes array for easier iteration
    if n_rows * n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Plot each variable
    for idx, col in enumerate(numeric_df.columns):
        ax = axes[idx]
        ax.plot(numeric_df.index, numeric_df[col], linewidth=0.5, alpha=0.7)

        # Use descriptive name with units if available
        title = variable_names.get(col, col)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Date', fontsize=8)
        ax.set_ylabel('Value', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.grid(True, alpha=0.3)

        # Rotate x-axis labels for better readability
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Hide unused subplots
    for idx in range(n_vars, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f'Time Series of All Variables ({n_vars} variables)', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    if save:
        fig.savefig('./figures/time_series_all_variables.png', dpi=800)
        print(f"Time series plot saved to './figures/time_series_all_variables.png' ({n_vars} variables)")

    if show:
        plt.show()
    else:
        plt.close(fig)


def PlotSeasonalBoxplot(df, show=True, save=True):
    # Generates and saves a seasonal boxplot for Temperature
    print("Generating seasonal boxplot...")
    if 'TG' not in df.columns:
        print("  Skipping seasonal boxplot: 'TG' column not found.")
        return
        
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: `matplotlib` is required for this step.")
        return

    # Create a copy to safely add a 'month' column for plotting
    plot_df = df.copy()
    plot_df['month'] = plot_df.index.month

    fig, ax = plt.subplots(figsize=(12, 7))
    # Use seaborn if available, otherwise fallback to pandas
    try:
        import seaborn as sns
        sns.boxplot(data=plot_df, x='month', y='TG', ax=ax)
    except ImportError:
        print("  (Install seaborn for a nicer-looking boxplot)")
        plot_df.boxplot(column='TG', by='month', ax=ax)
        ax.set_xlabel('Month')
        ax.set_ylabel('Mean Temperature (TG)')

    ax.set_title('Mean Temperature (TG) by Month (Seasonality)')
    fig.tight_layout()

    if save:
        fig.savefig('./figures/seasonal_boxplot_tg.png', dpi=800)
        print("Seasonal boxplot saved to './figures/seasonal_boxplot_tg.png'")
    
    if show:
        plt.show()
    else:
        plt.close(fig) # Close the figure if not showing


def PlotLongTermTrend(df, show=True, save=True):
    # Generates and saves a long-term (annual) trend plot for Temperature
    print("Generating long-term trend plot...")
    if 'TG' not in df.columns:
        print("  Skipping long-term trend: 'TG' column not found.")
        return
        
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: `matplotlib` is required for this step.")
        return

    # Resample daily data to annual averages
    annual_avg_temp = df['TG'].resample('YE').mean()
    
    fig, ax = plt.subplots(figsize=(12, 7))

    # Get the x-axis as numeric years
    x_values = annual_avg_temp.index.year

    # Get the y-axis as the temperature values
    y_values = annual_avg_temp.values
    
    # Plot the blue line using numeric years (x) and temp values (y)
    ax.plot(x_values, y_values, label='Annual Average Temp', marker='.', linestyle='-')
    
    # Add a rolling average and trendline
    try:
        import seaborn as sns
        
        # We need to re-create a Series with the numeric year index to roll properly
        rolling_series = pd.Series(y_values, index=x_values)
        rolling_avg = rolling_series.rolling(window=5).mean()
        
        sns.regplot(
            x=x_values, 
            y=y_values, 
            ci=None, 
            order=1, 
            line_kws={'color': 'red', 'linestyle': '--'}, 
            scatter_kws={'alpha': 0.5, 's': 15}, # Made scatter points smaller
            label='Linear Trend',
            ax=ax
        )
        # Plot the rolling average
        rolling_avg.plot(label='5-Year Rolling Avg', color='orange', linestyle='--', ax=ax)
    except ImportError:
        print("  (Install seaborn for a nice trendline on the annual plot)")
    
    ax.set_title('Long-Term Annual Average Temperature (TG) Trend')
    ax.set_ylabel('Annual Avg. Temp (0.1 C)')
    ax.set_xlabel('Year')
    ax.legend()
    fig.tight_layout()

    if save:
        fig.savefig('./figures/long_term_trend_tg.png', dpi=800)
        print("Long-term trend plot saved to './figures/long_term_trend_tg.png'")

    if show:
        plt.show()
    else:
        plt.close(fig) # Close the figure if not showing

def EngineerFeatures(master_df):
    # Creates the IS_RAIN target for presipication classification
    # Adds cyclical date features for climate tracking
    print("Engineering Features...")

    # Create a new binary target column `IS_RAIN`
    # 1 if precipitation was > 0.1mm (RR > 1), 0 otherwise
    master_df['IS_RAIN'] = (master_df['RR'] > 1).astype(int)

    # Create "Climate Trend" Features
    day_of_year = master_df.index.dayofyear
    master_df['DAY_SIN'] = np.sin(2 * np.pi * day_of_year / 365.25)
    master_df['DAY_COS'] = np.cos(2 * np.pi * day_of_year / 365.25)

    # Define final feature and target columns
    # EXCLUDE PP and QQ (contaminated data - see data quality analysis)
    # EXCLUDE TG from features
    FEATURE_COLS = [
        'TN', 'TX', 'RR', 'SS', 'HU',
        'FG', 'FX', 'CC', 'SD'
    ]
    TARGET_COLS = ['TG']

    # Get lists of columns that actually exist in the dataframe in case some files failed to load
    final_features = [col for col in FEATURE_COLS if col in master_df.columns]
    final_targets = [col for col in TARGET_COLS if col in master_df.columns]

    # Combine features and targets, removing duplicates while preserving order
    # Use dict.fromkeys() to remove duplicates while maintaining order
    all_columns = list(dict.fromkeys(final_features + final_targets))

    clean_df = master_df[all_columns]

    print(f"Complete - Total features: {len(final_features)}")
    return clean_df, final_features, final_targets

def load_and_process_data(run_plots=True, save_plots=True, show_plots=False, save_csv=False):
    # This function controls the behaviour of the dataMAnager module

    # Read and Merge the data
    master_df = ReadAndMerge()

    # Clean the data
    master_df_cleaned = CleanData(master_df.copy())

    # Feature Engineering
    clean_feature_df, features, targets = EngineerFeatures(master_df_cleaned)

    if run_plots:
        # Run all plots on the final engineered data
        PlotHeatmap(clean_feature_df, show=show_plots, save=save_plots, plot_3d=True)
        PlotHistograms(clean_feature_df, show=show_plots, save=save_plots)
        PlotTimeSeries(clean_feature_df, show=show_plots, save=save_plots)
        PlotLongTermTrend(clean_feature_df, show=show_plots, save=save_plots)
        PlotSeasonalBoxplot(clean_feature_df, show=show_plots, save=save_plots)

    if save_csv:
        clean_feature_df.to_csv('./modifiedData/clean_data.csv')
        print("\nSuccessfully saved final processed data to 'clean_data.csv'")

    print(f"\nTotal Features: {len(features)}")
    print(f"Total Targets: {len(targets)}")

    print("\nData processing complete!")

    return clean_feature_df, features, targets


if __name__ == "__main__":
    load_and_process_data(
        run_plots=True, 
        save_plots=True, 
        show_plots=True,
        save_csv=True
    )