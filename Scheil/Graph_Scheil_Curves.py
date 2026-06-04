# %% Functions

'''This is still under development. --Trevor H.'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pandas as pd

def change_directory_to_current_py_file() -> None:
    import os
    try:
        os.chdir(os.path.split(os.path.abspath(__file__))[0])
    except Exception as e:
        print(f'\nDirectory change or os module error:\n\n{e}')
    return None

def import_data(file_name: str) -> pd.DataFrame:
    import os
    csv_file = f"{file_name}.csv"
    if os.path.exists(csv_file):
        data = pd.read_csv(csv_file)
        return data
    
    with pd.ExcelFile(f"{file_name}.xlsx") as xlsx:
        sheet_names = xlsx.sheet_names
        data = pd.read_excel(xlsx, sheet_name=sheet_names[0])
    data = data.filter(regex='^(?!Unnamed|Index)')
    # data.to_csv(csv_file, index=False)  # Save as CSV for future use
    return data

def extract_elements(data: pd.DataFrame) -> list[str]:
    elements = [col for col in data.columns if len(col) <= 2]
    return elements

def pareto_maximum(
        data_points: np.ndarray
        ) -> np.ndarray:
    
    data_points = data_points[data_points.sum(1).argsort()[::-1]]
    is_not_dominated = np.ones(data_points.shape[0], dtype = bool)
    for i in range(data_points.shape[0]):
        n = data_points.shape[0]
        if i >= n:
            break
        is_not_dominated[i+1:n] = (data_points[i+1:] > data_points[i]).any(1)
        data_points = data_points[is_not_dominated[:n]]
        is_not_dominated = np.array([True] * len(data_points))
        
    return data_points

def calculate_graph_properties(
        data: pd.DataFrame,
        outputs: list[str],
        ) -> dict:
    
    # data_buffer = 0.05
    data_x_min: float = min(data[outputs[0]]) 
    data_x_max: float = max(data[outputs[0]]) 
    data_y_min: float = min(data[outputs[1]]) 
    data_y_max: float = max(data[outputs[1]]) 
    data_x_range: float = data_x_max - data_x_min
    data_y_range: float = data_y_max - data_y_min
    # plot_x_min: float = data_x_min - data_buffer * data_x_range
    plot_x_min: float = -0.05
    # plot_x_max: float = data_x_max + data_buffer * data_x_range
    plot_x_max: float = 1.05
    plot_y_min: float = int(data_y_min / 100 + 0.5) * 100 - 100
    plot_y_max: float = int(data_y_max / 100 + 0.5) * 100 + 100
    plot_x_range: float = plot_x_max - plot_x_min
    plot_y_range: float = plot_y_max - plot_y_min
    plot_x_tick: float = 10 ** np.floor(np.log10(plot_x_range * 0.5))
    plot_y_tick: float = 10 ** np.floor(np.log10(plot_y_range * 0.4))
    # print(10 ** np.floor(np.log10(plot_y_range * 0.2)))
    # print(plot_x_tick, plot_y_tick)
    
    graph_properties = {
        "data_x_min": data_x_min,
        "data_x_max": data_x_max,
        "data_y_min": data_y_min,
        "data_y_max": data_y_max,
        "data_x_range": data_x_range,
        "data_y_range": data_y_range,
        "plot_x_min": plot_x_min,
        "plot_x_max": plot_x_max,
        "plot_y_min": plot_y_min,
        "plot_y_max": plot_y_max,
        "plot_x_range": plot_x_range,
        "plot_y_range": plot_y_range,
        "plot_x_tick": plot_x_tick,
        "plot_y_tick": plot_y_tick,
    }

    return graph_properties

def set_plot_parameters(
        ax: plt.Axes,
        outputs: list[str],
        graph_properties: dict,
        ) -> None:
    
    ax.set_xlabel(outputs[0], fontsize=20)
    ax.set_ylabel(outputs[1], fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    # x_ticks = np.arange(
    #     np.floor(graph_properties["plot_x_min"]), 
    #     np.ceil(graph_properties["plot_x_max"]) + graph_properties["plot_x_tick"], 
    #     graph_properties["plot_x_tick"])
    x_ticks = np.arange(
        0, 
        np.ceil(graph_properties["plot_x_max"]) + graph_properties["plot_x_tick"], 
        graph_properties["plot_x_tick"])
    y_ticks = np.arange(
        np.floor(graph_properties["plot_y_min"]), 
        np.ceil(graph_properties["plot_y_max"]) + graph_properties["plot_y_tick"], 
        graph_properties["plot_y_tick"])
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{i:.2g}" for i in x_ticks])
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{i:.0f}" for i in y_ticks])
    # ax.set_yticklabels([f"{i:.1f}" for i in y_ticks])
    ax.set_xlim(graph_properties["plot_x_min"], graph_properties["plot_x_max"])
    ax.set_ylim(graph_properties["plot_y_min"], graph_properties["plot_y_max"])
    
    return None

def add_color_bar(
        ax: plt.Axes,
        dataset: pd.DataFrame,
        column: str,
        cmap: str,
        font_size: int,
        label: str,
        increment: float,
        ) -> None:
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    
    vmin, vmax = dataset[column].min(), dataset[column].max()
    print(vmin, vmax)
    norm = mcolors.Normalize(vmin = vmin, vmax = vmax)
    sm = cm.ScalarMappable(cmap = cmap, norm = norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax = ax)
    cbar.set_label(label, fontsize = font_size)
    cbar.ax.tick_params(labelsize = font_size)
    cbar.set_ticks(np.arange(round(vmin,2),round(vmax,2) + 0.01, increment))
    
    return None


# %% Import data

change_directory_to_current_py_file()
file_name: str = 'Graph_Scheil_Example'
dataset_all: pd.DataFrame = import_data(file_name)
elements = extract_elements(dataset_all)

print('\n'.join([f'{i}, {c}' for i, c in enumerate(dataset_all.columns)]))
outputs = ['Vol_Fra_SOL', 'Temp_(C)']

# %% Graph a single material

dataset = dataset_all[dataset_all['Material'] == 1]

solidification_range = dataset.iloc[0]['Liquidus_temperature_(C)'] - dataset.iloc[0]['Solidus_temperature_(C)']
solidus_scheil = int(dataset['Temp_(C)'].min())

colors_number = len(dataset.groupby('Label'))
cmap = plt.colormaps.get_cmap('managua_r')
lower_bound = 0.0
upper_bound = 1.0
gap_start = 0.20
gap_end = 0.75
half1 = colors_number // 2
half2 = colors_number - half1
first_half = np.linspace(lower_bound, gap_start, half1)
second_half = np.linspace(gap_end, upper_bound, half2)
spliced_intervals = np.concatenate([first_half, second_half])
color_list = [cmap(val) for val in spliced_intervals]

from matplotlib.lines import Line2D

graph_properties = calculate_graph_properties(dataset, outputs)
import seaborn as sns
fig, ax = plt.subplots(figsize=(8, 8), frameon=True)
set_plot_parameters(ax, outputs, graph_properties)

sns.lineplot(data = dataset, x = outputs[0], y = outputs[1], hue = 'Label', palette=color_list, alpha = 1.0, legend = None, linewidth = 3, zorder = 2)

sns.scatterplot(data = dataset, x = outputs[0], y = outputs[1], hue = 'Label', palette=color_list, s = 100, alpha = 1.0, legend = 'full', edgecolors = None, linewidth = None, zorder = 2)


sns.lineplot(x = [0, 1],
              y = [dataset.iloc[0]['Liquidus_temperature_(C)'],
                    dataset.iloc[0]['Solidus_temperature_(C)']], linewidth = 2.5,
              linestyle = '--', color = 'k', marker = 'o', markersize = 12, zorder = 1)

plt.text(graph_properties['plot_x_min'] + 0.16, dataset.iloc[0]['Liquidus_temperature_(C)'] - 0.0 * solidification_range, f"{int(dataset.iloc[0]['Liquidus_temperature_(C)'] + 0.5)}\n(°C)", fontsize = 20, color = 'k', ha = 'left', va = 'center', zorder = 3)
plt.text(graph_properties['plot_x_max'] - 0.01, dataset.iloc[0]['Solidus_temperature_(C)'] + 0.1 * solidification_range, f"{int(dataset.iloc[0]['Solidus_temperature_(C)'] + 0.5)}\n(°C)", fontsize = 20, color = 'k', ha = 'right', va = 'bottom', zorder = 3)

ax.plot([0.05, 0.10], [dataset.iloc[0]['Liquidus_temperature_(C)'], dataset.iloc[0]['Liquidus_temperature_(C)']], color = 'k', linestyle = '-', linewidth = 3, zorder = 1)
ax.plot([0.05, 0.10], [dataset.iloc[0]['Solidus_temperature_(C)'], dataset.iloc[0]['Solidus_temperature_(C)']], color = 'k', linestyle = '-', linewidth = 3, zorder = 1)
ax.plot([0.10, 0.10], [dataset.iloc[0]['Solidus_temperature_(C)'], dataset.iloc[0]['Liquidus_temperature_(C)']], color = 'k', linestyle = '-', linewidth = 3, zorder = 1)
plt.text(0.12, dataset.iloc[0]['Solidus_temperature_(C)'], 'EQ\nsolidification range', fontsize = 20, color = 'k', ha = 'left', va = 'bottom', zorder = 3)
plt.text(0.12, np.mean([dataset.iloc[0]['Liquidus_temperature_(C)'], dataset.iloc[0]['Solidus_temperature_(C)']]), f'{int(solidification_range + 0.5)} (K)', fontsize = 20, color = 'k', ha = 'left', va = 'center', zorder = 3)

ax.plot([graph_properties['plot_x_min'], 0.5], [solidus_scheil, solidus_scheil], color = 'k', linestyle = ':', linewidth = 4, zorder = 1)
plt.text(graph_properties['plot_x_min'] + 0.01, solidus_scheil + 0.01 * graph_properties['plot_y_range'], f'{solidus_scheil} (°C)', fontsize = 20, color = 'k', ha = 'left', va = 'bottom', zorder = 3)

for i, mean in enumerate(
        dataset.groupby('Label', sort = False)[['Temp_(C)','Vol_Fra_SOL']].mean().values):
    plt.text(round(mean[1], 3) - graph_properties['plot_x_range'] * 0.01,
              int(mean[0]) - graph_properties['plot_y_range'] * 0.03,
              f'{colors_number - i}',
              fontsize = 30, color = 'k', ha = 'right', va = 'center', zorder = 3)

text_list = [f'{col}: {dataset[col].iloc[0]:.3f}' if len(col) == 2 else f' {col}: {dataset[col].iloc[0]:.3f}' for col in elements]
text_composition = '\n'.join(text_list)
ax.text(1.02, 1.0, f'{text_composition}', transform=ax.transAxes,
        ha = 'left', va = 'top', fontsize = 20, fontname = 'Consolas',
        bbox = dict(facecolor = '#FAE4FF', alpha = 1.0, edgecolor = 'black', lw = 2))


ax.set_ylabel('Temperature (°C)')
ax.set_xlabel('Volume fraction of solid')

class NumberedHandler:
    def __init__(self, color, number):
        self.color = color
        self.number = number
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = Line2D([x0 + width/2], [y0 + height/2], marker = 'o', color = 'w',
                        markerfacecolor = self.color, markersize = 30,
                        markeredgecolor = 'black', markeredgewidth = 1.5)
        handlebox.add_artist(patch)

        txt = plt.Text(x0 + width/2, y0 + height/2, str(self.number), 
                        va='center', ha='center', color='k',
                        fontsize= fontsize + 6, weight='normal')
        handlebox.add_artist(txt)
        return patch

def wrap_label(label):
    words = label.split()
    label = '\n'.join([' '.join(words[:5]), 
                         ' '.join(words[5:])])
    return label

handles, labels = ax.get_legend_handles_labels()

handles = [Line2D([0], [0]) for _ in range(colors_number)]
handler_map = {handle: NumberedHandler(color, colors_number - i) for i, (handle, color) in enumerate(
    zip(handles, color_list))}

labels = [label[:3] + label[6:] for label in labels]
labels = [label if len(label.split()) <= 5 else wrap_label(label) for label in labels]

legend = ax.legend(
    handles=handles, 
    labels=labels, 
    handler_map=handler_map,
    markerfirst=False, 
    fontsize=14, 
    loc='upper right', 
    bbox_to_anchor=[1, 1], 
    markerscale=3, 
    reverse=True, 
    framealpha=0.0
)
for text in legend.get_texts():
    text.set_ha('right')



plt.show()
    
# composition = int(dataset.iloc[0,0])
# save_graphs = True
# if save_graphs:
#     dpi = 96
#     export_file_name = f'material_{composition}_scheil.png'
#     plt.savefig(export_file_name, format='png', dpi=dpi, bbox_inches='tight')



# %% Loop and save all materials

for i in dataset_all['Material'].unique():
    dataset = dataset_all[dataset_all['Material'] == i]
    solidification_range = dataset.iloc[0]['Liquidus_temperature_(C)'] - dataset.iloc[0]['Solidus_temperature_(C)']
    solidus_scheil = int(dataset['Temp_(C)'].min())
    
    colors_number = len(dataset.groupby('Label'))
    cmap = plt.colormaps.get_cmap('managua_r')
    lower_bound = 0.0
    upper_bound = 1.0
    gap_start = 0.20
    gap_end = 0.75
    half1 = colors_number // 2
    half2 = colors_number - half1
    first_half = np.linspace(lower_bound, gap_start, half1)
    second_half = np.linspace(gap_end, upper_bound, half2)
    spliced_intervals = np.concatenate([first_half, second_half])
    color_list = [cmap(val) for val in spliced_intervals]
    
    from matplotlib.lines import Line2D
    
    graph_properties = calculate_graph_properties(dataset, outputs)
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(8, 8), frameon=True)
    set_plot_parameters(ax, outputs, graph_properties)
    
    sns.lineplot(data = dataset, x = outputs[0], y = outputs[1], hue = 'Label', palette=color_list, alpha = 1.0, legend = None, linewidth = 3, zorder = 2)
    
    sns.scatterplot(data = dataset, x = outputs[0], y = outputs[1], hue = 'Label', palette=color_list, s = 100, alpha = 1.0, legend = 'full', edgecolors = None, linewidth = None, zorder = 2)
    
    
    sns.lineplot(x = [0, 1],
                  y = [dataset.iloc[0]['Liquidus_temperature_(C)'],
                        dataset.iloc[0]['Solidus_temperature_(C)']], linewidth = 2.5,
                  linestyle = '--', color = 'k', marker = 'o', markersize = 12, zorder = 1)
    
    plt.text(graph_properties['plot_x_min'] + 0.16, dataset.iloc[0]['Liquidus_temperature_(C)'] - 0.0 * solidification_range, f"{int(dataset.iloc[0]['Liquidus_temperature_(C)'] + 0.5)}\n(°C)", fontsize = 20, color = 'k', ha = 'left', va = 'center', zorder = 3)
    plt.text(graph_properties['plot_x_max'] - 0.01, dataset.iloc[0]['Solidus_temperature_(C)'] + 0.1 * solidification_range, f"{int(dataset.iloc[0]['Solidus_temperature_(C)'] + 0.5)}\n(°C)", fontsize = 20, color = 'k', ha = 'right', va = 'bottom', zorder = 3)
    
    ax.plot([0.05, 0.10], [dataset.iloc[0]['Liquidus_temperature_(C)'], dataset.iloc[0]['Liquidus_temperature_(C)']], color = 'k', linestyle = '-', linewidth = 3, zorder = 1)
    ax.plot([0.05, 0.10], [dataset.iloc[0]['Solidus_temperature_(C)'], dataset.iloc[0]['Solidus_temperature_(C)']], color = 'k', linestyle = '-', linewidth = 3, zorder = 1)
    ax.plot([0.10, 0.10], [dataset.iloc[0]['Solidus_temperature_(C)'], dataset.iloc[0]['Liquidus_temperature_(C)']], color = 'k', linestyle = '-', linewidth = 3, zorder = 1)
    plt.text(0.12, dataset.iloc[0]['Solidus_temperature_(C)'], 'EQ\nsolidification range', fontsize = 20, color = 'k', ha = 'left', va = 'bottom', zorder = 3)
    plt.text(0.12, np.mean([dataset.iloc[0]['Liquidus_temperature_(C)'], dataset.iloc[0]['Solidus_temperature_(C)']]), f'{int(solidification_range + 0.5)} (K)', fontsize = 20, color = 'k', ha = 'left', va = 'center', zorder = 3)
    
    ax.plot([graph_properties['plot_x_min'], 0.5], [solidus_scheil, solidus_scheil], color = 'k', linestyle = ':', linewidth = 4, zorder = 1)
    plt.text(graph_properties['plot_x_min'] + 0.01, solidus_scheil + 0.01 * graph_properties['plot_y_range'], f'{solidus_scheil} (°C)', fontsize = 20, color = 'k', ha = 'left', va = 'bottom', zorder = 3)
    
    for i, mean in enumerate(
            dataset.groupby('Label', sort = False)[['Temp_(C)','Vol_Fra_SOL']].mean().values):
        plt.text(round(mean[1], 3) - graph_properties['plot_x_range'] * 0.01,
                  int(mean[0]) - graph_properties['plot_y_range'] * 0.03,
                  f'{colors_number - i}',
                  fontsize = 30, color = 'k', ha = 'right', va = 'center', zorder = 3)
    
    text_list = [f'{col}: {dataset[col].iloc[0]:.3f}' if len(col) == 2 else f' {col}: {dataset[col].iloc[0]:.3f}' for col in elements]
    text_composition = '\n'.join(text_list)
    ax.text(1.02, 1.0, f'{text_composition}', transform=ax.transAxes,
            ha = 'left', va = 'top', fontsize = 20, fontname = 'Consolas',
            bbox = dict(facecolor = '#FAE4FF', alpha = 1.0, edgecolor = 'black', lw = 2))
    
    
    ax.set_ylabel('Temperature (°C)')
    ax.set_xlabel('Volume fraction of solid')
    
    handles, labels = ax.get_legend_handles_labels()
    
    handles = [Line2D([0], [0]) for _ in range(colors_number)]
    handler_map = {handle: NumberedHandler(color, colors_number - i) for i, (handle, color) in enumerate(
        zip(handles, color_list))}
    
    labels = [label[:3] + label[6:] for label in labels]
    labels = [label if len(label.split()) <= 5 else wrap_label(label) for label in labels]
    
    legend = ax.legend(
        handles=handles, 
        labels=labels, 
        handler_map=handler_map,
        markerfirst=False, 
        fontsize=14, 
        loc='upper right', 
        bbox_to_anchor=[1, 1], 
        markerscale=3, 
        reverse=True, 
        framealpha=0.0
    )
    for text in legend.get_texts():
        text.set_ha('right')
        
    composition = int(dataset.iloc[0,0])
    save_graphs = True
    if save_graphs:
        dpi = 96
        export_file_name = f'material_{composition}_scheil.png'
        plt.savefig(export_file_name, format='png', dpi=dpi, bbox_inches='tight')


