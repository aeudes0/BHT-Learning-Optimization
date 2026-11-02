import pandas as pd
import matplotlib.pyplot as plt

# 1. Read the data with the correct separators for your European CSV format
df = pd.read_csv('perf.csv', sep=';', decimal=',')

# 2. Filter out the empty rows (future dates where no revenue is logged yet)
df = df.dropna(subset=['Alois avg revenue', 'Simon avg revenue'], how='all').copy()

# 3. Create the 'Round' text label for the X-axis dynamically
df['Round'] = 'Week ' + df['Week'].astype(int).astype(str) + ' Sim ' + df['Simulation round'].astype(int).astype(str)

# 4. Set up the 2x2 Plotting Grid
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Agent Comparison: Alois vs. Simon', fontsize=18, fontweight='bold')

my_color, my_marker, my_style = 'tab:blue', 'o', '-'
cw_color, cw_marker, cw_style = 'tab:orange', 'x', '--'

def format_xaxis(ax):
    ax.set_xticks(range(len(df['Round'])))
    ax.set_xticklabels(df['Round'], rotation=45, ha='right')

# Top-Left: Average Revenue
axes[0, 0].plot(df['Round'], df['Alois avg revenue'], label='Alois', color=my_color, marker=my_marker, linestyle=my_style, linewidth=2)
axes[0, 0].plot(df['Round'], df['Simon avg revenue'], label='Simon', color=cw_color, marker=cw_marker, linestyle=cw_style, linewidth=2)
axes[0, 0].set_title('Average Revenue (€)')
format_xaxis(axes[0, 0])
axes[0, 0].legend()

# Top-Right: Simulation Position (Rank)
axes[0, 1].plot(df['Round'], df['Alois position'], label='Alois', color=my_color, marker=my_marker, linestyle=my_style, linewidth=2)
axes[0, 1].plot(df['Round'], df['Simon position'], label='Simon', color=cw_color, marker=cw_marker, linestyle=cw_style, linewidth=2)
axes[0, 1].set_title('Simulation Position (Rank)')
format_xaxis(axes[0, 1])
axes[0, 1].invert_yaxis() # Invert so Rank 1 is at the top
axes[0, 1].legend()

# Bottom-Left: Winrate
axes[1, 0].plot(df['Round'], df['Alois winrate (%)'], label='Alois', color=my_color, marker=my_marker, linestyle=my_style, linewidth=2)
axes[1, 0].plot(df['Round'], df['Simon winrate'], label='Simon', color=cw_color, marker=cw_marker, linestyle=cw_style, linewidth=2)
axes[1, 0].set_title('Winrate')
format_xaxis(axes[1, 0])
axes[1, 0].legend()

# Bottom-Right: Stockout Frequency
axes[1, 1].plot(df['Round'], df['Alois stockout frequency'], label='Alois', color=my_color, marker=my_marker, linestyle=my_style, linewidth=2)
axes[1, 1].plot(df['Round'], df['Simon stockout frequency'], label='Simon', color=cw_color, marker=cw_marker, linestyle=cw_style, linewidth=2)
axes[1, 1].set_title('Stockout Frequency')
format_xaxis(axes[1, 1])
axes[1, 1].legend()

plt.tight_layout()
plt.subplots_adjust(top=0.92, bottom=0.15) 

# 5. Save the plot as an image and display it
plt.savefig('simulation_dashboard.png', dpi=300)
print("Dashboard successfully saved as 'simulation_dashboard.png'")
plt.show()