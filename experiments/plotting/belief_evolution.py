import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the CSV file
file_path = os.path.join(os.path.dirname(__file__), "../output/communication_vision/results.csv")
df = pd.read_csv(file_path)

def plot_most_likely_gem_values(problem_name, plot_type='scatter'):
    # Filter data for the specific problem
    problem_data = df[(df['problem_name'] == problem_name) & (df['repeat'] == 1)]

    # Function to find the most likely value and its probability for each timestep for a specific gem color
    def most_likely_values_and_probs(agent_data, gem_prefix):
        cols = [f'{gem_prefix}_m5', f'{gem_prefix}_1', f'{gem_prefix}_2', f'{gem_prefix}_3']
        values = np.array([-5, 1, 2, 3])
        probs = agent_data[cols].values
        max_indices = np.argmax(probs, axis=1)
        return values[max_indices], np.max(probs, axis=1)

    # Map the gem values to discrete numbers for categorical plotting
    value_map = {-5: 0, 1: 1, 2: 2, 3: 3}

    for current_agent in [1, 2]:
        other_agent = 3 - current_agent  # If current_agent is 1, other_agent is 2, and vice versa

        # Filter data for the current agent within the specific problem
        agent_data = problem_data[problem_data['agent'] == current_agent].copy()
        other_agent_data = problem_data[problem_data['agent'] == other_agent].copy()

        # Calculate most likely gem value and its probability for each timestep for all gem colors
        for color in ['red', 'blue', 'green', 'yellow']:
            agent_data[f'most_likely_{color}'], agent_data[f'{color}_prob'] = most_likely_values_and_probs(agent_data, color)
            agent_data[f'most_likely_{color}_mapped'] = agent_data[f'most_likely_{color}'].map(value_map)

        # Apply small vertical offsets to make points more distinct
        offsets = [-0.15, -0.05, 0.05, 0.15]
        for color, offset in zip(['red', 'blue', 'green', 'yellow'], offsets):
            agent_data[f'most_likely_{color}_mapped'] += offset

        # Extend the last beliefs to t=100
        last_timestep = agent_data['timestep'].max()
        if last_timestep < 100:
            extension = pd.DataFrame({
                'timestep': range(last_timestep + 1, 101),
                'agent': current_agent,
                'pickup': 'none'
            })
            for color in ['red', 'blue', 'green', 'yellow']:
                extension[f'most_likely_{color}_mapped'] = agent_data[f'most_likely_{color}_mapped'].iloc[-1]
                extension[f'{color}_prob'] = agent_data[f'{color}_prob'].iloc[-1]
            agent_data = pd.concat([agent_data, extension], ignore_index=True)

        # Create a figure with 5 subplots (1 main + 4 for individual gem probabilities)
        fig, axs = plt.subplots(5, 1, figsize=(12, 20), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1, 1, 1]})

        colors = ['crimson', 'royalblue', 'forestgreen', 'gold']
        labels = ['Red Gem', 'Blue Gem', 'Green Gem', 'Yellow Gem']

        # Add vertical lines for current agent's pickups across all subplots
        pickup_timesteps = agent_data[agent_data['pickup'] != 'none']['timestep']
        for t in pickup_timesteps:
            for ax in axs:
                ax.axvline(x=t, color='black', linestyle='-', alpha=0.2, zorder=0)

        # Add vertical lines for other agent's communication (received at t+1) across all subplots
        comm_timesteps = other_agent_data[other_agent_data['utterance'] != 'none']['timestep']
        for t in comm_timesteps:
            if t < 100:  # Ensure we don't plot beyond t=100
                for ax in axs:
                    ax.axvline(x=t+1, color='black', linestyle='--', alpha=0.15, zorder=0)

        # Main plot
        for color, label in zip(colors, labels):
            if plot_type == 'scatter':
                axs[0].scatter(agent_data['timestep'], agent_data[f'most_likely_{label.split()[0].lower()}_mapped'], 
                               label=label, color=color, alpha=0.8, s=30)
            else:  # line plot
                axs[0].plot(agent_data['timestep'], agent_data[f'most_likely_{label.split()[0].lower()}_mapped'], 
                            label=label, color=color, alpha=0.8)

        # Adjust y-ticks to reflect the reward values
        axs[0].set_yticks(list(value_map.values()))
        axs[0].set_yticklabels(list(value_map.keys()))

        # Set axis limits for main plot
        axs[0].set_xlim(1, 50)
        axs[0].set_ylim(-0.6, 3.5)

        axs[0].set_ylabel('Reward')
        axs[0].set_title(f"Evolution of Reward Beliefs for Agent {current_agent}")
        
        # Create a custom legend for main plot
        legend_elements = [plt.Line2D([0], [0], color=color, label=label) for color, label in zip(colors, labels)]
        legend_elements.append(plt.Line2D([0], [0], color='black', linestyle='-', alpha=0.2, label=f'Agent {current_agent} Pickup'))
        legend_elements.append(plt.Line2D([0], [0], color='black', linestyle='--', alpha=0.15, label=f'Agent {other_agent} Communicated'))
        axs[0].legend(handles=legend_elements, loc='best')

        # Plot individual gem probabilities with shaded areas (reordered)
        for i, color in enumerate(['yellow', 'green', 'red', 'blue']):
            axs[i+1].plot(agent_data['timestep'], agent_data[f'{color}_prob'], color=colors[['red', 'blue', 'green', 'yellow'].index(color)])
            axs[i+1].fill_between(agent_data['timestep'], 0, agent_data[f'{color}_prob'], color=colors[['red', 'blue', 'green', 'yellow'].index(color)], alpha=0.1)
            axs[i+1].set_ylabel(f'{color.capitalize()} Prob')
            axs[i+1].set_ylim(0, 1)

        # Set common x-axis label
        axs[-1].set_xlabel('Timestep')

        # Remove overlapping x-axis labels
        for ax in axs[:-1]:
            ax.tick_params(labelbottom=False)

        plt.tight_layout()

        # Ensure the output directory exists
        output_dir = os.path.join(os.path.dirname(__file__), "../output/communication_vision/plots")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the plot
        plot_type_str = 'scatter' if plot_type == 'scatter' else 'line'
        output_path = os.path.join(output_dir, f'{problem_name.replace("/", "-")}_belief_evolution_agent_{current_agent}_{plot_type_str}.png')
        print(f"Saving plot to {output_path}")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

# Example usage:
plot_most_likely_gem_values('medium/1', plot_type='scatter')
plot_most_likely_gem_values('medium/1', plot_type='line')