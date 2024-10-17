import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the CSV file
file_path = os.path.join(os.path.dirname(__file__), "../output/communication_vision/results.csv")
df = pd.read_csv(file_path)

def plot_combined_beliefs_and_probabilities(problem_name, repeat, plot_type='scatter'):
    # Filter data for the specific problem
    problem_data = df[(df['problem_name'] == problem_name) & (df['repeat'] == repeat)]

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

        # Create a figure with 1 subplot
        fig, ax = plt.subplots(figsize=(24, 4))

        colors = ['crimson', 'royalblue', 'forestgreen', 'gold']
        labels = ['Red Gem', 'Blue Gem', 'Green Gem', 'Yellow Gem']

        # Add vertical lines for current agent's pickups
        pickup_timesteps = agent_data[agent_data['pickup'] != 'none']['timestep']
        for t in pickup_timesteps:
            ax.axvline(x=t, color='black', linestyle='-', alpha=0.2, zorder=0)

        # Add vertical lines for other agent's communication (received at t+1)
        comm_timesteps = other_agent_data[other_agent_data['utterance'] != 'none']['timestep']
        for t in comm_timesteps:
            if t < 100:  # Ensure we don't plot beyond t=100
                ax.axvline(x=t+1, color='black', linestyle='--', alpha=0.15, zorder=0)

        # Plot beliefs and probabilities
        for color, label in zip(colors, labels):
            gem_color = label.split()[0].lower()
            if plot_type == 'scatter':
                ax.scatter(agent_data['timestep'], agent_data[f'most_likely_{gem_color}_mapped'], 
                           label=f'{label} Belief', color=color, alpha=0.8, s=60, zorder=2)
            else:  # line plot
                ax.plot(agent_data['timestep'], agent_data[f'most_likely_{gem_color}_mapped'], 
                        label=f'{label} Belief', color=color, alpha=0.9, zorder=2,linewidth=2)
            
            # Plot probabilities
            ax.plot(agent_data['timestep'], agent_data[f'{gem_color}_prob'] * 4 - 0.6, 
                    label=f'{label} Probability', color=color, alpha=0.6, linestyle='--', zorder=1)

        # Adjust y-ticks to reflect the reward values
        ax.set_yticks(list(value_map.values()))
        ax.set_yticklabels(list(value_map.keys()))

        # Add a second y-axis for probabilities
        ax2 = ax.twinx()
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('Probability')

        # Set axis limits
        ax.set_xlim(1, 100)
        ax.set_ylim(-0.6, 3.5)

        ax.set_ylabel('Reward')
        ax.set_xlabel('Timestep')
        ax.set_title(f"Evolution of Reward Beliefs and Probabilities for Agent {current_agent}")
        
        # Create a custom legend
        legend_elements = [plt.Line2D([0], [0], color=color, label=f'{label} Belief') for color, label in zip(colors, labels)]
        legend_elements += [plt.Line2D([0], [0], color=color, linestyle='--', alpha=0.4, label=f'{label} Probability') for color, label in zip(colors, labels)]
        legend_elements.append(plt.Line2D([0], [0], color='black', linestyle='-', alpha=0.2, label=f'Agent {current_agent} Pickup'))
        legend_elements.append(plt.Line2D([0], [0], color='black', linestyle='--', alpha=0.15, label=f'Agent {other_agent} Communicated'))
        ax.legend(handles=legend_elements, loc='best')

        plt.tight_layout()

        # Ensure the output directory exists
        output_dir = os.path.join(os.path.dirname(__file__), "../output/communication_vision/plots")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the plot
        plot_type_str = 'scatter' if plot_type == 'scatter' else 'line'
        output_path = os.path.join(output_dir, f'{problem_name.replace("/", "-")}_{repeat}_combined_belief_prob_agent_{current_agent}_{plot_type_str}.png')
        print(f"Saving plot to {output_path}")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

# Example usage:
plot_combined_beliefs_and_probabilities('medium/1', 1, plot_type='line')