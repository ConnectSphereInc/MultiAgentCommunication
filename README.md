# MultiAgentCommunication

## Requirements

- Julia
- OpenAI API Key

## Installation

Install the required Julia packages:

```bash
julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate()'
```

Create a `.env` file in the root directory with the following content:

```bash
OPENAI_API_KEY=<your-api-key>
```

## Usage
The experiment can be run with: 

```bash
JULIA_NUM_THREADS=<THREADS> julia --project=. experiments/main.jl
```

Figures can be generated with:

```bash
python experiment/plotting/belief_evolution.py
```

```bash
python experiment/plotting/belief_heatmaps.py
```

Tables can be generated with:

```bash
python experiment/analysis/analysis.py
```