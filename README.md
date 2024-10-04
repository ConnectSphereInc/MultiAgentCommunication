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

```bash
julia --project=. experiments/main.jl
```