using DataStructures: OrderedDict
using PDDLViz: RGBA, to_color, set_alpha
using Logging
using OpenAI

"""
    Gets the (x, y) position of the specified agent.
"""
function get_agent_pos(state::State, agent::Symbol)
    return (state[Compound(:xloc, Term[Const(agent)])],
            state[Compound(:yloc, Term[Const(agent)])])
end

"""
    Gets the probability of each gem being assigned each possible reward.
"""
function get_gem_reward_probabilities(state::ParticleFilterState, possible_gems::Vector{Symbol}, possible_values::Vector{Int})
    traces = get_traces(state)
    weights = get_norm_weights(state)
    
    gem_value_counts = Dict(gem => Dict(value => 0.0 for value in possible_values) for gem in possible_gems)
    total_weight = sum(weights)

    for (tr, w) in zip(traces, weights)
        for gem in possible_gems
            value = tr[:reward => gem]
            gem_value_counts[gem][value] += w
        end
    end
    
    gem_value_probs = Dict(
        gem => Dict(
            reward_value => count / total_weight
            for (reward_value, count) in value_counts
        )
        for (gem, value_counts) in gem_value_counts
    )
    return gem_value_probs
end

"""
    Get the reawrd of each gem as the most likely value, or if uncertain, assign a random positive reward to encourate exploration.
"""
function calculate_gem_utility(gem_value_probs, possible_rewards; certainty_threshold=0.5)
    utilities = Dict{Symbol, Int}()
    
    positive_values = filter(x -> x > 0, possible_rewards)
    
    for (gem, value_probs) in gem_value_probs
        # Find the most likely value and its probability
        max_prob, most_likely_value = findmax(value_probs)

        # Determine utility
        if max_prob < certainty_threshold
            utility = rand(positive_values)
        else
            utility = most_likely_value
        end
        
        utilities[gem] = utility
    end
    
    return utilities
end

"""
    Manually extract the color of a gem from an utterance using regex.
"""
function parse_gem(utterance::String)
    # First, try to match full color names
    color_pattern = r"\b(red|blue|yellow|green)\b"
    match_result = match(color_pattern, lowercase(utterance))
    if match_result !== nothing
        return String(match_result.match)
    end
    
    # If no match, try to extract color from gem names like "blue_gem2"
    gem_pattern = r"\b(red|blue|yellow|green)_gem\d*\b"
    match_result = match(gem_pattern, lowercase(utterance))
    if match_result !== nothing
        return Symbol(split(match_result.match, "_")[1])
    end
    
    return nothing
end

"""
    parse_reward(utterance::String, gem::String)

    Manually extract the reward from an utterance using regex.
"""
function parse_reward(utterance::String)
    score_pattern = r"\b(-1|1|3|5)\b"
    match_result = match(score_pattern, utterance)
    if match_result !== nothing
        return parse(Int, match_result.match)
    end
    return nothing
end

"""
    best_action(sol::TabularVPolicy, state::State, agent::Symbol)

    Returns the best action for the specified agent in the given state.
"""
function best_action(sol::TabularVPolicy, state::State, agent::Symbol)
    best_val = -Inf
    best_acts = []
    for act in available(sol.domain, state)
        if Symbol(act.args[1]) == agent
            val = get_value(sol, state, act)
            if val > best_val
                best_val = val
                best_acts = [act]
            elseif val == best_val
                push!(best_acts, act)
            end
        end
    end
    return isempty(best_acts) ? missing : rand(best_acts)
end

"""
    boltzmann_action(sol::TabularVPolicy, state::State, agent::Symbol, temperature::Float64)

    Returns an action for the specified agent in the given state using the Boltzmann distribution.
"""
function boltzmann_action(sol::TabularVPolicy, state::State, agent::Symbol, temperature::Float64)
    actions = []
    values = []
    
    for act in available(sol.domain, state)
        if Symbol(act.args[1]) == agent
            push!(actions, act)
            push!(values, get_value(sol, state, act))
        end
    end
    
    if isempty(actions)
        return missing
    end
    
    if temperature == 0
        # Deterministic case: choose the action with the highest value
        max_value = maximum(values)
        best_actions = actions[values .== max_value]
        return rand(best_actions)
    else
        # Stochastic case: use Boltzmann distribution
        # Apply Gumbel-max trick for numerical stability
        scores = values ./ temperature .+ rand(Gumbel(), length(values))
        return actions[argmax(scores)]
    end
end

"""
    gem_to_color(gem::Symbol)

    Extracts the color of a gem from its name.
"""
function gem_to_color(gem::Symbol)
    return Symbol(split(string(gem), "_")[1])
end

"""
    setup_renderer(agents, gridworld_only)

    Sets up the renderer for the gridworld environment.
"""
function setup_renderer(agents, gridworld_only)
    return PDDLViz.GridworldRenderer(
        resolution = (600,1100),
        has_agent = false,
        obj_renderers = Dict{Symbol, Function}(
            key => (d, s, o) -> begin
                if key == :agent
                    PDDLViz.MultiGraphic(
                        PDDLViz.RobotGraphic(color = :slategray),
                        PDDLViz.TextGraphic(
                            string(o.name)[end:end], 0.3, 0.2, 0.5,
                            color = :black, font = :bold
                        )
                    )
                else
                    PDDLViz.GemGraphic(color = key)
                end
            end
            for key in [:agent, :red, :yellow, :blue, :green]
        ),
        show_inventory = !gridworld_only,
        inventory_fns = [
            (d, s, o) -> s[PDDL.Compound(:has, [PDDL.Const(agent), o])] for agent in agents
        ],
        inventory_types = [:item for agent in agents],
        inventory_labels = ["$agent Inventory" for agent in agents],
        show_vision = !gridworld_only,
        vision_fns = [
            (d, s, o) -> s[PDDL.Compound(:visible, [PDDL.Const(agent), o])] for agent in agents
        ],
        vision_types = [:item for agent in agents],
        vision_labels = ["$agent Vision" for agent in agents],
    )
end

"""
    gpt4o(prompt::String; max_retries=5, base_wait_time=1.0)

    Queries the GPT-4o model with the specified prompt.
"""
function gpt4o(prompt::String; max_retries=5, base_wait_time=1.0)
    secret_key = ENV["OPENAI_API_KEY"]
    model = "gpt-4o"

    for attempt in 1:max_retries
        try
            r = create_chat(
                secret_key,
                model,
                [Dict("role" => "user", "content" => prompt)]
            )
            
            return r.response[:choices][1][:message][:content]
        catch e
            if attempt < max_retries
                wait_time = base_wait_time * (2^(attempt - 1))
                sleep(wait_time)
            else
                throw(e)
            end
        end
    end
    
    error("Max retries reached. Unable to complete the request.")
end

"""
    parse_belief(input::String)

    Parses the agent's belief from the specified input string. For use with the `gpt4o` model.
"""
function parse_belief(input::String)
    beliefs = Dict{Symbol, Float64}()
    
    # Regular expression to match gem colors and their associated values
    pattern = r"(red|blue|yellow|green)\s*:\s*([-]?\d+(?:\.\d+)?)"
    
    # Find all matches in the input string
    matches = eachmatch(pattern, lowercase(input))
    
    # Process each match
    for m in matches
        color = Symbol(m.captures[1])
        value = parse(Float64, m.captures[2])
        beliefs[color] = value
    end
    
    # Ensure all colors are present, default to 0.0 if missing
    for color in [:red, :blue, :yellow, :green]
        if !haskey(beliefs, color)
            beliefs[color] = 0.0
        end
    end
    
    return beliefs
end

"""
    setup_results()

    Sets up the results DataFrame for storing simulation data.
"""
function setup_results()
    return DataFrame(
        timestep = Int[],
        agent = Int[],
        score = Float64[],
        gems_picked_up = Int[],
        red_m5 = Float64[], red_1 = Float64[], red_2 = Float64[], red_3 = Float64[],
        blue_m5 = Float64[], blue_1 = Float64[], blue_2 = Float64[], blue_3 = Float64[],
        green_m5 = Float64[], green_1 = Float64[], green_2 = Float64[], green_3 = Float64[],
        yellow_m5 = Float64[], yellow_1 = Float64[], yellow_2 = Float64[], yellow_3 = Float64[],
        pickup = String[],
        utterance = String[]
    )
end

"""
    setup_results()

    Sets up the results DataFrame for storing simulation data.
"""
function setup_results_gpt4o()
    return DataFrame(
        timestep = Int[],
        agent = Int[],
        score = Float64[],
        gems_picked_up = Int[],
        red = Float64[],
        blue = Float64[],
        green = Float64[],
        yellow = Float64[],
        pickup = String[],
        utterance = String[]
    )
end

"""
    append_to_results!(results, t, i, combined_score, total_gems_picked_up, gem_value_probs, item, utterance)
    
    Appends a row to the results DataFrame.
"""
function append_to_results!(results::DataFrame, t::Int, i::Int, combined_score::Int, total_gems_picked_up::Int, gem_value_probs::Dict, item::String, utterance::String)
    row = Dict(
        :timestep => t,
        :agent => i,
        :score => combined_score,
        :gems_picked_up => total_gems_picked_up,
        :red_m5 => gem_value_probs[:red][-5],
        :red_1 => gem_value_probs[:red][1],
        :red_2 => gem_value_probs[:red][2],
        :red_3 => gem_value_probs[:red][3],
        :blue_m5 => gem_value_probs[:blue][-5],
        :blue_1 => gem_value_probs[:blue][1],
        :blue_2 => gem_value_probs[:blue][2],
        :blue_3 => gem_value_probs[:blue][3],
        :green_m5 => gem_value_probs[:green][-5],
        :green_1 => gem_value_probs[:green][1],
        :green_2 => gem_value_probs[:green][2],
        :green_3 => gem_value_probs[:green][3],
        :yellow_m5 => gem_value_probs[:yellow][-5],
        :yellow_1 => gem_value_probs[:yellow][1],
        :yellow_2 => gem_value_probs[:yellow][2],
        :yellow_3 => gem_value_probs[:yellow][3],
        :pickup => item,
        :utterance => utterance
    )
    push!(results, row)
end

"""
    append_to_results!(results, t, i, combined_score, total_gems_picked_up, gem_value_probs, item, utterance)
    
    Appends a row to the results DataFrame.
"""
function append_to_results_gpt4o!(results::DataFrame, t::Int, i::Int, combined_score::Int, total_gems_picked_up::Int, gem_values::Dict, item::String, utterance::String)
    row = Dict(
        :timestep => t,
        :agent => i,
        :score => combined_score,
        :gems_picked_up => total_gems_picked_up,
        :red => gem_values[:red],
        :blue => gem_values[:blue],
        :green => gem_values[:green],
        :yellow => gem_values[:yellow],
        :pickup => item,
        :utterance => utterance
    )
    push!(results, row)
end

"""
    get_visible_gems(domain::Domain, state::State, agent::Symbol)

    Get the visible gems in the state.
"""
function get_visible_gems(domain::Domain, state::State, agent::Symbol)
    visible_gems = Tuple{Symbol, Tuple{Int64, Int64}}[]
    for obj in PDDL.get_objects(domain, state, :item)
        if PDDL.satisfy(domain, state, PDDL.parse_pddl("(visible $agent $(obj.name))"))
            x = state[Compound(:xloc, [obj])]
            y = state[Compound(:yloc, [obj])]
            push!(visible_gems, (Symbol(obj.name), (x, y)))
        end
    end
    return visible_gems
end

"""
    astar_path_length(domain::Domain, state::State, agent::Symbol, gem_pos::Tuple{Int,Int})

    Compute the length of the shortest path from the agent to the gem at the given position.
"""
function astar_path_length(domain::Domain, state::State, agent::Symbol, gem_pos::Tuple{Int,Int})
    goal = PDDL.parse_pddl("(and (= (xloc $agent) $(gem_pos[1])) (= (yloc $agent) $(gem_pos[2])))")
    planner = AStarPlanner(GoalCountHeuristic())
    spec = MinStepsGoal(goal)
    solution = planner(domain, state, spec)
    return length(solution)
end

"""
    get_gem_on_tile(domain::Domain, state::State, agent::Symbol)

    Get the gem on the same tile as the agent.
"""
function get_gem_on_tile(domain::Domain, state::State, agent::Symbol)
    agent_pos = get_agent_pos(state, agent)
    agent_x, agent_y = agent_pos
    
    for obj in PDDL.get_objects(domain, state, :item)
        x = state[Compound(:xloc, [obj])]
        y = state[Compound(:yloc, [obj])]
        if x == agent_x && y == agent_y
            return (Symbol(obj.name), (x, y))
        end
    end
    
    return nothing
end

"""
    get_on_grid_gems(domain::Domain, state::State)

    Get the gems that are on the grid.
"""
function get_on_grid_gems(domain::Domain, state::State)
    on_grid_gems = Tuple{Symbol, Tuple{Int64, Int64}}[]
    all_gems = PDDL.get_objects(domain, state, :gem)    
    for obj in all_gems
        is_offgrid = PDDL.satisfy(domain, state, PDDL.parse_pddl("(offgrid $(obj.name))"))
        if !is_offgrid
            x = state[Compound(:xloc, [obj])]
            y = state[Compound(:yloc, [obj])]
            push!(on_grid_gems, (Symbol(obj.name), (x, y)))
        end
    end
    return on_grid_gems
end