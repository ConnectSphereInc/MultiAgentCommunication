using DataStructures: OrderedDict
using PDDLViz: RGBA, to_color, set_alpha
using Logging
using OpenAI

"Gets the (x, y) position of the specified agent."
function get_agent_pos(state::State, agent::Symbol)
    return (state[Compound(:xloc, Term[Const(agent)])],
            state[Compound(:yloc, Term[Const(agent)])])
end

"""
    get_top_weighted_rewards(state::ParticleFilterState, n::Int)

    Get the top `n` most likely reward distributions based on the particle filter state.
"""
function get_top_weighted_rewards(state::ParticleFilterState, n::Int, possible_gems::Vector{Symbol})
    traces = get_traces(state)
    weights = get_norm_weights(state)
    reward_weights = Dict()
    
    # Accumulate rewards and weights
    for (tr, w) in zip(traces, weights)
        rewards = Dict{Symbol, Int}()
        for gem in possible_gems
            rewards[gem] = tr[:reward => gem]
        end
        rewards_tuple = Tuple(sort(collect(rewards)))
        reward_weights[rewards_tuple] = get(reward_weights, rewards_tuple, 0.0) + w
    end
    
    total_weight = sum(values(reward_weights))
    weighted_rewards = [(Dict(rewards), weight / total_weight) 
                        for (rewards, weight) in reward_weights]
    
    # Sort by probability
    sort!(weighted_rewards, by = x -> x[2], rev = true)
    
    # Return top n results
    return weighted_rewards[1:min(n, length(weighted_rewards))]
end

function quantify_gem_certainty(weighted_rewards)
    total_weight = sum(wr[2] for wr in weighted_rewards)
    gem_values = Dict(gem => Dict() for gem in keys(weighted_rewards[1][1]))
    for (rewards, weight) in weighted_rewards
        for (gem, value) in rewards
            value_float = if isa(value, Number)
                Float64(value)
            else
                parse(Float64, value)
            end
            if !haskey(gem_values[gem], value_float)
                gem_values[gem][value_float] = 0.0
            end
            gem_values[gem][value_float] += weight
        end
    end
    gem_certainty = Dict()
    for (gem, values) in gem_values
        probability, most_likely_value = findmax(values)
        certainty = (probability / total_weight) * 100
        gem_certainty[gem] = Dict(
            "most_likely_value" => most_likely_value,
            "certainty_percentage" => round(certainty, digits=1)
        )
    end
    return gem_certainty
end

function calculate_gem_utility(gem_certainty; risk_aversion=0.0, certainty_threshold=0.5)
    utilities = Dict{Symbol, Int}()
    certainties = Dict{Symbol, Float64}()
    for (gem, info) in gem_certainty
        value = if isa(info["most_likely_value"], Number)
            info["most_likely_value"]
        else
            parse(Float64, info["most_likely_value"])
        end
        certainty = info["certainty_percentage"] / 100
        
        # Favor positive rewards when certainty is low
        if certainty < certainty_threshold && value < 0
            utility = 1.0  # Ensure a minimum positive utility
        else
            utility = (1 - risk_aversion) * value + risk_aversion * certainty * value
        end
        
        # Add error checking
        if isnan(utility)
            @warn "NaN utility calculated for gem $gem. Using 0 instead."
            utility = 0
        end
        
        utilities[gem] = round(Int, utility)  # Round to nearest integer
        certainties[gem] = certainty
    end
    return utilities, certainties
end

"""
    gem_from_utterance(utterance::String)

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

function gem_to_color(gem::Symbol)
    return Symbol(split(string(gem), "_")[1])
end

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

function print_estimated_rewards(agent, beliefs, certainties)
    @info "       $agent's Estimated Rewards:"
    for (gem, value) in beliefs
        @info "              $gem: value = $value, certainty = $(round(certainties[gem], digits=2))"
    end
end

function setup_logging(output_folder::String, filename::String)
    # Ensure the output folder exists
    mkpath(output_folder)
    
    log_file = joinpath(output_folder, filename)
    io = open(log_file, "w")
    logger = CleanLogger(io, Logging.Info)
    global_logger(logger)
    return io
end

struct CleanLogger <: AbstractLogger
    io::IO
    min_level::LogLevel
end

function Logging.handle_message(logger::CleanLogger, level, message, _module, group, id, file, line; kwargs...)
    if level >= logger.min_level
        println(logger.io, message)
    end
    return nothing
end

Logging.shouldlog(logger::CleanLogger, level, _module, group, id) = level >= logger.min_level
Logging.min_enabled_level(logger::CleanLogger) = logger.min_level
Logging.catch_exceptions(logger::CleanLogger) = false

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