module MultiAgentCommunication

using PDDL, PlanningDomains, SymbolicPlanners
using Gen, GenParticleFilters
using PDDLViz, GLMakie
using Random
using SymbolicPlanners: get_value, get_goal_terms
using DotEnv
using DataFrames
using Distributions

include("agent.jl")
include("utils.jl")
include("heuristics.jl")
include("inference.jl")

export run_simulation_communication_vision, run_simulation_no_communication_vision, run_simulation_communication_restricted_vision, run_simulation_communication_perfect_vision, run_simulation_gpt4o
export agent_model_communication, agent_model_no_communication
export update_beliefs_communication, update_beliefs_no_communication, enum_inference, enum_inference_step

function run_simulation_communication_vision(
    problem_path::String,
    output_folder::String,
    ess_thresh::Float64,
    num_particles::Int,
    ground_truth_rewards::Dict{Symbol, Int},
    T::Int,
    gridworld_only::Bool = false,
    inference_type::String = "enum"
)
    results = setup_results()

    # Load domain and problem
    domain = load_domain(joinpath(@__DIR__, "domain.pddl"))
    problem::Problem = load_problem(problem_path)
    initial_state = initstate(domain, problem)
    
    items = [obj.name for obj in PDDL.get_objects(domain, initial_state, :gem)]
    agents = Symbol[obj.name for obj in PDDL.get_objects(domain, initial_state, :agent)]
    
    # Renderer setup
    renderer = setup_renderer(agents, gridworld_only)
    canvas = renderer(domain, initial_state)
    save(joinpath(output_folder, "initial_state.png"), canvas)

    # Main simulation loop
    actions = []
    combined_score = 0
    total_gems_picked_up = 0
    num_gems_picked_up = Dict(agent => 0 for agent in agents)
    remaining_items = copy(items)
    possible_gems = collect(keys(ground_truth_rewards))
    possible_rewards = collect(values(ground_truth_rewards))
    state = initial_state

    # Initialize beliefs
    if inference_type == "pf"
        pf_states = Dict{Symbol, Union{Nothing, ParticleFilterState{Gen.DynamicDSLTrace}}}(agent => nothing for agent in agents)
    elseif inference_type == "enum"
        enum_results = Dict{Symbol, Union{Nothing, NamedTuple}}(agent => nothing for agent in agents)
    else
        error("Invalid inference type")
    end

    beliefs = Dict(agent => Dict(gem => 5.0 for gem in [:red, :blue, :yellow, :green]) for agent in agents)

    # Initialize planners
    heuristics = [VisionGemHeuristic(agent, beliefs[agent]) for agent in agents]
    planners = [RTHS(heuristic, n_iters=0, max_nodes=5) for heuristic in heuristics]

    # Initialize observations
    observations = Dict(agent => Gen.choicemap() for agent in agents)
    previous_utterances = Dict{Symbol, Union{Nothing, String}}(agent => nothing for agent in agents)

    # Main simulation loop
    t = 1
    while !isempty(remaining_items) && t <= T
        print(t)
        current_utterances = Dict{Symbol, Union{Nothing, String}}(agent => nothing for agent in agents)
        for (i, agent) in enumerate(agents)

            # Update the agents goal
            goals = PDDL.Term[]
            rewards = Float64[]
            for gem in remaining_items
                gem_obj = PDDL.Const(gem)
                color = Symbol(split(string(gem), "_")[1])
                reward = beliefs[agent][color]
                heuristics[i].rewards[color] = reward
                if reward >= 0
                    push!(goals, PDDL.pddl"(has $agent $gem_obj)")
                    push!(rewards, reward)
                end
            end

            # Solve plan, select action and transition state
            spec = MultiGoalReward(goals, rewards, 0.95)
            sol = planners[i](domain, state, spec)
            action = boltzmann_action(sol.value_policy, state, agent, 0.)
            push!(actions, action)
            state = transition(domain, state, action; check=true)

            # Set observations
            observations[agent] = Gen.choicemap()
            observations[agent][(t => :self => :gem_pickup)] = false
            item, utterance, observed_reward = "none", "none", 0
            if action.name == :pickup
                item = action.args[2].name
                remaining_items = filter(x -> x != item, remaining_items)
                gem = parse_gem(String(item))
                num_gems_picked_up[agent] += 1
                total_gems_picked_up += 1
                combined_score += ground_truth_rewards[gem]

                # Generate noisy reward observation
                observed_reward = sample_noisy_reward(ground_truth_rewards[gem], possible_rewards)

                # Update the observations for pickup
                observations[agent][(t => :self => :gem_pickup)] = true
                observations[agent][(t => :self => :gem)] = gem
                observations[agent][(t => :self => :reward_received)] = observed_reward

                # Generate utterance
                utterance_tr, _ = Gen.generate(utterance_model, (gem, observed_reward), Gen.choicemap())
                utterance = Gen.get_retval(utterance_tr)
                current_utterances[agent] = utterance
                observations[agent][(t => :self => :utterance => :output)] = utterance
            end

            # Agent observes other agents' utterances from the previous timestep
            other_agent_index = 1
            for other_agent in agents
                if other_agent != agent
                    if previous_utterances[other_agent] !== nothing
                        observations[agent][(t => :other_agents => other_agent_index => :spoke)] = true
                        observations[agent][(t => :other_agents => other_agent_index => :utterance => :output)] = previous_utterances[other_agent]
                    else
                        observations[agent][(t => :other_agents => other_agent_index => :spoke)] = false
                    end
                    other_agent_index += 1
                end
            end

            if inference_type == "pf"
                pf_states[agent] = update_beliefs_communication(pf_states[agent], t, length(agents), possible_gems, possible_rewards, observations[agent], num_particles, ess_thresh)
                gem_reward_probs = get_gem_reward_probabilities(pf_states[agent], possible_gems, possible_rewards)
            elseif inference_type == "enum" && t == 1
                enum_results[agent] = enum_inference(agent_model_communication, (t, length(agents), possible_gems, possible_rewards), observations[agent], ([(:reward, :red), (:reward, :blue), (:reward, :yellow), (:reward, :green)]), (possible_rewards,possible_rewards,possible_rewards,possible_rewards,))
                gem_reward_probs = get_enum_gem_reward_probabilities(enum_results[agent], possible_gems, possible_rewards)
            elseif inference_type == "enum"
                enum_results[agent] = enum_inference_step(enum_results[agent], (t, length(agents), possible_gems, possible_rewards), observations[agent])
                gem_reward_probs = get_enum_gem_reward_probabilities(enum_results[agent], possible_gems, possible_rewards)
            end
            utilities = calculate_gem_utility(gem_reward_probs, possible_rewards)
            beliefs[agent] = utilities
            append_to_results!(results, t, i, combined_score, total_gems_picked_up, gem_reward_probs, string(item), utterance, observed_reward)
        end
        
        previous_utterances = current_utterances
        t += 1
    end

    # Generate and save animation
    anim = anim_plan(renderer, domain, initial_state, actions; format="gif", framerate=1)
    save(joinpath(output_folder, "plan.mp4"), anim)

    return results
end

function run_simulation_no_communication_vision(
    problem_path::String,
    output_folder::String,
    ess_thresh::Float64,
    num_particles::Int,
    ground_truth_rewards::Dict{Symbol, Int},
    T::Int,
    gridworld_only::Bool = false
)

    PDDL.Arrays.register!()
    results = setup_results()

    # Load domain and problem
    domain = load_domain(joinpath(@__DIR__, "domain.pddl"))
    problem::Problem = load_problem(problem_path)
    initial_state = initstate(domain, problem)

    items = [obj.name for obj in PDDL.get_objects(domain, initial_state, :gem)]
    agents = Symbol[obj.name for obj in PDDL.get_objects(domain, initial_state, :agent)]

    # Renderer setup
    renderer = setup_renderer(agents, gridworld_only)
    canvas = renderer(domain, initial_state)
    save(joinpath(output_folder, "initial_state.png"), canvas)

    # Main simulation loop
    actions = []
    combined_score = 0
    total_gems_picked_up = 0
    num_gems_picked_up = Dict(agent => 0 for agent in agents)
    remaining_items = copy(items)
    possible_gems = collect(keys(ground_truth_rewards))
    possible_rewards = collect(values(ground_truth_rewards))
    state = initial_state

    # Initialize beliefs
    pf_states = Dict{Symbol, Union{Nothing, ParticleFilterState{Gen.DynamicDSLTrace}}}(agent => nothing for agent in agents)
    beliefs = Dict(agent => Dict(gem => 5.0 for gem in [:red, :blue, :yellow, :green]) for agent in agents)

    # Initialize planners
    heuristics = [VisionGemHeuristic(agent, beliefs[agent]) for agent in agents]
    planners = [RTHS(heuristic, n_iters=0, max_nodes=5) for heuristic in heuristics]

    # Initialize observations
    observations = Dict(agent => Gen.choicemap() for agent in agents)

    # Main simulation loop
    t = 1
    while !isempty(remaining_items) && t <= T

        for (i, agent) in enumerate(agents)

            # Update the agents goal
            goals = PDDL.Term[]
            rewards = Float64[]        
            for gem in remaining_items
                gem_obj = PDDL.Const(gem)
                color = Symbol(split(string(gem), "_")[1])
                reward = beliefs[agent][color]
                heuristics[i].rewards[color] = reward
                if reward >= 0
                    push!(goals, PDDL.pddl"(has $agent $gem_obj)")
                    push!(rewards, reward)
                end
            end

            # Solve plan, select action and transition state
            spec = MultiGoalReward(goals, rewards, 0.95)
            sol = planners[i](domain, state, spec)
            action = boltzmann_action(sol.value_policy, state, agent, 0.)
            push!(actions, action)
            state = transition(domain, state, action; check=true)

            # Set observations
            observations[agent] = Gen.choicemap()
            observations[agent][(t => :self => :gem_pickup)] = false
            item, utterance, observed_reward = "none", "none", 0
            if action.name == :pickup
                item = action.args[2].name
                remaining_items = filter(x -> x != item, remaining_items)
                gem = parse_gem(String(item))
                num_gems_picked_up[agent] += 1
                total_gems_picked_up += 1
                combined_score += ground_truth_rewards[gem]

                observed_reward = sample_noisy_reward(ground_truth_rewards[gem], possible_rewards)

                # Update the observations for pickup
                observations[agent][(t => :self => :gem_pickup)] = true
                observations[agent][(t => :self => :gem)] = gem
                observations[agent][(t => :self => :reward_received)] = observed_reward
            end

            # Run particle filter and update beliefs
            pf_states[agent] = update_beliefs_no_communication(pf_states[agent], t, length(agents), possible_gems, possible_rewards, observations[agent], num_particles, ess_thresh)
            gem_reward_probs = get_gem_reward_probabilities(pf_states[agent], possible_gems, possible_rewards)
            utilities = calculate_gem_utility(gem_reward_probs, possible_rewards)
            beliefs[agent] = utilities

            append_to_results!(results, t, i, combined_score, total_gems_picked_up, gem_reward_probs, string(item), utterance, observed_reward)
        end

        t += 1
    end

    # Generate and save animation
    anim = anim_plan(renderer, domain, initial_state, actions; format="gif", framerate=2)
    save(joinpath(output_folder, "plan.mp4"), anim)

    return results
end

function run_simulation_communication_restricted_vision(
    problem_path::String,
    output_folder::String,
    ess_thresh::Float64,
    num_particles::Int,
    ground_truth_rewards::Dict{Symbol, Int},
    T::Int,
    gridworld_only::Bool = false
)
    results = setup_results()

    # Load domain and problem
    domain = load_domain(joinpath(@__DIR__, "domain.pddl"))
    problem::Problem = load_problem(problem_path)
    initial_state = initstate(domain, problem)

    items = [obj.name for obj in PDDL.get_objects(domain, initial_state, :gem)]
    agents = Symbol[obj.name for obj in PDDL.get_objects(domain, initial_state, :agent)]

    # Renderer setup
    renderer = setup_renderer(agents, gridworld_only)
    canvas = renderer(domain, initial_state)
    save(joinpath(output_folder, "initial_state.png"), canvas)

    # Main simulation loop
    actions = []
    combined_score = 0
    total_gems_picked_up = 0
    num_gems_picked_up = Dict(agent => 0 for agent in agents)
    remaining_items = copy(items)
    possible_gems = collect(keys(ground_truth_rewards))
    possible_rewards = collect(values(ground_truth_rewards))
    state = initial_state

    # Initialize beliefs and heuristics optimistically
    pf_states = Dict{Symbol, Union{Nothing, ParticleFilterState{Gen.DynamicDSLTrace}}}(agent => nothing for agent in agents)
    beliefs = Dict(agent => Dict(gem => 5.0 for gem in [:red, :blue, :yellow, :green]) for agent in agents)

    # Initialize planners
    heuristics = [RestrictedVisionGemHeuristic(agent, beliefs[agent]) for agent in agents]
    planners = [RTHS(heuristic, n_iters=0, max_nodes=2) for heuristic in heuristics]
    
    # Initialize observations for each agent
    observations = Dict(agent => Gen.choicemap() for agent in agents)
    previous_utterances = Dict{Symbol, Union{Nothing, String}}(agent => nothing for agent in agents)

    # Main simulation loop
    t = 1
    while !isempty(remaining_items) && t <= T
        current_utterances = Dict{Symbol, Union{Nothing, String}}(agent => nothing for agent in agents)
        for (i, agent) in enumerate(agents)

            # Update the agents goal
            goals = PDDL.Term[]
            rewards = Float64[]        
            for gem in remaining_items
                gem_obj = PDDL.Const(gem)
                color = Symbol(split(string(gem), "_")[1])
                reward = beliefs[agent][color]
                heuristics[i].rewards[color] = reward

                if reward >= 0
                    push!(goals, PDDL.pddl"(has $agent $gem_obj)")
                    push!(rewards, reward)
                end
            end

            # Solve plan, select action and transition state
            spec = MultiGoalReward(goals, rewards, 0.95)
            sol = planners[i](domain, state, spec)
            action = boltzmann_action(sol.value_policy, state, agent, 0.)
            push!(actions, action)
            state = transition(domain, state, action; check=true)

            # Set observations
            observations[agent] = Gen.choicemap()
            observations[agent][(t => :self => :gem_pickup)] = false
            item, utterance, observed_reward = "none", "none", 0
            if action.name == :pickup
                item = action.args[2].name
                remaining_items = filter(x -> x != item, remaining_items)
                gem = parse_gem(String(item))
                num_gems_picked_up[agent] += 1
                total_gems_picked_up += 1
                combined_score += ground_truth_rewards[gem]

                # Generate noisy reward observation
                observed_reward = sample_noisy_reward(ground_truth_rewards[gem], possible_rewards)

                # Update the observations for pickup
                observations[agent][(t => :self => :gem_pickup)] = true
                observations[agent][(t => :self => :gem)] = gem
                observations[agent][(t => :self => :reward_received)] = observed_reward

                # Generate utterance
                utterance_tr, _ = Gen.generate(utterance_model, (gem, observed_reward), Gen.choicemap())
                utterance = Gen.get_retval(utterance_tr)
                current_utterances[agent] = utterance
                observations[agent][(t => :self => :utterance => :output)] = utterance
            end

            # Add other agents' utterances from the previous timestep to the observations
            other_agent_index = 1
            for other_agent in agents
                if other_agent != agent
                    if previous_utterances[other_agent] !== nothing
                        observations[agent][(t => :other_agents => other_agent_index => :spoke)] = true
                        observations[agent][(t => :other_agents => other_agent_index => :utterance => :output)] = previous_utterances[other_agent]
                    else
                        observations[agent][(t => :other_agents => other_agent_index => :spoke)] = false
                    end
                    other_agent_index += 1
                end
            end

            # Update beliefs for the current agent using only the most recent observation
            pf_states[agent] = update_beliefs_communication(pf_states[agent], t, length(agents), possible_gems, possible_rewards, observations[agent], num_particles, ess_thresh)
            gem_reward_probs = get_gem_reward_probabilities(pf_states[agent], possible_gems, possible_rewards)
            utilities = calculate_gem_utility(gem_reward_probs, possible_rewards)
            beliefs[agent] = utilities

            append_to_results!(results, t, i, combined_score, total_gems_picked_up, gem_reward_probs, string(item), utterance, observed_reward)
        end

        previous_utterances = current_utterances
        t += 1
    end

    # Generate and save animation
    anim = anim_plan(renderer, domain, initial_state, actions; format="gif", framerate=2)
    save(joinpath(output_folder, "plan.mp4"), anim)

    return results
end

function run_simulation_communication_perfect_vision(
    problem_path::String,
    output_folder::String,
    ess_thresh::Float64,
    num_particles::Int,
    ground_truth_rewards::Dict{Symbol, Int},
    T::Int,
    gridworld_only::Bool = false
)
    results = setup_results()

    # Load domain and problem
    domain = load_domain(joinpath(@__DIR__, "domain.pddl"))
    problem::Problem = load_problem(problem_path)
    initial_state = initstate(domain, problem)

    items = [obj.name for obj in PDDL.get_objects(domain, initial_state, :gem)]
    agents = Symbol[obj.name for obj in PDDL.get_objects(domain, initial_state, :agent)]

    # Renderer setup
    renderer = setup_renderer(agents, gridworld_only)
    canvas = renderer(domain, initial_state)
    save(joinpath(output_folder, "initial_state.png"), canvas)

    # Main simulation loop
    actions = []
    combined_score = 0
    total_gems_picked_up = 0
    num_gems_picked_up = Dict(agent => 0 for agent in agents)
    remaining_items = copy(items)
    possible_gems = collect(keys(ground_truth_rewards))
    possible_rewards = collect(values(ground_truth_rewards))
    state = initial_state

    # Initialize beliefs
    pf_states = Dict{Symbol, Union{Nothing, ParticleFilterState{Gen.DynamicDSLTrace}}}(agent => nothing for agent in agents)
    beliefs = Dict(agent => Dict(gem => 5.0 for gem in [:red, :blue, :yellow, :green]) for agent in agents)

    # Initialize heuristics
    heuristics = [GoalManhattan(agent) for agent in agents]
    planners = [AStarPlanner(heuristic) for heuristic in heuristics]
    
    # Initialize observations
    observations = Dict(agent => Gen.choicemap() for agent in agents)
    previous_utterances = Dict{Symbol, Union{Nothing, String}}(agent => nothing for agent in agents)

    # Main simulation loop
    t = 1
    while !isempty(remaining_items) && t <= T
        current_utterances = Dict{Symbol, Union{Nothing, String}}(agent => nothing for agent in agents)
        for (i, agent) in enumerate(agents)

            # Update the agents goal
            closest_gem = nothing
            closest_distance = Inf
            closest_reward = 0.0
            for gem in remaining_items
                gem_obj = PDDL.Const(gem)
                color = Symbol(split(string(gem), "_")[1])
                reward = beliefs[agent][color]

                if reward >= 0
                    agent_x, agent_y = get_agent_pos(state, agent)
                    gem_x = state[Compound(:xloc, [gem_obj])]
                    gem_y = state[Compound(:yloc, [gem_obj])]
                    distance = sqrt((agent_x - gem_x)^2 + (agent_y - gem_y)^2)
                                        
                    if distance < closest_distance
                        closest_gem = gem_obj
                        closest_distance = distance
                        closest_reward = reward
                    end
                end
            end

            if closest_gem !== nothing
                goal = PDDL.pddl"(has $agent $closest_gem)"

                # Solve plan, select action and transition state
                spec = MultiGoalReward([goal], [closest_reward], 0.95)
                sol = planners[i](domain, state, spec)
                action = collect(sol)[1]
                push!(actions, action)
                state = transition(domain, state, action; check=true)

                # Set observations
                observations[agent] = Gen.choicemap()
                observations[agent][(t => :self => :gem_pickup)] = false
                item, utterance, observed_reward = "none", "none", 0
                if action.name == :pickup
                    item = action.args[2].name
                    remaining_items = filter(x -> x != item, remaining_items)
                    gem = parse_gem(String(item))
                    num_gems_picked_up[agent] += 1
                    total_gems_picked_up += 1
                    combined_score += ground_truth_rewards[gem]

                    # Generate noisy reward observation
                    observed_reward = sample_noisy_reward(ground_truth_rewards[gem], possible_rewards)

                    # Update the observations for pickup
                    observations[agent][(t => :self => :gem_pickup)] = true
                    observations[agent][(t => :self => :gem)] = gem
                    observations[agent][(t => :self => :reward_received)] = observed_reward

                    # Generate utterance
                    utterance_tr, _ = Gen.generate(utterance_model, (gem, observed_reward), Gen.choicemap())
                    utterance = Gen.get_retval(utterance_tr)
                    current_utterances[agent] = utterance
                    observations[agent][(t => :self => :utterance => :output)] = utterance
                end

                # Add other agents' utterances from the previous timestep to the observations
                other_agent_index = 1
                for other_agent in agents
                    if other_agent != agent
                        if previous_utterances[other_agent] !== nothing
                            observations[agent][(t => :other_agents => other_agent_index => :spoke)] = true
                            observations[agent][(t => :other_agents => other_agent_index => :utterance => :output)] = previous_utterances[other_agent]
                        else
                            observations[agent][(t => :other_agents => other_agent_index => :spoke)] = false
                        end
                        other_agent_index += 1
                    end
                end

                # Run particle filter and update beliefs
                pf_states[agent] = update_beliefs_communication(pf_states[agent], t, length(agents), possible_gems, possible_rewards, observations[agent], num_particles, ess_thresh)
                gem_reward_probs = get_gem_reward_probabilities(pf_states[agent], possible_gems, possible_rewards)
                utilities = calculate_gem_utility(gem_reward_probs, possible_rewards)
                beliefs[agent] = utilities

                append_to_results!(results, t, i, combined_score, total_gems_picked_up, gem_reward_probs, string(item), utterance, observed_reward)
            else
                @warn "No gems with non-negative reward found for $agent"
                gem_reward_probs = get_gem_reward_probabilities(pf_states[agent], possible_gems, possible_rewards)
                append_to_results!(results, t, i, combined_score, total_gems_picked_up, gem_reward_probs, "none", "none", 0)
            end

        end

        previous_utterances = current_utterances
        t += 1
    end

    # Generate and save animation
    anim = anim_plan(renderer, domain, initial_state, actions; format="gif", framerate=2)
    save(joinpath(output_folder, "plan.mp4"), anim)

    return results
end

function run_simulation_gpt4o(
    problem_path::String,
    output_folder::String,
    ground_truth_rewards::Dict{Symbol, Int},
    T::Int,
    gridworld_only::Bool = false
)
    results = setup_results_gpt4o()

    # Load domain and problem
    domain = load_domain(joinpath(@__DIR__, "domain.pddl"))
    problem::Problem = load_problem(problem_path)
    initial_state = initstate(domain, problem)

    items = [obj.name for obj in PDDL.get_objects(domain, initial_state, :gem)]
    agents = Symbol[obj.name for obj in PDDL.get_objects(domain, initial_state, :agent)]

    # Renderer setup
    renderer = setup_renderer(agents, gridworld_only)
    canvas = renderer(domain, initial_state)
    save(joinpath(output_folder, "initial_state.png"), canvas)

    # Main simulation loop
    actions = []
    combined_score = 0
    total_gems_picked_up = 0
    num_gems_picked_up = Dict(agent => 0 for agent in agents)
    remaining_items = copy(items)
    possible_gems = collect(keys(ground_truth_rewards))
    possible_rewards = collect(values(ground_truth_rewards))
    state = initial_state

    # Initialize beliefs
    gem_string = join(possible_gems, ", ")
    reward_string = join(possible_rewards, ", ")
    gpt4o_context = Dict{Symbol, String}(agent => "Please determine the reward associated with each gem given the following observations observed from traversing the gridworld environment. The possible gems are $gem_string. The possible rewards are $reward_string. Please responds with only the determined gem rewards (for each gem type), using the format <color>:<reward> separated by commas. If you are very uncertain of a reward for a gem, set the reward to 1.0." for agent in agents)
    beliefs = Dict(agent => Dict(gem => 5.0 for gem in [:red, :blue, :yellow, :green]) for agent in agents)

    # Initialize planners
    heuristics = [VisionGemHeuristic(agent, beliefs[agent]) for agent in agents]
    planners = [RTHS(heuristic, n_iters=0, max_nodes=5) for heuristic in heuristics]
    
    # Initialize observations for each agent
    observations = Dict(agent => Gen.choicemap() for agent in agents)
    previous_utterances = Dict{Symbol, Union{Nothing, String}}(agent => nothing for agent in agents)

    # Main simulation loop
    t = 1
    while !isempty(remaining_items) && t <= T
        current_utterances = Dict{Symbol, Union{Nothing, String}}(agent => nothing for agent in agents)
        for (i, agent) in enumerate(agents)

            gpt4o_context[agent] *= "\n timestep t = $t"

            goals = PDDL.Term[]
            rewards = Float64[]        
            for gem in remaining_items
                gem_obj = PDDL.Const(gem)
                color = Symbol(split(string(gem), "_")[1])
                reward = beliefs[agent][color]
                heuristics[i].rewards[color] = reward

                if reward >= 0
                    push!(goals, PDDL.pddl"(has $agent $gem_obj)")
                    push!(rewards, reward)
                end
            end

            # Solve plan, select action and transition state
            spec = MultiGoalReward(goals, rewards, 0.95)
            sol = planners[i](domain, state, spec)
            action = boltzmann_action(sol.value_policy, state, agent, 0.)
            push!(actions, action)
            state = transition(domain, state, action; check=true)

            # Clear previous observations and create new observation for this timestep
            observations[agent] = Gen.choicemap()
            observations[agent][(t => :self => :gem_pickup)] = false
            item, utterance, observed_reward = "none", "none", 0
            if action.name == :pickup
                item = action.args[2].name
                remaining_items = filter(x -> x != item, remaining_items)
                gem = parse_gem(String(item))
                num_gems_picked_up[agent] += 1
                total_gems_picked_up += 1
                combined_score += ground_truth_rewards[gem]

                # Generate noisy reward observation
                observed_reward = sample_noisy_reward(ground_truth_rewards[gem], possible_rewards)


                # Update the observations for pickup
                gpt4o_context[agent] *= "\n self gem pickup = true"
                gpt4o_context[agent] *= "\n self gem type = $gem"
                gpt4o_context[agent] *= "\n self gem reward = $observed_reward"

                # Generate utterance
                utterance_tr, _ = Gen.generate(utterance_model, (gem, observed_reward), Gen.choicemap())
                utterance = Gen.get_retval(utterance_tr)
                current_utterances[agent] = utterance
                gpt4o_context[agent] *= "\n self utterance = $utterance"
            else
                gpt4o_context[agent] *= "\n self gem pickup = false"
            end

            # Add other agents' utterances from the previous timestep to the observations
            other_agent_index = 1
            for other_agent in agents
                if other_agent != agent
                    if previous_utterances[other_agent] !== nothing
                        gpt4o_context[agent] *= "\n other agent spoke = true"
                        gpt4o_context[agent] = gpt4o_context[agent] * "\n other agent utterance = $(previous_utterances[other_agent])"
                    else
                        gpt4o_context[agent] *= "\n other agent spoke = false"
                    end
                    other_agent_index += 1
                end
            end

            beliefs[agent] = parse_belief(gpt4o(gpt4o_context[agent]))

            append_to_results_gpt4o!(results, t, i, combined_score, total_gems_picked_up, beliefs[agent], string(item), utterance, observed_reward)
        end

        previous_utterances = current_utterances
        t += 1
    end

    # Generate and save animation
    anim = anim_plan(renderer, domain, initial_state, actions; format="gif", framerate=2)
    save(joinpath(output_folder, "plan.mp4"), anim)

    return results
end

end