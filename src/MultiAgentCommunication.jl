module MultiAgentCommunication

using PDDL, PlanningDomains, SymbolicPlanners
using Gen, GenParticleFilters
using PDDLViz, GLMakie
using Random
using SymbolicPlanners: get_value, get_goal_terms
using DotEnv
include("agent.jl")
include("utils.jl")
include("heuristics.jl")
include("inference.jl")

export run_simulation_communication_vision, run_simulation_no_communication_vision, run_simulation_communication_restricted_vision, run_simulation_communication_perfect_vision, run_simulation_gpt4o

function run_simulation_communication_vision(
    problem_name::String,
    ess_thresh::Float64,
    num_particles::Int,
    ground_truth_rewards::Dict{Symbol, Int},
    T::Int,
    output_dir::String,
    gridworld_only::Bool = false
)
    task = "communication_vision"
    output_folder = joinpath(output_dir, task, problem_name)
    mkpath(output_folder)
    io = setup_logging(output_folder, "simulation_log.txt")

    PDDL.Arrays.register!()

    # Load domain and problem
    domain = load_domain(joinpath(@__DIR__, "domain.pddl"))
    problem::Problem = load_problem(joinpath(@__DIR__, "problems", problem_name * ".pddl"))
    initial_state = initstate(domain, problem)
    
    # Retrieve objects before compilation
    items = [obj.name for obj in PDDL.get_objects(domain, initial_state, :gem)]
    agents = Symbol[obj.name for obj in PDDL.get_objects(domain, initial_state, :agent)]
    
    # Renderer setup
    renderer = setup_renderer(agents, gridworld_only)
    canvas = renderer(domain, initial_state)

    # Output setup
    
    io = setup_logging(output_folder, "simulation_log.txt")
    save(joinpath(output_folder, "initial_state.png"), canvas)

    # Main simulation loop
    actions = []
    combined_score = 0
    remaining_items = copy(items)
    possible_gems = collect(keys(ground_truth_rewards))
    possible_rewards = collect(values(ground_truth_rewards))
    num_gems_picked_up = Dict(agent => 0 for agent in agents)
    total_gems_picked_up = 0
    state = initial_state

    # Initialize beliefs and heuristics optimistically
    pf_states = Dict{Symbol, Union{Nothing, ParticleFilterState{Gen.DynamicDSLTrace}}}(agent => nothing for agent in agents)
    beliefs = Dict(agent => Dict(gem => 5.0 for gem in [:red, :blue, :yellow, :green]) for agent in agents)
    heuristics = [VisionGemHeuristic(agent, beliefs[agent]) for agent in agents]

    # Initialize planners
    planners = [RTHS(heuristic, n_iters=0, max_nodes=5) for heuristic in heuristics]
    
    # Initialize observations for each agent
    observations = Dict(agent => Gen.choicemap() for agent in agents)

    # Initialize previous utterances
    previous_utterances = Dict{Symbol, Union{Nothing, String}}(agent => nothing for agent in agents)

    # Main simulation loop
    t = 1
    while !isempty(remaining_items) && t <= T
        @info "Step $t:"
        current_utterances = Dict{Symbol, Union{Nothing, String}}(agent => nothing for agent in agents)
        for (i, agent) in enumerate(agents)
            goals = PDDL.Term[]
            rewards = Float64[]        
            current_beliefs = beliefs[agent]
            for gem in remaining_items
                gem_obj = PDDL.Const(gem)
                color = Symbol(split(string(gem), "_")[1])
                reward = current_beliefs[color]
                heuristics[i].rewards[color] = reward

                if reward >= 0
                    push!(goals, PDDL.pddl"(has $agent $gem_obj)")
                    push!(rewards, reward)
                end
            end

            spec = MultiGoalReward(goals, rewards, 0.95)
            sol = planners[i](domain, state, spec)
            action = boltzmann_action(sol.value_policy, state, agent, 0.)
            state = transition(domain, state, action; check=true)

            # Clear previous observations and create new observation for this timestep
            observations[agent] = Gen.choicemap()
            observations[agent][(t => :self => :gem_pickup)] = false

            if action.name == :pickup
                item = action.args[2].name
                @info "       $agent picked up $item."
                remaining_items = filter(x -> x != item, remaining_items)
                gem = parse_gem(String(item))
                num_gems_picked_up[agent] += 1
                total_gems_picked_up += 1
                reward = ground_truth_rewards[gem]
                combined_score += reward
                @info "       $agent received $reward score."
                @info "       Combined score is now $combined_score."

                # Update the observations for pickup
                observations[agent][(t => :self => :gem_pickup)] = true
                observations[agent][(t => :self => :gem)] = gem
                observations[agent][(t => :self => :reward_received)] = reward

                # Generate utterance
                utterance_tr, _ = Gen.generate(utterance_model, (gem, reward), Gen.choicemap())
                utterance = Gen.get_retval(utterance_tr)
                current_utterances[agent] = utterance
                observations[agent][(t => :self => :utterance => :output)] = utterance
                @info "       $agent communicated: $utterance"
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

            # Calculate and update gem utilities for the current agent
            current_pf_state = pf_states[agent]
            top_rewards = get_top_weighted_rewards(current_pf_state, 10, possible_gems)
            gem_certainty = quantify_gem_certainty(top_rewards)
            utilities, certainties = calculate_gem_utility(gem_certainty)
            beliefs[agent] = utilities

            print_estimated_rewards(agent, beliefs[agent], certainties)

            push!(actions, action)
        end

        # Update previous_utterances for the next timestep
        previous_utterances = current_utterances

        t += 1
    end

    @info "Time: $t"
    @info "Total gems picked up: $total_gems_picked_up"
    @info "Final score: $combined_score"

    # Generate and save animation
    anim = anim_plan(renderer, domain, initial_state, actions; format="gif", framerate=2)
    save(joinpath(output_folder, "plan.mp4"), anim)

    close(io)

    return combined_score
end

function run_simulation_no_communication_vision(
    problem_name::String,
    ess_thresh::Float64,
    num_particles::Int,
    ground_truth_rewards::Dict{Symbol, Int},
    T::Int,
    output_dir::String,
    gridworld_only::Bool = false
)
    task = "no_communication_vision"
    output_folder = joinpath(output_dir, task, problem_name)
    mkpath(output_folder)
    io = setup_logging(output_folder, "simulation_log.txt")

    PDDL.Arrays.register!()

    # Load domain and problem
    domain = load_domain(joinpath(@__DIR__, "domain.pddl"))
    problem::Problem = load_problem(joinpath(@__DIR__, "problems", problem_name * ".pddl"))
    initial_state = initstate(domain, problem)

    # Retrieve objects before compilation
    items = [obj.name for obj in PDDL.get_objects(domain, initial_state, :gem)]
    agents = Symbol[obj.name for obj in PDDL.get_objects(domain, initial_state, :agent)]

    # Renderer setup
    renderer = setup_renderer(agents, gridworld_only)
    canvas = renderer(domain, initial_state)
    save(joinpath(output_folder, "initial_state.png"), canvas)

    # Main simulation loop
    actions = []
    combined_score = 0
    remaining_items = copy(items)
    possible_gems = collect(keys(ground_truth_rewards))
    possible_rewards = collect(values(ground_truth_rewards))
    num_gems_picked_up = Dict(agent => 0 for agent in agents)
    total_gems_picked_up = 0
    state = initial_state

    # Initialize beliefs and heuristics optimistically
    pf_states = Dict{Symbol, Union{Nothing, ParticleFilterState{Gen.DynamicDSLTrace}}}(agent => nothing for agent in agents)
    beliefs = Dict(agent => Dict(gem => 5.0 for gem in [:red, :blue, :yellow, :green]) for agent in agents)
    heuristics = [VisionGemHeuristic(agent, beliefs[agent]) for agent in agents]

    # Initialize planners
    planners = [RTHS(heuristic, n_iters=0, max_nodes=5) for heuristic in heuristics]
    
    # Initialize observations for each agent
    observations = Dict(agent => Gen.choicemap() for agent in agents)

    # Initialize previous utterances
    previous_utterances = Dict{Symbol, Union{Nothing, String}}(agent => nothing for agent in agents)

    # Main simulation loop
    t = 1
    while !isempty(remaining_items) && t <= T
        @info "Step $t:"
        current_utterances = Dict{Symbol, Union{Nothing, String}}(agent => nothing for agent in agents)
        for (i, agent) in enumerate(agents)
            goals = PDDL.Term[]
            rewards = Float64[]        
            current_beliefs = beliefs[agent]
            for gem in remaining_items
                gem_obj = PDDL.Const(gem)
                color = Symbol(split(string(gem), "_")[1])
                reward = current_beliefs[color]
                heuristics[i].rewards[color] = reward

                if reward >= 0
                    push!(goals, PDDL.pddl"(has $agent $gem_obj)")
                    push!(rewards, reward)
                end
            end

            spec = MultiGoalReward(goals, rewards, 0.95)
            sol = planners[i](domain, state, spec)
            action = boltzmann_action(sol.value_policy, state, agent, 0.)
            state = transition(domain, state, action; check=true)

            # Clear previous observations and create new observation for this timestep
            observations[agent] = Gen.choicemap()
            observations[agent][(t => :self => :gem_pickup)] = false

            if action.name == :pickup
                item = action.args[2].name
                @info "       $agent picked up $item."
                remaining_items = filter(x -> x != item, remaining_items)
                gem = parse_gem(String(item))
                num_gems_picked_up[agent] += 1
                total_gems_picked_up += 1
                reward = ground_truth_rewards[gem]
                combined_score += reward
                @info "       $agent received $reward score."
                @info "       Combined score is now $combined_score."

                # Update the observations for pickup
                observations[agent][(t => :self => :gem_pickup)] = true
                observations[agent][(t => :self => :gem)] = gem
                observations[agent][(t => :self => :reward_received)] = reward
            end

            # Update beliefs for the current agent using only the most recent observation
            pf_states[agent] = update_beliefs_no_communication(pf_states[agent], t, length(agents), possible_gems, possible_rewards, observations[agent], num_particles, ess_thresh)

            # Calculate and update gem utilities for the current agent
            current_pf_state = pf_states[agent]
            top_rewards = get_top_weighted_rewards(current_pf_state, 10, possible_gems)
            gem_certainty = quantify_gem_certainty(top_rewards)
            utilities, certainties = calculate_gem_utility(gem_certainty)
            beliefs[agent] = utilities

            print_estimated_rewards(agent, beliefs[agent], certainties)

            push!(actions, action)
        end

        # Update previous_utterances for the next timestep
        previous_utterances = current_utterances

        t += 1
    end

    @info "Time: $t"
    @info "Total gems picked up: $total_gems_picked_up"
    @info "Final score: $combined_score"

    # Generate and save animation
    anim = anim_plan(renderer, domain, initial_state, actions; format="gif", framerate=2)
    save(joinpath(output_folder, "plan.mp4"), anim)

    close(io)

    return combined_score
end

function run_simulation_communication_restricted_vision(
    problem_name::String,
    ess_thresh::Float64,
    num_particles::Int,
    ground_truth_rewards::Dict{Symbol, Int},
    T::Int,
    output_dir::String,
    gridworld_only::Bool = false
)
    task = "communication_restricted_vision"
    output_folder = joinpath(output_dir, task, problem_name)
    mkpath(output_folder)
    io = setup_logging(output_folder, "simulation_log.txt")

    PDDL.Arrays.register!()

    # Load domain and problem
    domain = load_domain(joinpath(@__DIR__, "domain.pddl"))
    problem::Problem = load_problem(joinpath(@__DIR__, "problems", problem_name * ".pddl"))
    initial_state = initstate(domain, problem)

    # Retrieve objects before compilation
    items = [obj.name for obj in PDDL.get_objects(domain, initial_state, :gem)]
    agents = Symbol[obj.name for obj in PDDL.get_objects(domain, initial_state, :agent)]

    # Renderer setup
    renderer = setup_renderer(agents, gridworld_only)
    canvas = renderer(domain, initial_state)
    save(joinpath(output_folder, "initial_state.png"), canvas)

    # Main simulation loop
    actions = []
    combined_score = 0
    remaining_items = copy(items)
    possible_gems = collect(keys(ground_truth_rewards))
    possible_rewards = collect(values(ground_truth_rewards))
    num_gems_picked_up = Dict(agent => 0 for agent in agents)
    total_gems_picked_up = 0
    state = initial_state

    # Initialize beliefs and heuristics optimistically
    pf_states = Dict{Symbol, Union{Nothing, ParticleFilterState{Gen.DynamicDSLTrace}}}(agent => nothing for agent in agents)
    beliefs = Dict(agent => Dict(gem => 5.0 for gem in [:red, :blue, :yellow, :green]) for agent in agents)
    heuristics = [ShortSightedVisionGemHeuristic(agent, beliefs[agent]) for agent in agents]

    # Initialize planners
    planners = [RTHS(heuristic, n_iters=0, max_nodes=2) for heuristic in heuristics]
    
    # Initialize observations for each agent
    observations = Dict(agent => Gen.choicemap() for agent in agents)

    # Initialize previous utterances
    previous_utterances = Dict{Symbol, Union{Nothing, String}}(agent => nothing for agent in agents)

    # Main simulation loop
    t = 1
    while !isempty(remaining_items) && t <= T
        @info "Step $t:"
        current_utterances = Dict{Symbol, Union{Nothing, String}}(agent => nothing for agent in agents)
        for (i, agent) in enumerate(agents)
            goals = PDDL.Term[]
            rewards = Float64[]        
            current_beliefs = beliefs[agent]
            for gem in remaining_items
                gem_obj = PDDL.Const(gem)
                color = Symbol(split(string(gem), "_")[1])
                reward = current_beliefs[color]
                heuristics[i].rewards[color] = reward

                if reward >= 0
                    push!(goals, PDDL.pddl"(has $agent $gem_obj)")
                    push!(rewards, reward)
                end
            end

            spec = MultiGoalReward(goals, rewards, 0.95)
            sol = planners[i](domain, state, spec)
            action = boltzmann_action(sol.value_policy, state, agent, 0.)
            state = transition(domain, state, action; check=true)

            # Clear previous observations and create new observation for this timestep
            observations[agent] = Gen.choicemap()
            observations[agent][(t => :self => :gem_pickup)] = false

            if action.name == :pickup
                item = action.args[2].name
                @info "       $agent picked up $item."
                remaining_items = filter(x -> x != item, remaining_items)
                gem = parse_gem(String(item))
                num_gems_picked_up[agent] += 1
                total_gems_picked_up += 1
                reward = ground_truth_rewards[gem]
                combined_score += reward
                @info "       $agent received $reward score."
                @info "       Combined score is now $combined_score."

                # Update the observations for pickup
                observations[agent][(t => :self => :gem_pickup)] = true
                observations[agent][(t => :self => :gem)] = gem
                observations[agent][(t => :self => :reward_received)] = reward

                # Generate utterance
                utterance_tr, _ = Gen.generate(utterance_model, (gem, reward), Gen.choicemap())
                utterance = Gen.get_retval(utterance_tr)
                current_utterances[agent] = utterance
                observations[agent][(t => :self => :utterance => :output)] = utterance
                @info "       $agent communicated: $utterance"
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

            # Calculate and update gem utilities for the current agent
            current_pf_state = pf_states[agent]
            top_rewards = get_top_weighted_rewards(current_pf_state, 10, possible_gems)
            gem_certainty = quantify_gem_certainty(top_rewards)
            utilities, certainties = calculate_gem_utility(gem_certainty)
            beliefs[agent] = utilities

            print_estimated_rewards(agent, beliefs[agent], certainties)

            push!(actions, action)
        end

        # Update previous_utterances for the next timestep
        previous_utterances = current_utterances

        t += 1
    end

    @info "Time: $t"
    @info "Total gems picked up: $total_gems_picked_up"
    @info "Final score: $combined_score"

    # Generate and save animation
    anim = anim_plan(renderer, domain, initial_state, actions; format="gif", framerate=2)
    save(joinpath(output_folder, "plan.mp4"), anim)

    close(io)

    return combined_score
end

function run_simulation_communication_perfect_vision(
    problem_name::String,
    ess_thresh::Float64,
    num_particles::Int,
    ground_truth_rewards::Dict{Symbol, Int},
    T::Int,
    output_dir::String,
    gridworld_only::Bool = false
)
    task = "communication_perfect_vision"
    output_folder = joinpath(output_dir, task, problem_name)
    mkpath(output_folder)
    io = setup_logging(output_folder, "simulation_log.txt")

    PDDL.Arrays.register!()

    # Load domain and problem
    domain = load_domain(joinpath(@__DIR__, "domain.pddl"))
    problem::Problem = load_problem(joinpath(@__DIR__, "problems", problem_name * ".pddl"))
    initial_state = initstate(domain, problem)

    # Retrieve objects before compilation
    items = [obj.name for obj in PDDL.get_objects(domain, initial_state, :gem)]
    agents = Symbol[obj.name for obj in PDDL.get_objects(domain, initial_state, :agent)]

    # Renderer setup
    renderer = setup_renderer(agents, gridworld_only)
    canvas = renderer(domain, initial_state)
    save(joinpath(output_folder, "initial_state.png"), canvas)

    # Main simulation loop
    actions = []
    combined_score = 0
    remaining_items = copy(items)
    possible_gems = collect(keys(ground_truth_rewards))
    possible_rewards = collect(values(ground_truth_rewards))
    num_gems_picked_up = Dict(agent => 0 for agent in agents)
    total_gems_picked_up = 0
    state = initial_state

    # Initialize beliefs and heuristics optimistically
    pf_states = Dict{Symbol, Union{Nothing, ParticleFilterState{Gen.DynamicDSLTrace}}}(agent => nothing for agent in agents)
    beliefs = Dict(agent => Dict(gem => 5.0 for gem in [:red, :blue, :yellow, :green]) for agent in agents)

    # Initialize heuristics
    heuristics = [GoalManhattan(agent) for agent in agents]

    # Initialize planners
    planners = [AStarPlanner(heuristic) for heuristic in heuristics]
    
    # Initialize observations for each agent
    observations = Dict(agent => Gen.choicemap() for agent in agents)

    # Initialize previous utterances
    previous_utterances = Dict{Symbol, Union{Nothing, String}}(agent => nothing for agent in agents)

    # Main simulation loop
    t = 1
    while !isempty(remaining_items) && t <= T

        @info "Step $t:"
        current_utterances = Dict{Symbol, Union{Nothing, String}}(agent => nothing for agent in agents)
        for (i, agent) in enumerate(agents)

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
                spec = MultiGoalReward([goal], [closest_reward], 0.95)
                sol = planners[i](domain, state, spec)
                action = collect(sol)[1]
                push!(actions, action)
                state = transition(domain, state, action; check=true)

                # Clear previous observations and create new observation for this timestep
                observations[agent] = Gen.choicemap()
                observations[agent][(t => :self => :gem_pickup)] = false

                if action.name == :pickup
                    item = action.args[2].name
                    @info "       $agent picked up $item."
                    remaining_items = filter(x -> x != item, remaining_items)
                    gem = parse_gem(String(item))
                    num_gems_picked_up[agent] += 1
                    total_gems_picked_up += 1
                    reward = ground_truth_rewards[gem]
                    combined_score += reward
                    @info "       $agent received $reward score."
                    @info "       Combined score is now $combined_score."

                    # Update the observations for pickup
                    observations[agent][(t => :self => :gem_pickup)] = true
                    observations[agent][(t => :self => :gem)] = gem
                    observations[agent][(t => :self => :reward_received)] = reward

                    # Generate utterance
                    utterance_tr, _ = Gen.generate(utterance_model, (gem, reward), Gen.choicemap())
                    utterance = Gen.get_retval(utterance_tr)
                    current_utterances[agent] = utterance
                    observations[agent][(t => :self => :utterance => :output)] = utterance
                    @info "       $agent communicated: $utterance"
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

                # Calculate and update gem utilities for the current agent
                current_pf_state = pf_states[agent]
                top_rewards = get_top_weighted_rewards(current_pf_state, 10, possible_gems)
                gem_certainty = quantify_gem_certainty(top_rewards)
                utilities, certainties = calculate_gem_utility(gem_certainty)
                beliefs[agent] = utilities

                print_estimated_rewards(agent, beliefs[agent], certainties)
            else
                @warn "No gems with non-negative reward found for $agent"
            end

        end

        # Update previous_utterances for the next timestep
        previous_utterances = current_utterances

        t += 1
    end

    @info "Time: $t"
    @info "Total gems picked up: $total_gems_picked_up"
    @info "Final score: $combined_score"

    # Generate and save animation
    anim = anim_plan(renderer, domain, initial_state, actions; format="gif", framerate=2)
    save(joinpath(output_folder, "plan.mp4"), anim)

    close(io)

    return combined_score
end

function run_simulation_gpt4o(
    problem_name::String,
    ground_truth_rewards::Dict{Symbol, Int},
    T::Int,
    output_dir::String,
    gridworld_only::Bool = false
)
    task = "gpt4o"
    output_folder = joinpath(output_dir, task, problem_name)
    mkpath(output_folder)
    io = setup_logging(output_folder, "simulation_log.txt")

    PDDL.Arrays.register!()

    # Load domain and problem
    domain = load_domain(joinpath(@__DIR__, "domain.pddl"))
    problem::Problem = load_problem(joinpath(@__DIR__, "problems", problem_name * ".pddl"))
    initial_state = initstate(domain, problem)
    
    # Retrieve objects before compilation
    items = [obj.name for obj in PDDL.get_objects(domain, initial_state, :gem)]
    agents = Symbol[obj.name for obj in PDDL.get_objects(domain, initial_state, :agent)]
    
    # Renderer setup
    renderer = setup_renderer(agents, gridworld_only)
    canvas = renderer(domain, initial_state)

    # Output setup
    
    io = setup_logging(output_folder, "simulation_log.txt")
    save(joinpath(output_folder, "initial_state.png"), canvas)

    # Main simulation loop
    actions = []
    combined_score = 0
    remaining_items = copy(items)
    possible_gems = collect(keys(ground_truth_rewards))
    possible_rewards = collect(values(ground_truth_rewards))
    num_gems_picked_up = Dict(agent => 0 for agent in agents)
    total_gems_picked_up = 0
    state = initial_state

    # Initialize beliefs and heuristics optimistically
    gem_string = join(possible_gems, ", ")
    reward_string = join(possible_rewards, ", ")
    gpt4o_context = Dict{Symbol, String}(agent => "Please determine the reward associated with each gem given the following observations observed from traversing the gridworld environment. The possible gems are $gem_string. The possible rewards are $reward_string. Please responds with only the determined gem rewards (for each gem type), using the format <color>:<reward> separated by commas. If you are very uncertain of a reward for a gem, set the reward to 1.0." for agent in agents)
    beliefs = Dict(agent => Dict(gem => 5.0 for gem in [:red, :blue, :yellow, :green]) for agent in agents)
    heuristics = [VisionGemHeuristic(agent, beliefs[agent]) for agent in agents]

    # Initialize planners
    planners = [RTHS(heuristic, n_iters=0, max_nodes=5) for heuristic in heuristics]
    
    # Initialize observations for each agent
    observations = Dict(agent => Gen.choicemap() for agent in agents)

    # Initialize previous utterances
    previous_utterances = Dict{Symbol, Union{Nothing, String}}(agent => nothing for agent in agents)

    # Main simulation loop
    t = 1
    while !isempty(remaining_items) && t <= T
        @info "Step $t:"
        current_utterances = Dict{Symbol, Union{Nothing, String}}(agent => nothing for agent in agents)
        for (i, agent) in enumerate(agents)

            gpt4o_context[agent] *= "\n timestep t = $t"

            goals = PDDL.Term[]
            rewards = Float64[]        
            current_beliefs = beliefs[agent]
            for gem in remaining_items
                gem_obj = PDDL.Const(gem)
                color = Symbol(split(string(gem), "_")[1])
                reward = current_beliefs[color]
                heuristics[i].rewards[color] = reward

                if reward >= 0
                    push!(goals, PDDL.pddl"(has $agent $gem_obj)")
                    push!(rewards, reward)
                end
            end

            spec = MultiGoalReward(goals, rewards, 0.95)
            sol = planners[i](domain, state, spec)
            action = boltzmann_action(sol.value_policy, state, agent, 0.)
            state = transition(domain, state, action; check=true)

            # Clear previous observations and create new observation for this timestep
            observations[agent] = Gen.choicemap()
            observations[agent][(t => :self => :gem_pickup)] = false

            if action.name == :pickup
                item = action.args[2].name
                @info "       $agent picked up $item."
                remaining_items = filter(x -> x != item, remaining_items)
                gem = parse_gem(String(item))
                num_gems_picked_up[agent] += 1
                total_gems_picked_up += 1
                reward = ground_truth_rewards[gem]
                combined_score += reward
                @info "       $agent received $reward score."
                @info "       Combined score is now $combined_score."

                # Update the observations for pickup
                gpt4o_context[agent] *= "\n self gem pickup = true"
                gpt4o_context[agent] *= "\n self gem type = $gem"
                gpt4o_context[agent] *= "\n self gem reward = $reward"

                # Generate utterance
                utterance_tr, _ = Gen.generate(utterance_model, (gem, reward), Gen.choicemap())
                utterance = Gen.get_retval(utterance_tr)
                current_utterances[agent] = utterance
                gpt4o_context[agent] *= "\n self utterance = $utterance"
                @info "       $agent communicated: $utterance"
            else
                gpt4o_context[agent] *= "\n gem pickup = false"
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
            certainties = Dict(gem => 1.0 for gem in possible_gems)

            print_estimated_rewards(agent, beliefs[agent], certainties)

            push!(actions, action)
        end

        # Update previous_utterances for the next timestep
        previous_utterances = current_utterances

        t += 1
    end

    @info "Time: $t"
    @info "Total gems picked up: $total_gems_picked_up"
    @info "Final score: $combined_score"

    # Generate and save animation
    anim = anim_plan(renderer, domain, initial_state, actions; format="gif", framerate=2)
    save(joinpath(output_folder, "plan.mp4"), anim)

    close(io)

    return combined_score
end

end