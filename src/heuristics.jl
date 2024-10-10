struct VisionGemHeuristic <: Heuristic
    agent::Symbol
    rewards::Dict{Symbol, Float64} # Colour to reward value
    visited_states::Dict{Tuple{Int,Int}, Int}
end

function VisionGemHeuristic(agent::Symbol, rewards::Dict{Symbol, Float64})
    return VisionGemHeuristic(agent, rewards, Dict{Tuple{Int,Int}, Int}())
end

function SymbolicPlanners.compute(h::VisionGemHeuristic, domain::Domain, state::State, spec::Specification)
    agent = h.agent
    agent_pos = get_agent_pos(state, agent)
    visible_gems = get_visible_gems(domain, state, agent)
    positive_reward_gems = [(gem, pos) for (gem, pos) in visible_gems if h.rewards[gem_to_color(gem)] >= 0]
    value = 0.0
    if !isempty(positive_reward_gems) # Exist gems with positive reward
        for (gem, gem_pos) in positive_reward_gems
            distance = astar_path_length(domain, state, h.agent, gem_pos)
            gem_value = h.rewards[gem_to_color(gem)]
            value -= gem_value / (distance + 1)  # Avoid division by zero
        end
    else # Apply backtracking penalty only when no gems are visible to encourage exploration
        visit_count = get(h.visited_states, agent_pos, 0)
        backtrack_penalty = 0.1 * visit_count
        value += backtrack_penalty
    end
    
    h.visited_states[agent_pos] = get(h.visited_states, agent_pos, 0) + 1
    
    return value
end

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

function astar_path_length(domain::Domain, state::State, agent::Symbol, gem_pos::Tuple{Int,Int})
    goal = PDDL.parse_pddl("(and (= (xloc $agent) $(gem_pos[1])) (= (yloc $agent) $(gem_pos[2])))")
    planner = AStarPlanner(GoalCountHeuristic())
    spec = MinStepsGoal(goal)
    solution = planner(domain, state, spec)
    return length(solution)
end

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


struct ShortSightedVisionGemHeuristic <: Heuristic
    agent::Symbol
    rewards::Dict{Symbol, Float64} # Colour to reward value
    visited_states::Dict{Tuple{Int,Int}, Int}
end

function ShortSightedVisionGemHeuristic(agent::Symbol, rewards::Dict{Symbol, Float64})
    return ShortSightedVisionGemHeuristic(agent, rewards, Dict{Tuple{Int,Int}, Int}())
end

function SymbolicPlanners.compute(h::ShortSightedVisionGemHeuristic, domain::Domain, state::State, spec::Specification)
    agent = h.agent
    agent_pos = get_agent_pos(state, agent)
    gem_on_tile = get_gem_on_tile(domain, state, agent)
    value = 0.0
    
    if gem_on_tile !== nothing
        gem, gem_pos = gem_on_tile
        gem_value = h.rewards[gem_to_color(gem)]
        value -= gem_value  # Direct reward, no distance calculation needed
    else
        # Apply backtracking penalty when no gem is on the same tile
        visit_count = get(h.visited_states, agent_pos, 0)
        backtrack_penalty = 0.1 * visit_count
        value += backtrack_penalty
    end
    
    h.visited_states[agent_pos] = get(h.visited_states, agent_pos, 0) + 1
    
    return value
end

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

struct GoalManhattan <: Heuristic
    agent::Symbol
    rewards::Dict{Symbol, Float64}
    visited_states::Dict{Tuple{Int,Int}, Int}
end

# Constructor
function GoalManhattan(agent::Symbol, rewards::Dict{Symbol, Float64})
    return GoalManhattan(agent, rewards, Dict{Tuple{Int,Int}, Int}())
end

function SymbolicPlanners.compute(h::GoalManhattan, domain::Domain, state::State, spec::Specification)
    agent = h.agent

    # Get all non-offgrid gems
    all_gems = get_on_grid_gems(domain, state)

    # Filter gems with non-negative reward
    positive_reward_gems = [(gem, pos) for (gem, pos) in all_gems if h.rewards[gem_to_color(gem)] >= 0]
    
    agent_pos = get_agent_pos(state, agent)
    
    if isempty(positive_reward_gems)
        # Exploration heuristic when no gems believed to be postive
        visit_count = get(h.visited_states, agent_pos, 0)
        exploration_value = 1.0 / (visit_count + 1)  # Encourage exploring less-visited states
        h.visited_states[agent_pos] = visit_count + 1
        return exploration_value
    else
        # Compute minimum distance to valuable gems
        min_dist = minimum(positive_reward_gems) do (gem, gem_pos)
            sum(abs.(agent_pos .- gem_pos))
        end
        
        return min_dist + length(positive_reward_gems)
    end
end