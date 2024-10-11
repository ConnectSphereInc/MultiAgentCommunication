"""
    VisionGemHeuristic(agent::Symbol, rewards::Dict{Symbol, Float64})

    A heuristic that encourages the agent to collect gems with positive rewards and discourages backtracking.
"""
struct VisionGemHeuristic <: Heuristic
    agent::Symbol
    rewards::Dict{Symbol, Float64} # Colour to reward value
    visited_states::Dict{Tuple{Int,Int}, Int}
end

"""
    VisionGemHeuristic(agent::Symbol, rewards::Dict{Symbol, Float64})

    Create a VisionGemHeuristic object with the given agent and rewards.
"""
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

"""
    RestrictedVisionGemHeuristic(agent::Symbol, rewards::Dict{Symbol, Float64})

    A heuristic that encourages the agent to collect gems with positive rewards and discourages backtracking.
    This heuristic does not consider the distance to the gems.
"""
struct RestrictedVisionGemHeuristic <: Heuristic
    agent::Symbol
    rewards::Dict{Symbol, Float64} # Colour to reward value
    visited_states::Dict{Tuple{Int,Int}, Int}
end

function RestrictedVisionGemHeuristic(agent::Symbol, rewards::Dict{Symbol, Float64})
    return RestrictedVisionGemHeuristic(agent, rewards, Dict{Tuple{Int,Int}, Int}())
end

function SymbolicPlanners.compute(h::RestrictedVisionGemHeuristic, domain::Domain, state::State, spec::Specification)
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

"""
The Manhattan distance heuristic as implemented in Plinf.jl.
https://github.com/ztangent/Plinf.jl
"""
struct GoalManhattan <: Heuristic
    agent::Symbol
end

function SymbolicPlanners.compute(h::GoalManhattan, domain::Domain, state::State, spec::Specification)
    agent = h.agent
    # Count number of remaining goals to satisfy
    goal_count = GoalCountHeuristic()(domain, state, spec)
    # Determine goal objects to collect
    goals = get_goal_terms(spec)
    goal_objs = [g.args[2] for g in goals if g.name == :has && g.args[1] == agent && !state[g]]
    if isempty(goal_objs)
        return goal_count
    end
    # Compute minimum distance to goal objects
    agent_pos = get_agent_pos(state, agent)
    min_dist = minimum(goal_objs) do obj
        gem_x = state[Compound(:xloc, [obj])]
        gem_y = state[Compound(:yloc, [obj])]
        abs(agent_pos[1] - gem_x) + abs(agent_pos[2] - gem_y)
    end
    value = min_dist + goal_count
    return value
end