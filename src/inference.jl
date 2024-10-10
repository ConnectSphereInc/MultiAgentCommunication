import GenParticleFilters: pf_initialize, pf_update!, pf_resample!, pf_rejuvenate!, get_norm_weights, get_traces, effective_sample_size

function update_beliefs_communication(pf_state, current_timestep, num_agents, possible_gems, possible_rewards, observation, num_particles, ess_thresh)
    if pf_state === nothing
        pf_state = pf_initialize(agent_model_ours, (current_timestep, num_agents, possible_gems, possible_rewards), observation, num_particles)
    else
        pf_update!(pf_state, (current_timestep, num_agents, possible_gems, possible_rewards), (UnknownChange(),), observation)
    end
    return pf_state
end

function update_beliefs_no_communication(pf_state, current_timestep, num_agents, possible_gems, possible_rewards, observation, num_particles, ess_thresh)
    if pf_state === nothing
        pf_state = pf_initialize(agent_model_no_communication, (current_timestep, num_agents, possible_gems, possible_rewards), observation, num_particles)
    else
        pf_update!(pf_state, (current_timestep, num_agents, possible_gems, possible_rewards), (UnknownChange(),), observation)
    end
    return pf_state
end
