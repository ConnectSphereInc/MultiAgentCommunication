function update_beliefs_communication(pf_state, current_timestep, num_agents, possible_gems, possible_rewards, observation, num_particles, ess_thresh)
    if pf_state === nothing
        pf_state = pf_initialize(agent_model_communication, (current_timestep, num_agents, possible_gems, possible_rewards), observation, num_particles)
    else
        num_threads = Threads.nthreads()
        worker_indices = [((i-1)*num_particles÷num_threads + 1):(i*num_particles÷num_threads) for i in 1:num_threads]
        
        Threads.@threads for i in 1:length(worker_indices)
            indices = worker_indices[i]
            substate = view(pf_state, indices)
            pf_update!(substate, (current_timestep, num_agents, possible_gems, possible_rewards), (UnknownChange(),), observation)
        end
    end
    return pf_state
end

function update_beliefs_no_communication(pf_state, current_timestep, num_agents, possible_gems, possible_rewards, observation, num_particles, ess_thresh)
    if pf_state === nothing
        pf_state = pf_initialize(agent_model_no_communication, (current_timestep, num_agents, possible_gems, possible_rewards), observation, num_particles)
    else
        num_threads = Threads.nthreads()
        worker_indices = [((i - 1) * num_particles ÷ num_threads + 1):(i * num_particles ÷ num_threads) for i in 1:num_threads]
        Threads.@threads for i in 1:length(worker_indices)
            indices = worker_indices[i]
            substate = view(pf_state, indices)
            pf_update!(substate, (current_timestep, num_agents, possible_gems, possible_rewards), (), observation)
        end
    end
    return pf_state
end
