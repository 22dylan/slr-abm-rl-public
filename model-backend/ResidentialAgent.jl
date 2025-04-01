const State = MArray{Tuple{8,},Float32}    # state; using static array (11x11 grids by 6 metrics (neighborhood migration, exposure, electric, transportation, elevated, generator installed))
@agent struct ResidentialAgent(GridAgent{2})
    # id::Int64                 # agent id; requied by gridspace; automatically included in @agent
    # pos::Tuple(Int64,Int64)   # agent position; required by gridspace; automatically included in @agent
    pos_guid::String            # guid that agent is associated with or "none"
    pos_idx::Int
    state_prev::State
    state::State                # agent state defined as mutable static array
    actions::Vector{Symbol}     # actions available for agent to take
    action_indices::Dict
    action_weights::Vector{Float32} # weights to use when sampling random action in epsilon greedy 
    action::Symbol              # action chosen
    family::Symbol              # agent family; defines which Q-net to use
    view_radius::Int64          # radius, r, that agent can view; overall view grid is (2r+1)x(2r+1); this is auto-calculated from State
    center_pos::Tuple
    neighborhood_original::MArray{Tuple{11,11}} # number of neighbors at start of simulation
    age::Int64
    # alphas::Vector{Float64}
    # agent_decision_type::Symbol
    reward::Float32
    p_migr::Float32
    p_expd::Float32
    p_elec::Float32
    p_trns::Float32
    q_ntng::Float32
    q_leav::Float32
    q_elev::Float32
    q_gnrt::Float32
    # ntwk_num::Int64
end


# function define_network(::Type{ResidentialAgent}, rng::MersenneTwister)
#     network = Chain(
#                    Dense(length(State)=>6, tanh; init=Flux.glorot_uniform(rng)),
#                    Dense(6=>6, tanh; init=Flux.glorot_uniform(rng)),
#                    Dense(6=>length(define_actions(ResidentialAgent)))
#                    )
#     return network
# end

function define_network(::Type{ResidentialAgent}, rng::MersenneTwister)
    network = Chain(
                   Dense(length(State)=>6, tanh; init=Flux.glorot_uniform(rng)),
                   Dense(6=>6, tanh; init=Flux.glorot_uniform(rng)),
                   Dense(6=>length(define_actions(ResidentialAgent)))
                   )
    return network
end


define_actions(::Type{ResidentialAgent}) = [:nothing, :leave, :elevate, :generator]
define_action_weights(::Type{ResidentialAgent}) = Float32[0.25, 0.25, 0.25, 0.25]        # weights used for sampling action in epsilon greedy
define_action_indices(agent_type::Type{ResidentialAgent}) = Dict(a=>Int32(i) for (i,a) in enumerate(define_actions(agent_type)))
define_state(::Type{ResidentialAgent}) = State(zeros(Float32, size(State)))
available_agent_families(::Type{ResidentialAgent}) = [:family1] #, :family2] #, :family3, :family4] # todo: define agent families


"""
    get_agent_weights(::Type{ResidentialAgent}, family::Symbol)
returns agent's weights that go into reward function
w1: place attachment
w2: neighborhood weight
w3: adverseness to exposure
w4: adverseness to lack of electricity
w5: adverseness to lack of access to exit
"""

function get_agent_weights(agent_family::Symbol, ::Type{ResidentialAgent})
    f = joinpath("agent-weights.csv")
    df = read_csv(f)
    df_ = df[findfirst(df.runname.==String(agent_family)), :]
    w = [df_[:PR], df_[:w_NP], df_[:w_BP], df_[:w_EP], df_[:w_TP]]
    w = Float32.(w)
    return w
end  

function get_agent_costs(agent_family::Symbol, ::Type{ResidentialAgent})
    f = joinpath("agent-weights.csv")
    df = read_csv(f)
    df_ = df[findfirst(df.runname.==String(agent_family)), :]
    cost_elevate = -Float32(df_[:c_elevate])    # costs are negative
    cost_generator = -Float32(df_[:c_generator])        # costs are negative
    costs = [cost_elevate, cost_generator]
    return costs
end
   
function get_bson_episode(input_struct::InputStruct, ::Type{ResidentialAgent})
    f = joinpath("agent-weights.csv")
    df = read_csv(f)
    be = df[findfirst(df.runname.==input_struct.sub_runname), :bson_episode]
    be = "$(be)"
    return be
end


# function initialize_replay_buffer(model::ABM, agent::Type{ResidentialAgent}, action_indices::Dict, weights::Vector{Float32}, dqn_param::DQNParam)
#     # init and populate replay buffer
#     replay = MyPrioritizedReplayBuffer(agent, dqn_param.buffer_size, dqn_param.batch_size, rng=abmrng(model))
#     populate_replay_buffer!(model, replay, agent, action_indices, max_pop=dqn_param.train_start, weights=weights)
#     return replay
# end

# """
#     populate_replay_buffer!()
# do some clever coding to make it generalizable and in DQN.jl
# """
# function populate_replay_buffer!(model::ABM,
#                                  replay::MyPrioritizedReplayBuffer,
#                                  agent::Type{ResidentialAgent},
#                                  action_indices::Dict;
#                                  max_pop::Int64=replay.max_size, 
#                                  max_steps::Int64=100,
#                                  weights::Vector{Float32}=Float64[]
#                                  )
#     # populating replay buffer with random samples of agents and timesteps of exposure
#     actions = define_actions(agent)
    
#     # getting center position
#     # view_radius = div(size(State,1)-1,2)
#     view_radius = 5             # TODO: come back to this; don't hard code
#     center_pos = (view_radius+1, view_radius+1)
#     w = get_agent_weights(model.input_struct, ResidentialAgent, :family1)
#     c = get_agent_costs(model.input_struct, ResidentialAgent)
#     for n=1:(max_pop - replay._curr_size)
#         # --- getting random agent
#         rand_guid = rand(abmrng(model), model.prcl_df[!,:guid])
#         rand_guid_idx = guid2idx(rand_guid, model)
#         rand_pos = model.prcl_df[rand_guid_idx, :pos]

#         # --- getting parcel exposure at random time step
#         random_year = rand(abmrng(model), model.start_year:model.n_years+model.start_year-1)     # getting random year
#         random_tick = random_year - model.start_year + 1                        # model tick corresponding to random_year
        
#         # --- defining current state (s)
#         s = State(zeros(Float32, size(State)))
#         s[7] = sample(abmrng(model), [0.0f0, 0.5f0, 1.0f0])       # discount on cost to leave
#         s[8] = sample(abmrng(model), [0.0f0, 0.5f0, 1.0f0])       # discount on cost to elevate
#         s[9] = sample(abmrng(model), [0.0f0, 0.5f0, 1.0f0])       # discount on cost to install generator
#         update_state_replay_buffer!(s, rand_pos, rand_guid_idx, model, random_year)

#         # --- define random action (a, ai)
#         a = rand(abmrng(model), actions)
#         ai = action_indices[a]

#         # --- update next state (sp)
#         sp = State(zeros(Float32, size(State)))                                 # if agent leaves, then sp = zeros
#         (a != :leave) && (update_state_replay_buffer!(sp, rand_pos, rand_guid_idx, model, random_year+1, s))  # if agent is staying, update the next state

#         (s[5]==1f0) && (sp[2]=0f0)                # if elevated, then no exposure
#         (s[6]==1f0) && (sp[3]=0f0)                # if generator, then no electric outages
#         (a==:elevate) && (sp[2]=0f0)              # if elevating, assume exposure goes to 0
#         (a==:elevate) && (sp[5]=1f0)              # if elevating, then next state is elevated
#         (a==:generator)   && (sp[3]=0f0)              # if installing generator, assume no electric outages
#         (a==:generator)   && (sp[6]=1f0)              # if elevating, then next state is elevated

#         # --- compute reward (rew)
#         rew = reward_func(s, a, w, c)


#         # --- adding experience to replay buffer
#         done = false
#         (a==:leave) && (done=true)
#         exp = MyDQExperience(s, ai, Float32(rew), sp, done)
#         add_exp!(replay, exp, abs(Float32(rew)))
#     end
#     @assert replay._curr_size >= replay.batch_size
# end

function get_original_neighborhood(pos::Tuple, model::ABM)
    view_radius = model.input_struct.agent_view_radius
    center_pos = (view_radius+1, view_radius+1)
    nearby_pos = nearby_positions(pos, model, view_radius)
    nghbr_orgl = zeros(MArray{Tuple{11,11},Float32})
    for near_pos in nearby_pos
        d_pos = near_pos .- pos                                                 # difference between nearby position and current position
        idx = (center_pos[2]-d_pos[2], center_pos[1]+d_pos[1])                  # idx in state is (row,col); confusing because rows are changes in "y-direction", cols are changes in "x-direction"
        if haskey(model.pos_to_guid_idx, near_pos)                              # if position has a parcel in it
            nghbr_orgl[idx...] += 1f0
        else
            nghbr_orgl[idx...] += 0f0
        end
    end
    return nghbr_orgl
end

# function update_state_replay_buffer!(s::State, 
#                                      pos::Tuple, 
#                                      rand_guid_idx::Int64, 
#                                      model::ABM, 
#                                      year::Int64, 
#                                      )
#     view_radius = model.input_struct.agent_view_radius

#     center_pos = (view_radius+1, view_radius+1)
#     nearby_pos = nearby_positions(pos, model, view_radius)
#     year_string = "_$(year)"
#     n_migrate = 0
#     cnt_nghbrs = 0
#     for near_pos in nearby_pos
#         d_pos = near_pos .- pos                                                 # difference between nearby position and current position
#         idx = (center_pos[2]-d_pos[2], center_pos[1]+d_pos[1])                  # idx in state is (row,col); confusing because rows are changes in "y-direction", cols are changes in "x-direction"
#         if haskey(model.pos_to_guid_idx, near_pos)                              # if position has a parcel in it
#             e = 0f0                                                             # pre-allocating exposure; 0f0 is a 0.0 in Float32
#             prcl_indices = model.pos_to_guid_idx[near_pos]                      # get prcl_df indices of parcels in cell
#             for prcl_i in prcl_indices                                          # loop through parcels in the cell
#                 e_ = get_prcl_info(model.slr_exposure, prcl_i, year_string)     # getting slr exposure of parcel
#                 (e_>e) && (e=e_)                                                # taking max exposure of parcel in cell

#             end
#             e = e/365                                                           # normalzing to percent of year
#             (e>1) && (e=1)                                                      # if greater than 1, set to 1

#             # using above for migration
#             m = 0f0                                                             # assuming no migration
#             (e>0.3) && (m=1f0)                                                  # if exposure > 0.3, assuming neighbors have left

#             # saving to states
#             n_migrate += m
#             cnt_nghbrs += 1
#         end
#     end

#     ## updating state of position itself
#     # first exposure
#     e = get_prcl_info(model.slr_exposure, rand_guid_idx, year_string)
#     e = e/365
#     (e>1) && (e=1)
#     state_exposure = convert(Float32, e)

#     # now electric
#     e = get_prcl_info(model.slr_elec_acc, rand_guid_idx, year_string)
#     e = e/365
#     (e>1) && (e=1)
#     state_electric = convert(Float32, e)

#     # then transportation
#     e = get_prcl_info(model.slr_exit_tt, rand_guid_idx, year_string)
#     u = get_prcl_info(model.slr_utmb_tt, rand_guid_idx, year_string)
#     tt = max(e, u)
#     tt = tt/365
#     (tt>1) && (tt=1)
#     state_transportation = convert(Float32, tt)

#     s[1] = n_migrate/cnt_nghbrs
#     (cnt_nghbrs==0f0) && (s[1]=0f0)
#     s[2] = state_exposure
#     s[3] = state_electric
#     s[4] = state_transportation

#     s[5] = rand(abmrng(model), (0f0, 1f0))
#     s[6] = rand(abmrng(model), (0f0, 1f0))
# end


# function update_state_replay_buffer!(s::State, 
#                                      pos::Tuple, 
#                                      rand_guid_idx::Int64, 
#                                      model::ABM, 
#                                      year::Int64, 
#                                      s_prev::State, 
#                                      )
#     view_radius = model.input_struct.agent_view_radius

#     center_pos = (view_radius+1, view_radius+1)
#     nearby_pos = nearby_positions(pos, model, view_radius)
#     year_string = "_$(year)"
#     n_migrate = 0
#     cnt_nghbrs = 0
#     for near_pos in nearby_pos
#         d_pos = near_pos .- pos                                                 # difference between nearby position and current position
#         idx = (center_pos[2]-d_pos[2], center_pos[1]+d_pos[1])                  # idx in state is (row,col); confusing because rows are changes in "y-direction", cols are changes in "x-direction"
#         if haskey(model.pos_to_guid_idx, near_pos)                              # if position has a parcel in it
#             e = 0f0                                                             # pre-allocating exposure; 0f0 is a 0.0 in Float32
#             prcl_indices = model.pos_to_guid_idx[near_pos]                      # get prcl_df indices of parcels in cell
#             for prcl_i in prcl_indices                                          # loop through parcels in the cell
#                 e_ = get_prcl_info(model.slr_exposure, prcl_i, year_string)     # getting slr exposure of parcel
#                 (e_>e) && (e=e_)                                                # taking max exposure of parcel in cell

#             end
#             e = e/365                                                           # normalzing to percent of year
#             (e>1) && (e=1)                                                      # if greater than 1, set to 1

#             # using above for migration
#             m = 0f0                                                             # assuming no migration
#             (e>0.3) && (m=1f0)                                                  # if exposure > 0.3, assuming neighbors have left

#             # saving to states
#             n_migrate += m
#             cnt_nghbrs += 1
#         end
#     end

#     ## updating state of position itself
#     # first exposure
#     e = get_prcl_info(model.slr_exposure, rand_guid_idx, year_string)
#     e = e/365
#     (e>1) && (e=1)
#     state_exposure = convert(Float32, e)

#     # now electric
#     e = get_prcl_info(model.slr_elec_acc, rand_guid_idx, year_string)
#     e = e/365
#     (e>1) && (e=1)
#     state_electric = convert(Float32, e)

#     # then transportation
#     e = get_prcl_info(model.slr_exit_tt, rand_guid_idx, year_string)
#     u = get_prcl_info(model.slr_utmb_tt, rand_guid_idx, year_string)
#     tt = max(e, u)
#     tt = tt/365
#     (tt>1) && (tt=1)
#     state_transportation = convert(Float32, tt)

#     s[1] = n_migrate/cnt_nghbrs
#     (cnt_nghbrs==0f0) && (s[1]=0f0)
#     s[2] = state_exposure
#     s[3] = state_electric
#     s[4] = state_transportation

#     s[5] = copy(s_prev[5])
#     s[6] = copy(s_prev[6])
# end


"""
    update_prev_state!(agnet::ResidentialAgent, model::ABM)
previous state becomes copy of agents state from last step
"""
function update_prev_state!(agent::ResidentialAgent, model::ABM)
    for i in 1:length(agent.state_prev)
        agent.state_prev[i] = agent.state[i]
    end
    agent.action = :nothing
end

function update_state!(agent::ResidentialAgent, model::ABM)
    update_state_migration!(agent, model)
    update_state_exposure!(agent, model, 0)
    update_state_electric!(agent, model, 0)
    update_state_transportation!(agent, model, 0)
    update_state_elevated!(agent, model)
    update_state_generator!(agent, model)
    # update_state_age!(agent, model)           # todo; uncomment this when ready
    # update_state_c_leave!(agent, model)
    update_state_c_elevate!(agent, model)
    update_state_c_generator!(agent, model)
end

function update_state_migration!(agent::ResidentialAgent, model::ABM)
    # s = get_neighborhood_new(agent, model)
    s = get_neighborhood(agent, model)
    n_migrate = sum(agent.neighborhood_original) - sum(s)
    agent.state[1] = (n_migrate)/sum(agent.neighborhood_original)
    agent.p_migr = (n_migrate)/sum(agent.neighborhood_original)
end


function count_n_migrate(o, s)
    f = Array{Int64}(undef, size(o))
    for i in eachindex(o, s)
        f[i] = o[i] - s[i]
    end
    return f
end

# """
#     get_neighborhood(agent, model)
# returns the neighborhood around the agent. 
# There is a allocation issue when using nearby_positions. 
# This function returns a generator, that then results in an allocation when calling
#     the values from this generator
# """
# function get_neighborhood_new(agent::ResidentialAgent, model::ABM)
#     # nghbrhd = zeros(Int64, size(agent.neighborhood_original))                 # pre-allocating space
#     nghbrhd = Array{Int64}(undef, size(agent.neighborhood_original))
#     nearby_pos = Agents.nearby_positions(agent.pos, model, agent.view_radius)   # getting nearby positions in ABM model
#     idx = Array{Int64}(undef, 2)
#     for near_pos in nearby_pos                                                  # loop through nearby positions
#         d_pos = near_pos .- agent.pos                                           # difference between nearby position and current position
#         idx[1] = agent.center_pos[2]-d_pos[2]
#         idx[2] = agent.center_pos[1]+d_pos[1]
#         if haskey(model.pos_to_guid_idx, near_pos)                              # if position has a parcel in it
#             id_pos = ids_in_position(near_pos, model)
#             if isempty(id_pos)
#                 nghbrhd[idx[1], idx[2]]=0
#             else
#                 nghbrhd[idx[1], idx[2]]=length(id_pos)
#             end
#         else
#             nghbrhd[idx[1], idx[2]] = 0                                         # if no parcel in position, then 0.0
#         end
#     end
#     # updating state of position itself
#     # nghbrhd[agent.center_pos...] = length(ids_in_position(agent.pos, model))
#     nghbrhd[agent.center_pos[1], agent.center_pos[2]] = length(ids_in_position(agent.pos, model))
#     return nghbrhd
# end

"""
    get_neighborhood(agent, model)
returns the neighborhood around the agent. 
There is a allocation issue when using nearby_positions. 
This function returns a generator, that then results in an allocation when calling
    the values from this generator
"""
function get_neighborhood(agent::ResidentialAgent, model::ABM)
    nghbrhd = zeros(Float32, size(agent.neighborhood_original))                 # pre-allocating space
    nearby_pos = Agents.nearby_positions(agent.pos, model, agent.view_radius)          # getting nearby positions in ABM model
    for near_pos in nearby_pos                                                  # loop through nearby positions
        d_pos = near_pos .- agent.pos                                           # difference between nearby position and current position
        idx = (agent.center_pos[2]-d_pos[2], agent.center_pos[1]+d_pos[1])                  # idx in state is (row,col); confusing because rows are changes in "y-direction", cols are changes in "x-direction"
        if haskey(model.pos_to_guid_idx, near_pos)                              # if position has a parcel in it
            id_pos = ids_in_position(near_pos, model)
            (isempty(id_pos))  && (nghbrhd[idx...]=0f0)                         # if no agent in position, then 0
            (~isempty(id_pos)) && (nghbrhd[idx...]=convert(Float32, length(id_pos)))  # if agent in position, then length of ids in pos
        else
            nghbrhd[idx...] = 0f0                                               # if no parcel in position, then 0.0
        end
    end
    # updating state of position itself
    nghbrhd[agent.center_pos...] = convert(Float32, length(ids_in_position(agent.pos, model)))
    return nghbrhd
end

function update_state_exposure!(agent::ResidentialAgent, model::ABM, dt::Int64)
    year_string = "_$(string(model.year+dt))"
    e = get_prcl_info(model.slr_exposure, agent.pos_idx, year_string)
    
    # check if building is elevated
    elevated = model.prcl_df[agent.pos_idx[1], :elevated]
    (elevated == true) && (e = 0)           # if so, then no exposure

    e = e/365
    (e>1) && (e=1)
    agent.state[2] = convert(Float32, e)
    agent.p_expd = convert(Float32, e)
    model.prcl_df[agent.pos_idx, :p_expd] = convert(Float32, e)
end


function update_state_electric!(agent::ResidentialAgent, model::ABM, dt::Int64)
    year_string = "_$(string(model.year+dt))"
    e = get_prcl_info(model.slr_elec_acc, agent.pos_idx, year_string)
    
    # check if building has generator
    generator = model.prcl_df[agent.pos_idx[1], :generator]
    (generator == true) && (e = 0)           # if so, then no loss of electricty

    e = e/365
    (e>1) && (e=1)
    agent.state[3] = convert(Float32, e)
    agent.p_elec = convert(Float32, e)
    model.prcl_df[agent.pos_idx, :p_elec] = convert(Float32, e)

end

function update_state_transportation!(agent::ResidentialAgent, model::ABM, dt::Int64)
    year_string = "_$(string(model.year+dt))"

    # updating state of position itself
    e = get_prcl_info(model.slr_exit_tt, agent.pos_idx, year_string)
    u = get_prcl_info(model.slr_utmb_tt, agent.pos_idx, year_string)
    tt = max(e,u)
    tt = tt/365

    (tt>1) && (tt=1)
    agent.state[4] = convert(Float32, tt)
    agent.p_trns = convert(Float32, tt)
    model.prcl_df[agent.pos_idx, :p_trns] = convert(Float32,tt)
end

function update_state_elevated!(agent::ResidentialAgent, model::ABM)
    elevated = model.prcl_df[agent.pos_idx[1], :elevated]
    (elevated==true) &&  (agent.state[5]=1f0)
    (elevated==false) && (agent.state[5]=0f0)
end

function update_state_generator!(agent::ResidentialAgent, model::ABM)
    generator = model.prcl_df[agent.pos_idx[1], :generator]
    (generator==true) &&  (agent.state[6]=1f0)
    (generator==false) && (agent.state[6]=0f0)
end

# function update_state_age!(agent::ResidentialAgent, model::ABM)
#     agent.age += 1
#     agent.state[7] = agent.age/80
# end

# function update_state_c_leave!(agent::ResidentialAgent, model::ABM)
#     agent.state[7] = 1.0f0
# end

"""
    update_state_c_elevate!(agent::ResidentialAgent, model::ABM)
function to update the costs to elevate
note that this is currently represented a percentage of `c_elevate` that is
  passed into the model via agent-weights.csv
Example, agent.state[7] = 0.5 indicates that the cost to elevate is 50% of `c_elevate`
"""
function update_state_c_elevate!(agent::ResidentialAgent, model::ABM)
    agent.state[7] = 0.82f0
    # if (model.year >= 2090) && (model.year < 2100)
    # if agent.state[2]> 0f0
        # agent.state[7] = 0.41f0
    # else
        # agent.state[7] = 0.82f0
    # end
end

"""
    update_state_c_generator!(agent::ResidentialAgent, model::ABM)
function to update the costs to install a generator
note that this is currently represented a percentage of `c_generator` that is
  passed into the model via agent-weights.csv
"""
function update_state_c_generator!(agent::ResidentialAgent, model::ABM)
    agent.state[8] = 1.0f0
end



# """
#     act!(agent, model, action)
# residential agent action,
# if agent has already left, then action is nothing.
#   note that agent still collects reward if they've left
# """
function act!(agent::ResidentialAgent, model::ABM)
    (agent.action == :nothing)   && (return)
    # (agent.action == :nothing)   && (agent_stays!(agent, model))
    (agent.action == :leave)     && (agent_leaves!(agent, model))
    (agent.action == :elevate)   && (agent_elevates!(agent, model))
    (agent.action == :generator) && (agent_installs_generator!(agent, model))
end

function agent_stays!(agent::ResidentialAgent, model::ABM)
    model.prcl_df[!,:occupied][agent.pos_idx] = 1
end

function agent_leaves!(agent::ResidentialAgent, model::ABM)
    model.prcl_df[!,:occupied][agent.pos_idx] = 0
    move_agent!(agent, (1, 250), model)     # if agent leaves, temporarily move to null position
end

function agent_elevates!(agent::ResidentialAgent, model::ABM)
    model.prcl_df[!,:elevated][agent.pos_idx] = 1
end

function agent_installs_generator!(agent::ResidentialAgent, model::ABM)
    model.prcl_df[!,:generator][agent.pos_idx] = 1
end


# # removing agent's chosen action from the their options
# function remove_action!(agent::ResidentialAgent, model::ABM)
#     deleteat!(agent.action_weights, findfirst(x->x==agent.action, agent.actions))
#     deleteat!(agent.actions, findfirst(x->x==agent.action, agent.actions))
# end

"""
    reward_func(agent::ResidentialAgent)
reward signal for ResidentialAgent. This is function of:
    PR: place reward; e.g., place-based attachment to home 
    NP: neighbor penalty
    BP: building exposure penaly; e.g., desire to not be in home when flooded
    EP: electric loss penalty; e.g. lose electricity
    TP: transportation penalty; e.g., increase in travel time
reward is 0 after agent has left; e.g., they either made a good or bad decision
"""
function reward_func(agent::ResidentialAgent, model::ABM)
    r = reward_func(agent.state_prev,
                    agent.action, 
                    model.ResidentialAgent_Dict[agent.family][:weights], 
                    model.ResidentialAgent_Dict[agent.family][:costs],
                    )
    return r
end


function reward_func(s::State, action::Symbol, w::Vector{Float32}, c::Vector{Float32})
    # (age>80) && (return 0)
    (action==:leave) && (return 0)
    (action==:elevate) && (return s[7]*c[1])            # if agent elevates, then there is a cost associated with it
    (action==:generator) && (return s[8]*c[2])          # if agent installs generator, then there is a cost associated with it

    # --- reward function ---
    PR = w[1]                                           # place attachment reward
    NP = w[2]*s[1]                                      # percent of neighbors migrated; neighbor penalty
    # BP = w[3]*sp[2]                                     # percent year with building exposed; exposure penalty
    
    BP = reward_tiers(w[3], s[2])
    # EP = reward_tiers(w[4], s[3])

    EP = w[4]*s[3]                                     # percent year with electric outage; electric penalty
    TP = w[5]*s[4]                                     # percent year with increase in travel time; transportation penalty

    r = PR - (NP+BP+EP+TP)
    r = convert(Float32,r)
    return r
end


function reward_tiers(w::Float32, s::Float32)
    if s == 0.0f0
        return 0
    elseif (s<0.1)
        return w
    elseif (s>=0.1) && (s<0.3)
        return w*2
    elseif s >= 0.3
        return w*3
    end

end

# """
#     check_add_experience_to_replay(exp::MyDQExperience, model::ABM)
# checks whether to add experience to replay buffer; used to avoid catastrophic 
# forgetting and oversaturating replay buffer
# """
# function check_add_experience_to_replay(exp::MyDQExperience, model::ABM)
#     return true

#     # add_exp = false                                                             # initiating at false
#     # (sum(exp.s) != 0f0) && (add_exp = true)                                     # if non-zero, return true
#     # (rand(abmrng(model)) < 0.05) && (add_exp = true)                            # otherwise, return true with p=0.05
#     # return add_exp
# end

function agent_close_step!(agent::ResidentialAgent, model::ABM)
    return
end

"""
    utility_theory(agent::ResidentialAgent, model::ABM)
Using utility theory for some agents during training.
This is here 
"""
# function utility_theory!(agent::ResidentialAgent, model::ABM)
#     # Extract the exposure of the agent's own position from the state
#     center_pos = (agent.view_radius + 1, agent.view_radius + 1)
#     penalties = agent.state[1] + agent.state[4]     # picking penalties that agent has no control over (e.g., neighbors and travel time)
#     if penalties > rand(Uniform(0.2,1))
#         agent.action = :leave
#     end

#     # ####################################
#     # # Calculate UStay for the current agent
#     # UStay_self = 100 ^ agent.alphas[1]

#     # # Initialize 2nd and 3rd alpha values
#     # alpha_2 = agent.alphas[2]
#     # alpha_3 = agent.alphas[3]

#     # # Calculate the number of neighbors at the start of the simulation
#     # num_neighbors_start = sum(agent.neighborhood_original)

#     # # Calculate the number of neighbors at the current time step
#     # num_neighbors_current = sum(get_neighborhood(agent, model))


#     # # Calculate P_neighbor using the given formula
#     # P_neighbor = 100 * (num_neighbors_current / num_neighbors_start)

#     # # Calculate Ustay_neighbor
#     # UStay_neighbor = P_neighbor ^ alpha_2

#     # # Determine UStay
#     # UStay = UStay_self * UStay_neighbor

#     # # Update state
#     # # update_state!(agent, model)  # Ensure this function is defined elsewhere in your code

#     # # Extract the exposure of the agent's own position from the state
#     # center_pos = (agent.view_radius + 1, agent.view_radius + 1)
#     # exposure = agent.state[center_pos..., 2] * 100

#     # # Calculate ULeave
#     # ULeave = exposure ^ alpha_3

#     # # Agent decision logic
#     # if UStay > ULeave
#     #     agent.action = :nothing
#     # elseif ULeave > UStay
#     #     agent.action = :leave
#     # end

#     # return agent
# end


function check_age!(agent::ResidentialAgent)
    if agent.age == 80
        agent.action = :leave
    end
end

"""
    decide_action(agent::T, network::Chain) where T<: AbstractAgent
function for agent to decide action using trained neural network
"""
function decide_action!(agent::T, network::Chain) where T<:AbstractAgent
    values = network(agent.state_prev)
    agent.q_ntng = values[1]
    agent.q_leav = values[2]
    agent.q_elev = values[3]
    agent.q_gnrt = values[4]
    agent.action = Flux.onecold(values, agent.actions) 
end


"""
    agent_evaluate_step!(agent::ResidentialAgent, model::ABM)
after all agents perform their selected action (previous function), then evaluate 
how good of a decision that action was.

this function, sets the next state (sp), computes reward, adds experience to 
the replay buffer, then trains the networks (if applicable)
"""
function evaluate_action!(agent::ResidentialAgent, model::ABM)
    s = copy(agent.state_prev)                                                  # agent's original state
    ai = agent.action_indices[agent.action]                                     # action index; for logging in replay buffer

    (agent.action==:leave)  && (agent.state=State(zeros(Float32, size(State)))) # if agent is leaving, reset next state to 0's
    rew = reward_func(agent, model)                                             # reward for taking action
    agent.reward += rew
    
    # logging reward to agent's network
    model.ResidentialAgent_Dict[agent.family][:training_df][model.ResidentialAgent_Dict[agent.family][:training_df][!,:ntwk_num].==agent.ntwk_num, Symbol("reward_$(model.iteration)")] .+= rew

    done = false
    (agent.action==:leave) && (done=true)                                       # if agent leaves, then done flag is true; note agent's action=:leave if age is 80
    sp = copy(agent.state)                                                      # next state as a result of taking the action

    exp = MyDQExperience(s, ai, Float32(rew), sp, done)                         # setting experience in tuple
    dqn_train_step!(agent, model, model.ResidentialAgent_Dict, exp)             # training dqn

    (agent.action==:leave) && (remove_agent!(agent, model))                     # if agent has left, remove agent from model after adding experience to replay buffer
end



### TODO: Update agent step such that :leave means the agent can look for other parcels, only during runtime
# """
#     agent_step!(agent::ResidentialAgent, model::ABM)
# agent taking action based on input state (or epsilon greedy)
# agent first decides the action to take (decide_action!), then performs action (act!)
# all agents first take an action (here), then compute reward based on next state (following function)
# """
# function agent_step!(agent::ResidentialAgent, model::ABM)
#     # aging the agent; if greater than 80, then remove from the model
#     agent.age += 1
#     if agent.age > 80
#         agent.action = :leave
#         act!(agent, model, agent.action)
#         remove_agent!(agent, model)
#         return agent
#     end

#     # if agent has not left due to age, then decide action based on exposure/migration in neighborhood
#     decide_action!(agent, model, model.ResidentialAgent_Dict[agent.family][:network]) # deciding action based curent state; epsilon greedy

#     # performing the action
#     act!(agent, model, agent.action)                                            # performing action       

#     # computing reward and updating the agent state, and logging to eval_log
#     # rew = reward_func(agent, model)
#     update_state!(agent, model)
#     # model.ResidentialAgent_Dict[agent.family][:DQNParam].eval_log[end] += rew

#     # if agent.action == nothing, then we're done here; return agent    
#     (agent.action == :nothing) && (return agent)

#     # otherwise, if agent decides to leave, then look through available parcels for one that's suitable
#     if agent.action==:leave
#         empty_prcls = model.prcl_df[model.prcl_df.occupied .== 0, :]
        
#         prcl_rows = sample(abmrng(model), 
#                            1:size(empty_prcls)[1], 
#                            min(size(empty_prcls)[1], 10), 
#                            replace=false)
#         for i in prcl_rows
#             p = empty_prcls[i,:]                        # parcel to try out
#             move_agent!(agent, p.pos, model)            # temporarily move agent to new parcel; see if it works
#             update_state!(agent, model)                 # updating state of agent; view exposure/neighborhood

#             # deciding action based curent state
#             decide_action!(agent, model, model.ResidentialAgent_Dict[agent.family][:network])
#             # if action is :nothing, then agent likes parcel, decides to stay
#             if agent.action == :nothing
#                 act!(agent, model, agent.action)
#                 return agent
#             end

#         end

#     end
#     # if action is still :leave, after checking available parcels, then agent leaves Galveston
#     (agent.action==:leave) && (act!(agent, model, agent.action))                # updates prcl df
#     (agent.action==:leave) && (remove_agent!(agent, model))                     # now remove agent from model
#     return agent
# end


function evaluate_run_action!(agent::ResidentialAgent, model::ABM)
    # computing reward and updating the agent state, and logging to eval_log
    rew = reward_func(agent, model)
    agent.reward += rew
    # model.ResidentialAgent_Dict[agent.family][:DQNParam].eval_log[end] += rew
    (agent.action==:leave) && (remove_agent!(agent, model))                     # if agent has left, remove agent from model after adding experience to replay buffer
end




