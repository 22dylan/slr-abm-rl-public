# file for training agents
using Agents
using DataFrames
using GeoDataFrames
using ArchGDAL
using ProgressMeter
using CSV
using BSON
using ArgParse
using Random
using Distributions
using Flux
using StaticArrays
using DeepQLearning
using POMDPPolicies
using StatsBase
import GeoFormatTypes as GFT
using JSON
using DataStructures: OrderedDict
using EllipsisNotation

include(joinpath("InputStructsTrain.jl"))
include(joinpath("model-backend", "DQN.jl"))
include(joinpath("model-backend", "MyModel.jl"))
include(joinpath("model-backend", "ResidentialAgent.jl"))
include(joinpath("model-backend", "MiscFunc.jl"))
include(joinpath("model-backend", "MyParcelSpace.jl"))
include(joinpath("model-backend", "weights.jl"))


function train(model_runname, run_subname)
    println("\nTraining Agent: $(run_subname)")

    parsed_args = parse_commandline()
    (parsed_args["model_runname"] != nothing) && (model_runname = parsed_args["model_runname"])

    input_struct = InputStruct()
    input_struct.model_runname = model_runname     # updating model_runname in input struct
    input_struct.sub_runname = run_subname     # updating runsubname in input struct

    (parsed_args["train"] != nothing) && (input_struct.train = parsed_args["train"])
    run_training(input_struct, parsed_args)
end


function run_training(input_struct::InputStruct, parsed_args::Dict)
    println("  running solo training: $(input_struct.sub_runname)")

    (check_run_previously_done(input_struct, parsed_args)) && (return)          # check if run has been completed
    dqn_param = DQNParam()                                                      # setup DQN params
    read_dqn_discount!(dqn_param, input_struct)                                 # read dqn discount and update
    # path_to_input_dir = joinpath(pwd(), "model-runs", input_struct.model_runname, "input")  # path to input dir
    path_to_input_dir = joinpath(pwd(), "input")

    # setting up path to training dir; checking parsed_args; creating training dir
    path_to_training_dir = pwd()
    path_to_training_dir = joinpath(path_to_training_dir, "model-runs", input_struct.model_runname, "training-results", input_struct.sub_runname)
    (parsed_args["training_dir"] != nothing) && (path_to_training_dir=parsed_args["training_dir"])
    makedir(path_to_training_dir)

    rng = MersenneTwister(input_struct.seed)        # random number generator (rng)

    input_dict = read_dir_CSV(path_to_input_dir)  # input dict contains CSV files that are in the input directory
    prcl_df = prepare_parcel_df(path_to_input_dir, seed=input_struct.seed, cell_size=25, rotate=true)
    pos_to_guid_idx = setup_pos2guididx(prcl_df)

    replay = populate_replay_buffer!(input_struct, prcl_df, rng, ResidentialAgent, max_pop=dqn_param.train_start, pos_to_guid_idx=pos_to_guid_idx)    
    train_agent!(replay, rng, dqn_param, input_struct, prcl_df, pos_to_guid_idx, path_to_training_dir)
    println("  solo training done: $(input_struct.sub_runname)")
    return replay, dqn_param
end


function age_calc(rng::MersenneTwister)
    age_alpha = 15
    age_theta = 2.667
    dist = Gamma(age_alpha, age_theta)
    
    age = rand(rng, dist, 1)[1]
    age = convert(Int64,round(age))
    if age < 18
        age = 18
    end
    (age < 18) && (age=18)
    (age > 80) && (age=79)
    return age
end

function setup_agent(p::DataFrameRow, rng::MersenneTwister)
    s = define_state(ResidentialAgent)
    actions = define_actions(ResidentialAgent)
    action_indices = define_action_indices(ResidentialAgent)
    action_weights = define_action_weights(ResidentialAgent)
    r = 5
    center_pos = (6,6)
    agent_decision_type=:train
    # age = age_calc(rng)

    agent = ResidentialAgent(
                id=1,                          # agent id (required)
                pos=p.pos,                      # agent pos as tuple (required)
                pos_guid=p.guid,                # agent's current  position (guid)
                pos_idx=p.row_idx,                  # pos_idx; position index - used to quickly look up from prcl_df
                state_prev=s,                   # previous state
                state=copy(s),                        # initializing with in parcel, unexposed
                actions=actions,                # actions for agent to take
                action_indices=action_indices,  # action indicies
                action_weights=action_weights,  # action weights for epsilon greedy; todo, try running this one
                action=:nothing,                # action
                family=:family1,                # agent family
                view_radius=r,                  # agent view radius
                center_pos=center_pos,          # center position
                neighborhood_original=zeros(MArray{Tuple{11,11},Float32}), # number of neighbors
                age=30,                        # agent age
                reward=0f0,
                p_migr=0f0,
                p_expd=0f0,
                p_elec=0f0,
                p_trns=0f0,
                q_ntng=0f0,
                q_leav=0f0,
                q_elev=0f0,
                q_gnrt=0f0,
                )
    return agent
end


function setup_agent(prcl_df::DataFrame, rng::MersenneTwister)
    idx = rand(rng, 1:size(prcl_df)[1])
    p = prcl_df[idx,:]
    s = define_state(ResidentialAgent)
    actions = define_actions(ResidentialAgent)
    action_indices = define_action_indices(ResidentialAgent)
    action_weights = define_action_weights(ResidentialAgent)
    r = 5
    center_pos = (6,6)
    agent_decision_type=:train
    # age = age_calc(rng)

    agent = ResidentialAgent(
                id=1,                          # agent id (required)
                pos=p.pos,                      # agent pos as tuple (required)
                pos_guid=p.guid,                # agent's current  position (guid)
                pos_idx=p.row_idx,                  # pos_idx; position index - used to quickly look up from prcl_df
                state_prev=s,                   # previous state
                state=copy(s),                        # initializing with in parcel, unexposed
                actions=actions,                # actions for agent to take
                action_indices=action_indices,  # action indicies
                action_weights=action_weights,  # action weights for epsilon greedy; todo, try running this one
                action=:nothing,                # action
                family=:family1,                  # agent family
                view_radius=r,                  # agent view radius
                center_pos=center_pos,          # center position
                neighborhood_original=zeros(MArray{Tuple{11,11},Float32}), # number of neighbors
                age=30,                        # agent age
                reward=0f0,
                p_migr=0f0,
                p_expd=0f0,
                p_elec=0f0,
                p_trns=0f0,
                q_ntng=0f0,
                q_leav=0f0,
                q_elev=0f0,
                q_gnrt=0f0,
                )
    return agent
end

sample_weights(rng) = sample(rng, [0.0f0, 0.5f0, 0.75f0, 1.0f0])
function populate_replay_buffer!(input_struct::InputStruct,
                                prcl_df::DataFrame,
                                rng::MersenneTwister,
                                agent::Type{ResidentialAgent};
                                max_pop::Int64=replay.max_size, 
                                max_steps::Int64=100,
                                pos_to_guid_idx::Dict
                                 )
    dqn_param = DQNParam()
    replay = MyPrioritizedReplayBuffer(ResidentialAgent, dqn_param.buffer_size, dqn_param.batch_size, rng=rng)
    # populating replay buffer with random samples of agents and timesteps of exposure
    actions = define_actions(agent)
    action_indices = define_action_indices(ResidentialAgent)
    slr_exposure, slr_elec_acc, slr_exit_tt, slr_utmb_tt = read_slr_exposure(input_struct, order=prcl_df.guid, rng=rng)    # getting random slr scenario

    # getting center position
    center_pos = (input_struct.agent_view_radius+1, input_struct.agent_view_radius+1)
    w = get_agent_weights(Symbol(input_struct.sub_runname), ResidentialAgent)
    c = get_agent_costs(Symbol(input_struct.sub_runname), ResidentialAgent)
    for n=1:(dqn_param.train_start - replay._curr_size)
        # --- getting random agent
        rand_guid = rand(rng, prcl_df[!,:guid])
        rand_guid_idx = guid2idx(rand_guid, prcl_df)
        rand_pos = prcl_df[rand_guid_idx, :pos]
        
        # --- getting parcel exposure at random time step
        random_year = rand(rng, input_struct.model_start_year:input_struct.model_start_year+input_struct.n_years-1)     # getting random year
        random_tick = random_year - input_struct.model_start_year + 1                        # tick corresponding to random_year

        # --- initialize state
        s = State(zeros(Float32, size(State)))

        # --- getting random discounts on elevate/generator
        s[7] = sample_weights(rng)       # discount on cost to elevate
        s[8] = sample_weights(rng)       # discount on cost to install generator
        
        # --- update current state (s)
        update_state_replay_buffer!(s, 
                                    rand_pos, 
                                    rand_guid_idx, 
                                    random_year, 
                                    pos_to_guid_idx,
                                    slr_exposure,
                                    slr_elec_acc,
                                    slr_exit_tt,
                                    slr_utmb_tt,
                                    input_struct.agent_view_radius,
                                    rng
                                    )

        # --- define random action (a, ai)
        a = rand(rng, actions)
        ai = action_indices[a]

        # --- update next state (sp)
        sp = State(zeros(Float32, size(State)))
        if a != :leave
            update_state_replay_buffer!(sp, 
                                    rand_pos, 
                                    rand_guid_idx, 
                                    random_year+1, 
                                    pos_to_guid_idx,
                                    slr_exposure,
                                    slr_elec_acc,
                                    slr_exit_tt,
                                    slr_utmb_tt,
                                    input_struct.agent_view_radius,
                                    rng
                                    )
            sp[5] = copy(s[5])
            sp[6] = copy(s[6])
        end

        (s[5]==1f0) && (s[2]=0f0)                 # if elevated, then no exposure (current state)
        (s[6]==1f0) && (s[3]=0f0)                 # if generator, then no electric outages (current state)
        
        (s[5]==1f0) && (sp[5]=1f0)                # if elevated, then next state is elevated
        (s[6]==1f0) && (sp[6]=1f0)                # if generator, then next state is generator
        
        (a==:elevate) && (sp[5]=1f0)              # if elevating, then next state is elevated
        (sp[5]==1f0)  && (sp[2]=0f0)              # if elevating, assume exposure goes to 0
        
        (a==:generator)   && (sp[6]=1f0)              # if elevating, then next state is elevated
        (sp[6]==1f0)  && (sp[3]=0f0)              # if installing generator, assume no electric outages
        
        (a == :leave) && (sp = State(zeros(Float32, size(State)))) # if agent leaves, then sp = zeros

        # --- compute reward (rew)
        rew = reward_func(s, a, w, c)

        # --- adding experience to replay buffer
        done = false
        (a==:leave) && (done=true)
        exp = MyDQExperience(s, ai, Float32(rew), sp, done)
        add_exp!(replay, exp, abs(Float32(rew)))
    end
    @assert replay._curr_size >= replay.batch_size
    return replay
end


function nearby_positions(pos::Tuple, view_radius::Int64)
    near_pos = Tuple[]
    for dx in collect(-view_radius:view_radius)
        for dy in collect(-view_radius:view_radius)
            push!(near_pos, (pos[1]+dx, pos[2]+dy))
        end
    end
    return near_pos
end


function update_state_replay_buffer!(s::State, 
                                    pos::Tuple, 
                                    rand_guid_idx::Int64, 
                                    year::Int64, 
                                    pos_to_guid_idx::Dict, 
                                    slr_exposure::DataFrame,
                                    slr_elec_acc::DataFrame,
                                    slr_exit_tt::DataFrame,
                                    slr_utmb_tt::DataFrame,
                                    view_radius::Int64,
                                    rng::MersenneTwister
                                )
    center_pos = (view_radius+1, view_radius+1)
    nearby_pos = nearby_positions(pos, view_radius)
    year_string = "_$(year)"
    n_migrate = 0
    cnt_nghbrs = 0
    migrtn_thrshld = rand(rng, migrtn_thrshld_dist())    
    for near_pos in nearby_pos
        d_pos = near_pos .- pos                                                 # difference between nearby position and current position
        idx = (center_pos[2]-d_pos[2], center_pos[1]+d_pos[1])                  # idx in state is (row,col); confusing because rows are changes in "y-direction", cols are changes in "x-direction"
        if haskey(pos_to_guid_idx, near_pos)                                    # if position has a parcel in it
            prcl_indices = pos_to_guid_idx[near_pos]                            # get prcl_df indices of parcels in cell
            for prcl_i in prcl_indices                                          # loop through parcels in the cell
                nghbr_mgrt = check_nghbr_migrate(slr_exposure, slr_elec_acc, slr_exit_tt, slr_utmb_tt, prcl_i, year_string, migrtn_thrshld, rng)
                (nghbr_mgrt) && (n_migrate += 1f0)
                cnt_nghbrs += 1
            end
        end
    end

    ## updating state of position itself
    # first exposure
    e = get_prcl_info(slr_exposure, rand_guid_idx, year_string)/365
    (e>1) && (e=1)
    state_exposure = convert(Float32, e)

    # now electric
    e = get_prcl_info(slr_elec_acc, rand_guid_idx, year_string)/365
    (e>1) && (e=1)
    state_electric = convert(Float32, e)

    # then transportation
    e = get_prcl_info(slr_exit_tt, rand_guid_idx, year_string)/365
    u = get_prcl_info(slr_utmb_tt, rand_guid_idx, year_string)/365
    tt = max(e, u)
    (tt>1) && (tt=1)
    state_transportation = convert(Float32, tt)

    s[1] = n_migrate/cnt_nghbrs
    (cnt_nghbrs==0f0) && (s[1]=0f0)
    s[2] = state_exposure
    s[3] = state_electric
    s[4] = state_transportation

    s[5] = rand(rng, (0f0, 1f0))
    s[6] = rand(rng, (0f0, 1f0))
end

migrtn_thrshld_dist() = Uniform(0.05, 0.5)
function train_agent!(replay::MyPrioritizedReplayBuffer, 
                      rng::MersenneTwister, 
                      dqn_param::DQNParam, 
                      input_struct::InputStruct, 
                      prcl_df::DataFrame,
                      pos_to_guid_idx::Dict,
                      path_to_training_dir::String)
    active_q = define_network(ResidentialAgent, rng)
    target_q = deepcopy(active_q)
    weights  = get_agent_weights(Symbol(input_struct.sub_runname), ResidentialAgent)
    costs    = get_agent_costs(Symbol(input_struct.sub_runname), ResidentialAgent)

    # --- for evaluating network performance
    idxs = shuffle(rng, 1:nrow(prcl_df))[1:dqn_param.n_episodes_eval]
    steps = rand(1:input_struct.n_years, dqn_param.n_episodes_eval)
    migrtn_thrshlds = rand(migrtn_thrshld_dist(), dqn_param.n_episodes_eval)
    # ---

    # reading slr scenario
    slr_exposure, slr_elec_acc, slr_exit_tt, slr_utmb_tt = read_slr_exposure(input_struct, order=prcl_df.guid, rng=rng)
    
    p = ProgressBar(input_struct.n_train_iterations; color=:black)
    for iteration = 1:input_struct.n_train_iterations
        agent = setup_agent(prcl_df, rng)
        migrtn_thrshld = rand(rng, migrtn_thrshld_dist())
        agent.state[7] = sample_weights(rng)            # cost to elevate
        agent.state[8] = sample_weights(rng)            # cost for electric backup

        # ---
        update_state!(agent, 
                      2025, 
                      pos_to_guid_idx, 
                      slr_exposure, 
                      slr_elec_acc, 
                      slr_exit_tt, 
                      slr_utmb_tt, 
                      migrtn_thrshld, 
                      rng)


        for i = 1:75
            update_prev_state!(agent)
            decide_action_train!(agent, rng, dqn_param, active_q)
            # check_age!(agent)
            year = 2025+i

            update_state!(agent, 
                          year, 
                          pos_to_guid_idx, 
                          slr_exposure, 
                          slr_elec_acc, 
                          slr_exit_tt, 
                          slr_utmb_tt, 
                          migrtn_thrshld, 
                          rng)

            evaluate_action_train!(
                                agent, 
                                weights, 
                                costs, 
                                replay,
                                dqn_param,
                                active_q,
                                target_q,
                                rng
                                )
            (agent.action == :leave) && (break)
        end
        # println()
        next!(p)

        (iteration%dqn_param.save_ntwk_freq==0) && (save_networks(path_to_training_dir, iteration, active_q))      # saving episode network every N episodes
        if iteration%dqn_param.eval_freq==0
            evaluate_network!(prcl_df,
                              idxs,
                              steps,
                              migrtn_thrshlds,
                              pos_to_guid_idx,
                              active_q,
                              dqn_param,
                              input_struct,
                              slr_exposure,
                              slr_elec_acc,
                              slr_exit_tt,
                              slr_utmb_tt,
                              rng
                              )
        end
        (iteration%dqn_param.write_freq==0) && (save_train_log(dqn_param, "test", iteration, input_struct, replay, path_to_training_dir))
        dqn_param.eps = epsilon_decay(iteration, dqn_param.eps_max, dqn_param.eps_min, dqn_param.n_steps_decay)
    end
end

function check_mem()
    if Sys.free_memory()/2^30 < 6.0
        GC.gc()
    end
end


function update_prev_state!(agent::ResidentialAgent)
    agent.state_prev = copy(agent.state)
    agent.action = :nothing
end

function check_nghbr_migrate(slr_exposure::DataFrame,
                             slr_elec_acc::DataFrame, 
                             slr_exit_tt::DataFrame, 
                             slr_utmb_tt::DataFrame, 
                             prcl_i::Int64, 
                             year_string::String,
                             migrtn_thrshld::Float64,
                             rng::MersenneTwister)

    exps_ = get_prcl_info(slr_exposure, prcl_i, year_string)/365     # getting slr exposure of parcel
    elec_ = get_prcl_info(slr_elec_acc, prcl_i, year_string)/365     # getting slr exposure of parcel
    exit_ = get_prcl_info(slr_exit_tt,  prcl_i, year_string)/365     # getting slr exposure of parcel
    utmb_ = get_prcl_info(slr_utmb_tt,  prcl_i, year_string)/365     # getting slr exposure of parcel
    (exps_>1) && (exps_=1)                                                      # if greater than 1, set to 1
    (elec_>1) && (elec_=1)                                                      # if greater than 1, set to 1
    (exit_>1) && (exit_=1)                                                      # if greater than 1, set to 1
    (utmb_>1) && (utmb_=1)                                                      # if greater than 1, set to 1
                                                
    # e_ = max(exit_, utmb_)
    e_ = max(exps_, elec_, exit_, utmb_)
    if e_ > migrtn_thrshld                                  # assuming random threshold during training
        return true
    else
        return false
    end
end


function update_state!(agent::ResidentialAgent, 
                       year::Int64,
                       pos_to_guid_idx::Dict,
                       slr_exposure::DataFrame,
                       slr_elec_acc::DataFrame,
                       slr_exit_tt::DataFrame,
                       slr_utmb_tt::DataFrame,
                       migrtn_thrshld::Float64,
                       rng::MersenneTwister
                       )
    # agent.age += 1
    nearby_pos = nearby_positions(agent.pos, agent.view_radius)
    year_string = "_$(year)"
    n_migrate = 0f0
    cnt_nghbrs = 0f0
    for near_pos in nearby_pos
        if haskey(pos_to_guid_idx, near_pos)                              # if position has a parcel in it
            e = 0f0                                                             # pre-allocating exposure; 0f0 is a 0.0 in Float32
            prcl_indices = pos_to_guid_idx[near_pos]                      # get prcl_df indices of parcels in cell
            for prcl_i in prcl_indices                                          # loop through parcels in the cell
                nghbr_mgrt = check_nghbr_migrate(slr_exposure, slr_elec_acc, slr_exit_tt, slr_utmb_tt, prcl_i, year_string, migrtn_thrshld, rng)
                (nghbr_mgrt) && (n_migrate += 1f0)
                cnt_nghbrs += 1f0
            end
        end
    end

    ## updating state of position itself
    # first exposure
    e = get_prcl_info(slr_exposure, agent.pos_idx, year_string)
    e = e/365
    (e>1) && (e=1)
    state_exposure = convert(Float32, e)

    # now electric
    e = get_prcl_info(slr_elec_acc, agent.pos_idx, year_string)
    e = e/365
    (e>1) && (e=1)
    state_electric = convert(Float32, e)

    # then transportation
    e = get_prcl_info(slr_exit_tt, agent.pos_idx, year_string)
    u = get_prcl_info(slr_utmb_tt, agent.pos_idx, year_string)
    tt = max(e, u)
    tt = tt/365
    (tt>1) && (tt=1)
    state_transportation = convert(Float32, tt)

    agent.state[1] = n_migrate/cnt_nghbrs
    (cnt_nghbrs==0f0) && (agent.state[1]=0f0)
    agent.state[2] = state_exposure
    agent.state[3] = state_electric
    agent.state[4] = state_transportation

    agent.state[5] = copy(agent.state_prev[5])              # initialize with previous state for elevated
    agent.state[6] = copy(agent.state_prev[6])              # initialize with previous state for generator
    # agent.state[7] = convert(Float32, agent.age/80)         # update agent age

    (agent.action==:elevate) && (agent.state[5] = 1f0)      # if action is elevate, then update state[5] (elevated)
    (agent.state[5]==1f0)    && (agent.state[2] = 0f0)      # and if elevated, then no exposure

    (agent.action==:generator)   && (agent.state[6] = 1f0)      # if action is generator, then update state[6] (generator installed)
    (agent.state[6]==1f0)    && (agent.state[3] = 0f0)      # and if generator, then no electricity outages

    (agent.action == :leave) && (agent.state = State(zeros(Float32, size(State)))) # if agent leaves, then sp = zeros
end


function evaluate_action_train!(agent::ResidentialAgent, 
                                weights::Vector{Float32}, 
                                costs::Vector{Float32}, 
                                replay::MyPrioritizedReplayBuffer,
                                dqn_param::DQNParam,
                                active_q::Chain,
                                target_q::Chain,
                                rng::MersenneTwister
                                )
    s = copy(agent.state_prev)                                                  # agent's original state
    ai = agent.action_indices[agent.action]                                     # action index; for logging in replay buffer

    (agent.action==:leave)  && (agent.state=State(zeros(Float32, size(State)))) # if agent is leaving, reset next state to 0's
    rew = reward_func(agent.state_prev, 
                      agent.action, 
                      weights,
                      costs,
                      )
    agent.reward += rew

    done = false
    (agent.action==:leave) && (done=true)                                       # if agent leaves, then done flag is true; note agent's action=:leave if age is 80
    sp = copy(agent.state)                                                      # next state as a result of taking the action

    exp = MyDQExperience(s, ai, Float32(rew), sp, done)                         # setting experience in tuple
    dqn_train_step!(replay, 
                    dqn_param,
                    exp,
                    active_q, 
                    target_q,
                    rng
                    )

end

function dqn_train_step!(
                        replay::MyPrioritizedReplayBuffer, 
                        dqn_param::DQNParam,
                        exp::MyDQExperience,
                        active_q::Chain, 
                        target_q::Chain,
                        rng::MersenneTwister
                        )
    optimizer = Adam(dqn_param.learning_rate)
    if dqn_param.prioritized_replay
        add_exp!(replay, exp, abs(exp.r))
    else
        add_exp!(replay, exp, 0f0)
    end

    dqn_param.episode_reward_log[end] += exp.r
    if (dqn_param.train_steps[end]%dqn_param.train_freq == 0) && (replay._curr_size > replay.batch_size) # todo: drs added this; only start training after replay buffer is larger than batchsize
        hs = hiddenstates(active_q)
        loss_val, grad_val = batch_train!(optimizer, active_q, target_q, replay, dqn_param)
        sethiddenstates!(active_q, hs)
        dqn_param.loss_log[end] += loss_val
    end
    if dqn_param.train_steps[end]%dqn_param.target_update_freq == 0
        weights = Flux.params(active_q)
        Flux.loadparams!(target_q, weights)
    end
    dqn_param.train_steps[end] += 1
end


"""
    batch_train!()
batch training network; taken from DeepQLearning.jl
"""
function batch_train!(
                      optimizer,
                      active_q::Chain, 
                      target_q::Chain,
                      replay::MyPrioritizedReplayBuffer,
                      DQNParam::DQNParam;
                     )
    s_batch, a_batch, r_batch, sp_batch, done_batch, indices, importance_weights = sample(replay)
    p = Flux.params(active_q)

    loss_val = nothing
    td_vals = nothing

    γ = DQNParam.discount
    if DQNParam.double_q
        qp_values = active_q(sp_batch)
        target_q_values = target_q(sp_batch)
        best_a = [CartesianIndex(argmax(qp_values[..,i]), i) for i=1:DQNParam.batch_size]
        q_sp_max = target_q_values[best_a]
    else
        q_sp_max = dropdims(maximum(target_q(sp_batch), dims=1), dims=1)
    end
    q_targets = r_batch .+ (1f0 .- done_batch) .* γ .* q_sp_max


    # for future gpu training: slight bottle neck here, ~0.001 sec; 961 allocations
    gs = Flux.gradient(p) do
        q_values = active_q(s_batch)        
        q_sa = q_values[a_batch]        
        td_vals = q_sa .- q_targets        
        loss_val = sum(huber_loss, importance_weights.*td_vals)        
        loss_val /= DQNParam.batch_size
    end

    grad_norm = globalnorm(p, gs)
    Flux.Optimise.update!(optimizer, p, gs)

    if DQNParam.prioritized_replay
        update_priorities!(replay, indices, td_vals)
    end
    return loss_val, grad_norm
end


function reset_training_counts!(dqn_param::DQNParam)
    push!(dqn_param.episode_log, 0.0)
    push!(dqn_param.train_steps, dqn_param.train_steps[end])  # running count of number of steps
    push!(dqn_param.epsilon_log, 0.0)
    push!(dqn_param.buffer_size_log, 0.0)
    push!(dqn_param.episode_reward_log, 0.0)
    push!(dqn_param.eval_log, 0.0)
    push!(dqn_param.loss_log, 0.0)
end

function save_train_log(dqn_param::DQNParam, model_runname::String, iteration::Int64, input_struct::InputStruct, replay::MyPrioritizedReplayBuffer, path_to_training_dir::String)
    # df = DataFrame(:episode=>collect(0:dqn_param.write_freq:iteration)[2:end])
    df = DataFrame()

    dqn_param.episode_log[end] = iteration
    dqn_param.epsilon_log[end] = dqn_param.eps
    dqn_param.buffer_size_log[end] = replay._curr_size

    df[!,Symbol("episode")]                = dqn_param.episode_log
    df[!,Symbol("train_steps_family1")]    = dqn_param.train_steps
    df[!,Symbol("epsilon_family1")]        = dqn_param.epsilon_log
    df[!,Symbol("buffer_size_family1")]    = dqn_param.buffer_size_log
    df[!,Symbol("episode_reward_family1")] = dqn_param.episode_reward_log
    df[!,Symbol("eval_reward_family1")]    = dqn_param.eval_log
    df[!,Symbol("loss_family1")]           = dqn_param.loss_log

    write_train(df, "$(input_struct.sub_runname)-solo-training-results.csv", path_to_training_dir)     # writing trainlog every N episodes    
    
    reset_training_counts!(dqn_param)

end

function ProgressBar(iters; color=:cyan, description="Status")
    p = Progress(iters, 
            desc="$(description): ",
            # desc="Iteration: $(i)/$(iters) | ",
            barlen=30, 
            color=color
        )
    return p
end

function evaluate_network!(prcl_df::DataFrame,
                            idxs::Vector{Int64},
                            steps::Vector{Int64},
                            migrtn_thrshlds::Vector{Float64},
                            # n_rows_eval::Int64, 
                            pos_to_guid_idx::Dict, 
                            active_q::Chain,
                            dqn_param::DQNParam,
                            input_struct::InputStruct,
                            slr_exposure::DataFrame,
                            slr_elec_acc::DataFrame,
                            slr_exit_tt::DataFrame,
                            slr_utmb_tt::DataFrame,
                            rng::MersenneTwister
                            )
    weights  = get_agent_weights(Symbol(input_struct.sub_runname), ResidentialAgent)
    costs    = get_agent_costs(Symbol(input_struct.sub_runname), ResidentialAgent)
    Qs = zeros(Float32, length(idxs))
    for i = 1:length(idxs)
        idx = idxs[i]
        step = steps[i]
        migrtn_thrshld = migrtn_thrshlds[i]

        year = 2025 + step

        p = prcl_df[idx,:]                              # parcel information
        agent = setup_agent(p, rng)
        update_state!(agent, year, pos_to_guid_idx, slr_exposure, slr_elec_acc, slr_exit_tt, slr_utmb_tt, migrtn_thrshld, rng)
        update_prev_state!(agent)

        q = active_q(agent.state_prev)
        Qs[i] = maximum(q)
    end
    dqn_param.eval_log[end] = mean(Qs)
end

function load_network(fn::String, agent_type::DataType, rng::MersenneTwister)
    model_state = BSON.load(fn)[:qnetwork]
    m = define_network(agent_type, rng)
    Flux.loadparams!(m, model_state)
    return m
end


"""
    parse_commandline()
Function to pass option values into the model from the command line
Created to start simulation from user-specified iteration
"""
function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--start_sim"
            help = "option to start simulation at iteration other than 1"
            default = 1
            arg_type = Int
            required = false
        "--end_sim"
            help = "option to end simulation at iteration other than 1"
            default = nothing
            arg_type = Int
            required = false
        "--model_runname"
            help = "option to run specific model runname"
            default = nothing
            arg_type = String
            required = false
        "--train"
            help = "option to run under training conditions or not"
            default = nothing
            arg_type = Bool
            required = false
        "--sub_runname"
            help = "which sub_runname to run"
            default = nothing
            arg_type = String
            required = false
        "--output_dir"
            help = "where to write output"
            default = nothing
            arg_type = String
            required = false
        "--training_dir"
            help = "where to write training results"
            default = nothing
            arg_type = String
            required = false
    end
    return parse_args(s)
end



"""
    check_run_previously_done(input_struct, parsed_args)
todo: temporarily checking if run has been completed.
"""
function check_run_previously_done(input_struct, parsed_args)
    path_to_training_dir = pwd()
    (parsed_args["training_dir"] != nothing) && (path_to_training_dir=parsed_args["training_dir"])
    path_to_training_dir = joinpath(path_to_training_dir, "model-runs", input_struct.model_runname, "training-results", input_struct.sub_runname)
    (~isdir(path_to_training_dir)) && (return false)         # if directory does not exist, then run has not been done; return
    ("episode_1000000" in readdir(path_to_training_dir)) && (return true)
    return false
end

run_subnames = ["agent1",
                "agent2",
                "agent3",
                "agent4",
                "agent5",
                "agent6",
                "agent7",
                "agent8",
]
Threads.@threads for run_subname in run_subnames
    train("status-quo", run_subname)
end













