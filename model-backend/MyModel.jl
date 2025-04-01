# file for model struct and operations


Base.@kwdef mutable struct Parameters
    # model values
    input_struct::InputStruct
    input_dir::String
    output_dir::String
    training_dir::String
    tick::Int64 = 1
    
    train::Bool
    iteration::Int64 = 1
    n_iterations::Int64

    n_years::Int64
    slr_scenario::String
    slr_exposure::DataFrame
    slr_exit_tt::DataFrame
    slr_utmb_tt::DataFrame
    slr_elec_acc::DataFrame

    year::Int64
    start_year::Int64
    # progress_bar::Progress
    n_prcls::Int64
    prcl_df::DataFrame
    ResidentialAgent_Dict::Dict
    pos_to_guid_idx::Dict{Tuple{Int64, Int64}, Vector{Int64}}
    
    n_occupied::Int64=0
    n_unoccupied::Int64=0
    
    n_elevated_occupied::Int64=0
    n_elevated_unoccupied::Int64=0
    
    n_generator_occupied::Int64=0
    n_generator_unoccupied::Int64=0
    
    n_elevated_generator_occupied::Int64=0
    n_elevated_generator_unoccupied::Int64=0
end


"""
    initialize(path_to_input_dir; seed, iter)
Sets up the initial model using the input directory
Sets up the parcel space
Assigns agents to the model
"""
function initialize(path_to_input_dir::String, input_struct::InputStruct, 
                    parsed_args::Dict; seed=1337, iter=1)
    seed = seed + iter
    rng = Random.MersenneTwister(seed)   # setting seed

    # preparing parcel dataframe
    cell_size = 25
    prcl_df = prepare_parcel_df(path_to_input_dir, seed=seed, cell_size=cell_size, rotate=true)

    # setting up ABM space
    gridsize = (div(47_000,cell_size), div(7_000,cell_size))
    space = GridSpace(gridsize; periodic=false)       # galveston is approx. 47,000m by 7,000m

    # getting model property dictionary
    properties = setup_model_properties(input_struct, parsed_args, prcl_df, iter, path_to_input_dir, rng)

    # setting up the model
    model = ABM(
                ResidentialAgent,
                space;
                rng=rng,
                properties=properties,
                model_step! =model_step!,  # note: a space is needed after "model_step!", otherwise this is interpreted as "model_step !="
                )


    # --- adding agents to model; either in parcel or general environment
    AddAgentsToModel!(model, prcl_df)
    SetupAgentDict!(model, ResidentialAgent)
    # (train==true) && (update_weights_json(model)) #TODO; temporarily commenting this out for testing
    UpdateModelCounts!(model)
    return model
end


function CleanUpDQNParam!(dqn_param)
    dqn_param.episode_log = Float64[0.0]
    dqn_param.train_steps = Int64[0]
    dqn_param.epsilon_log = [dqn_param.episode_log[end]]
    dqn_param.buffer_size_log = [dqn_param.buffer_size_log[end]]
    dqn_param.episode_reward_log = [0.0]
    dqn_param.eval_log = Float64[0.0]
    dqn_param.loss_log = Float64[0.0]

end


"""
    setup_model_properties(prcl_df::DataFrame, iter::Int64)
setting up model properties that go into the ABM.
prcl_df: parcel dataframe that was previously prepared
iter: iteration number in outer ABM cycle
"""
function setup_model_properties(input_struct::InputStruct, parsed_args::Dict,
                                prcl_df::DataFrame, iter::Int64, path_to_input_dir::String, 
                                rng::AbstractRNG)

    # progress_bar = ProgressBar(iter, input_struct.n_model_iterations, input_struct.n_years)
    # update!(progress_bar)

    slr_exposure, slr_elec_acc, slr_exit_tt, slr_utmb_tt = read_slr_exposure(input_struct, order=prcl_df.guid, rng=rng)
    pos_to_guid_idx = setup_pos2guididx(prcl_df)

    # setting up output_dir, checking parsed_args
    path_to_output_dir = pwd()
    (parsed_args["output_dir"]!= nothing) && (path_to_output_dir=parsed_args["output_dir"])
    path_to_output_dir = joinpath(path_to_output_dir, "model-runs", input_struct.model_runname, "output", input_struct.sub_runname)

    # setting up path to training dir, checking parsed_args
    path_to_training_dir = pwd()
    (parsed_args["training_dir"] != nothing) && (path_to_training_dir=parsed_args["training_dir"])
    path_to_training_dir = joinpath(path_to_training_dir, "model-runs", input_struct.model_runname, "training-results") #, input_struct.sub_runname)

    makedir(path_to_output_dir)
    makedir(path_to_training_dir)

    # ----------------------------------------
    properties = Parameters(
        input_struct=input_struct,
        input_dir=path_to_input_dir,
        output_dir=path_to_output_dir,
        training_dir=path_to_training_dir,
        train=input_struct.train,
        n_iterations=input_struct.n_iterations,
        n_years=input_struct.n_years,
        slr_scenario=input_struct.slr_scenario,
        slr_exposure=slr_exposure,
        slr_exit_tt=slr_exit_tt,
        slr_utmb_tt=slr_utmb_tt,
        slr_elec_acc=slr_elec_acc,
        year=input_struct.model_start_year,
        start_year=input_struct.model_start_year,
        # progress_bar=progress_bar,
        n_prcls=size(prcl_df, 1),
        prcl_df=prcl_df,
        ResidentialAgent_Dict=Dict(),       # empty placeholder for now, gets populated later
        pos_to_guid_idx=pos_to_guid_idx,
        )
    return properties
end


"""
    setup_pos2guididx(prcl_df::DataFrame)
setting up a dictionary to map pos (tuple) to row in prcl_df
e.g., 
    if pos_to_guid_idx[(563, 124)] = 1352, then this means that in position 
      (563, 124) is the guid/parcel that corresponds to model.prcl_df[1352,:].
"""
function setup_pos2guididx(prcl_df::DataFrame)
    # pos_to_guid_idx = Dict()
    pos_to_guid_idx = Dict{Tuple{Int64, Int64}, Vector{Int64}}()
    for pos in unique(prcl_df.pos)
        p_ = prcl_df[findall(==(pos),prcl_df.pos),:]
        pos_to_guid_idx[pos] = p_.row_idx
    end
    return pos_to_guid_idx
end

"""
    reset!(model::ABM, i::Int64)
resets the model during training; 
removes all agents then adds new ones
resets model properties (prcl_df, tick, year)
resets progress bar
"""
function reset!(model::ABM, i::Int64)
    model.iteration += 1                        # updating model iteration counter
    
    reset_model!(model)
    # model.progress_bar = ProgressBar(i+1, model.n_iterations, model.n_years)  # resetting
    
    for family in available_agent_families(ResidentialAgent)        
        push!(model.ResidentialAgent_Dict[family][:DQNParam].episode_log, i+1)
        
        push!(model.ResidentialAgent_Dict[family][:DQNParam].train_steps, model.ResidentialAgent_Dict[family][:DQNParam].train_steps[end])  # running count of number of steps
        push!(model.ResidentialAgent_Dict[family][:DQNParam].epsilon_log, 0.0)
        push!(model.ResidentialAgent_Dict[family][:DQNParam].buffer_size_log, 0.0)
        push!(model.ResidentialAgent_Dict[family][:DQNParam].episode_reward_log, 0.0)
        push!(model.ResidentialAgent_Dict[family][:DQNParam].eval_log, 0.0)
        push!(model.ResidentialAgent_Dict[family][:DQNParam].loss_log, 0.0)
        
        model.ResidentialAgent_Dict[family][:training_df][!,Symbol("reward_$(model.iteration)")] .= 0f0
    end

    # update_epsilon!(model)
    # update!(model.progress_bar)    
end


function reset_model!(model::ABM)
    remove_all!(model)                          # removing all agents from model
    AddAgentsToModel!(model, model.prcl_df)     # adding agents to model
    UpdateModelCounts!(model)                   # updating model counts

    model.slr_exposure, model.slr_elec_acc, model.slr_exit_tt, model.slr_utmb_tt = read_slr_exposure(model.input_struct, order=model.prcl_df.guid, rng=abmrng(model))    # getting random slr scenario
    model.prcl_df[!,:occupied] = ones(size(model.prcl_df,1))    # resetting prcl_df to be occupied
    model.tick = 1                              # resetting model tick
    model.year = model.start_year               # resetting model year
end

function update_percent_training_agents!(model)
    model.input_struct.percent_training_agents = p_training_agents(model.iteration, 
                                                                 model.input_struct.percent_training_agents_end, 
                                                                 model.input_struct.percent_training_agents_start, 
                                                                 model.input_struct.n_steps_decay_training_agents
                                                                 )
end

function update_epsilon!(model::ABM)
    for family in available_agent_families(ResidentialAgent)
        eps = epsilon_decay(
                            model.iteration, 
                            model.ResidentialAgent_Dict[family][:DQNParam].eps_max, 
                            model.ResidentialAgent_Dict[family][:DQNParam].eps_min, 
                            model.ResidentialAgent_Dict[family][:DQNParam].n_steps_decay)
        model.ResidentialAgent_Dict[family][:DQNParam].eps = eps
    end
end

function load_network(fn::String, agent_type::DataType, model::ABM)
    model_state = BSON.load(fn)[:qnetwork]
    m = define_network(agent_type, abmrng(model))
    Flux.loadparams!(m, model_state)
    return m
end

"""
    save_networks(model::ABM, iter::Int)
function to save the networks for agent to bson file
"""
function save_networks(model::ABM, iter::Int)    
    path_out = joinpath(model.training_dir, "model_episode_$(iter)")
    makedir(path_out)
    for ntwk_num in 1:model.input_struct.n_train_agents
        active_q = model.ResidentialAgent_Dict[:family1][Symbol("ntwk_$(ntwk_num)")][:active_q]
        fn = joinpath(path_out, "qnetwork-residential-agent-$(ntwk_num).bson")
        bson(fn, qnetwork=[w for w in Flux.params(active_q)])
    end
end

function save_networks(path_to_training_dir::String, iter::Int, active_q::Chain)
    path_out = joinpath(path_to_training_dir, "episode_$(iter)")
    makedir(path_out)
    fn = joinpath(path_out, "qnetwork-residential-agent-1.bson")
    bson(fn, qnetwork=[w for w in Flux.params(active_q)])
end


function evaluate_network(model::ABM, iter::Int)
    model.input_struct.train = false
    reset_model!(model)
    # model.progress_bar = ProgressBar(iter, model.n_iterations, model.n_years, color=:magenta, description="Evaluation")

    for i = 1:model.n_years
        model_run_step!(model)
    end
    model.input_struct.train = true
end

"""
    save_train_log(model::ABM, model_runname::String)
saving the training results to a dataframe, then csv file
This returns the total reward collected for each episode
"""
function save_train_log(model::ABM, model_runname::String)
    # df = DataFrame(:episode=>collect(1:model.iteration))
    df = DataFrame()
    for fam in available_agent_families(ResidentialAgent)
        model.ResidentialAgent_Dict[fam][:DQNParam].epsilon_log[end] = model.ResidentialAgent_Dict[fam][:DQNParam].eps
        model.ResidentialAgent_Dict[fam][:DQNParam].buffer_size_log[end] = model.ResidentialAgent_Dict[fam][:replay]._curr_size
        df[!,Symbol("episode")]               = model.ResidentialAgent_Dict[fam][:DQNParam].episode_log
        df[!,Symbol("train_steps_$(fam)")]    = model.ResidentialAgent_Dict[fam][:DQNParam].train_steps
        df[!,Symbol("epsilon_$(fam)")]        = model.ResidentialAgent_Dict[fam][:DQNParam].epsilon_log
        df[!,Symbol("buffer_size_$(fam)")]    = model.ResidentialAgent_Dict[fam][:DQNParam].buffer_size_log
        df[!,Symbol("episode_reward_$(fam)")] = model.ResidentialAgent_Dict[fam][:DQNParam].episode_reward_log
        df[!,Symbol("eval_reward_$(fam)")]    = model.ResidentialAgent_Dict[fam][:DQNParam].eval_log
        df[!,Symbol("loss_$(fam)")]           = model.ResidentialAgent_Dict[fam][:DQNParam].loss_log
    end
    write_train(df, model, "$(model.input_struct.sub_runname)-model-training-results.csv")

    # dqn_param = DQNParam()
    # (model.iteration%dqn_param.write_freq==0) && (write_train(df, model, "$(model.input_struct.sub_runname)-training-results.csv"))      # writing trainlog every N episodes    
    # (model.iteration%dqn_param.write_freq==0) && (write_train(model.ResidentialAgent_Dict[:family1][:training_df], model, "$(model.input_struct.sub_runname)-ntwk_log.csv"))
end


"""
    AddAgentsToModel!(model::ABM, parcel_df::DataFrame)
adds agents to the model to initialize
"""
function AddAgentsToModel!(model::ABM, parcel_df::DataFrame)
    add_agents!(model, parcel_df)
    setup_initial_states!(model)
end


"""
function add_agents!(model, parcel_df)
"""
function add_agents!(model::ABM, parcel_df::DataFrame)
    setup_agent_family_weights!(model)    
    for i = 1:size(parcel_df,1)
        id = i
        i !=1 && (id=next_avail_id(model))              # if id!=1, then get the next available id
        p = parcel_df[i,:]                              # parcel information
        s = define_state(ResidentialAgent)
        actions = define_actions(ResidentialAgent)
        action_indices = define_action_indices(ResidentialAgent)
        action_weights = define_action_weights(ResidentialAgent)
        r = 5
        center_pos = (r+1, r+1)

        # sampling family that agent belongs to
        family = StatsBase.wsample(abmrng(model), model.input_struct.agent_families, model.input_struct.agent_family_weights)
        age = age_calc(model)
        
        agent = ResidentialAgent(
                            id=id,                          # agent id (required)
                            pos=p.pos,                      # agent pos as tuple (required)
                            pos_guid=p.guid,                # agent's current  position (guid)
                            pos_idx=p.row_idx,                  # pos_idx; position index - used to quickly look up from prcl_df
                            state_prev=s,                   # previous state
                            state=copy(s),                        # initializing with in parcel, unexposed
                            actions=actions,                # actions for agent to take
                            action_indices=action_indices,  # action indicies
                            action_weights=action_weights,  # action weights for epsilon greedy;
                            action=:nothing,                # action
                            family=family,                  # agent family
                            view_radius=r,                  # agent view radius
                            center_pos=center_pos,          # center position
                            neighborhood_original=zeros(MArray{Tuple{11,11},Float32}), # number of neighbors; gets updated soon
                            age=age,                        # agent age
                            reward=0f0,                     # cumulative reward
                            p_migr=0f0,                     # percent of neighbors migrated
                            p_expd=0f0,                     # percent of year exposed
                            p_elec=0f0,                     # percent of year with elec. outage
                            p_trns=0f0,                     # percent of year with increase travel time
                            q_ntng=0f0,                     # q-value no action
                            q_leav=0f0,                     # q-value leaave
                            q_elev=0f0,                     # q-value elevate
                            q_gnrt=0f0,                     # q-value generator
                            )
        add_agent!(agent, agent.pos, model)
    end
end

"""
    setup_agent_family_weights!(model::ABM)
setting up weights that agents' families are sampled from. 
Only does this if not pre-defined in InputStruct.jl
"""
function setup_agent_family_weights!(model::ABM)

"""
test for number of samples
d = maximum(rand(Dirichlet(N,1.0), n_samples), dims=2)
    N = number of agent families
    1.0 = parameter in Dirichlet; do not change
    n_samples = number of samples
"""
    (~isempty(model.input_struct.agent_family_weights)) && (return)
    #--- using Dirichlet distribution
    model.input_struct.agent_family_weights = rand(abmrng(model), Dirichlet(length(model.input_struct.agent_families), 1.0))
    # #--- using non-uniform weights
    # model.input_struct.agent_family_weights = rand(abmrng(model), length(model.input_struct.agent_families))
    # model.input_struct.agent_family_weights = model.input_struct.agent_family_weights ./ (sum(model.input_struct.agent_family_weights))
end

"""
    setup_initial_states!(model)
gets each agent's initial state
updates agent.state, counts number of neighbors at model start;
this done is after adding all agents to the model; that way agents see who's around them
"""
function setup_initial_states!(model)
    for id in random_ids(model)
        model[id].neighborhood_original = get_neighborhood(model[id], model)    # get each agent's original neighborhood before any migration occurs
        update_state!(model[id], model)                                         # setting up the agent's initial state
    end
end

# function SetupModelNetworks!(model::ABM, train::Bool)
#     if train
#         model.ResidentialAgent_Dict = setup_agent_training_dict(model, ResidentialAgent)
#     else
#         model.ResidentialAgent_Dict = setup_agent_dict(model, ResidentialAgent)
#     end
# end

# function SetupAgentTrainingDict!(model::ABM, agent_type::DataType, replay::MyPrioritizedReplayBuffer, dqn_param::DQNParam)
#     action_indices = define_action_indices(agent_type)
#     agent_families = available_agent_families(agent_type)

#     ResidentialAgent_Dict = Dict()
#     for family in agent_families
#         # network = define_network(agent_type, model)
#         weights = get_agent_weights(model.input_struct, ResidentialAgent)
#         costs = get_agent_costs(model.input_struct, ResidentialAgent)
#         training_df = setup_training_df(model)
#         ResidentialAgent_Dict[family] = Dict(
#                                             :replay=>replay,                    # replay buffer
#                                             :weights=>weights,                  # weights in reward function
#                                             :costs=>costs,                      # costs for elevating/generator
#                                             :DQNParam=>dqn_param,               # DQN parameters
#                                             :training_df=>training_df
#                                             )
#         for ntwk_num in 1:model.input_struct.n_train_agents
#             fn = joinpath(model.training_dir, "episode_$(model.input_struct.n_train_iterations)", "qnetwork-residential-agent-1.bson")
#             active_q = load_network(fn, agent_type, model)

#             ResidentialAgent_Dict[family][Symbol("ntwk_$(ntwk_num)")] = Dict()
#             ResidentialAgent_Dict[family][Symbol("ntwk_$(ntwk_num)")][:active_q] = deepcopy(active_q)
#             ResidentialAgent_Dict[family][Symbol("ntwk_$(ntwk_num)")][:target_q] = deepcopy(active_q)
#         end
#     end
#     model.ResidentialAgent_Dict = ResidentialAgent_Dict
# end


# function setup_training_df(model::ABM)
#     df = DataFrame()
#     df[!,:ntwk_num] = 1:model.input_struct.n_train_agents
#     df[!,:train_steps] .= 0
#     df[!,:reward_1] .= 0f0

#     return df
# end

"""
    SetupAgentDict!(model::ABM, agent_type::DataType)
sets up dictionary that keeps agent qnetwork, weights, and costs.
Do this in dictionary so that agents that have the same decision network (e.g., 
  all agents with network 1) use the same actual network. This is done as opposed 
  to storing the network in each ResidentialAgent struct, thus reducing memory
  (e.g., ~10 networks, as opposed to ~20,000 networks)

"""
function SetupAgentDict!(model::ABM, agent_type::DataType)
    if model.input_struct.agent_network_episode=="none"
        model.input_struct.agent_network_episode = get_bson_episode(model.input_struct, agent_type)
    end

    ResidentialAgent_Dict = Dict()
    for family in model.input_struct.agent_families
        # dqn_param = DQNParam()
        weights = get_agent_weights(family, ResidentialAgent)
        costs = get_agent_costs(family, ResidentialAgent)

        fn = joinpath(model.training_dir, String(family), "episode_$(model.input_struct.agent_network_episode)", "qnetwork-residential-agent-$(1).bson") #todo: temporarily set to one
        active_q = load_network(fn, agent_type, model)

        ResidentialAgent_Dict[family] = Dict(
                                            :weights=>weights,                  # weights in reward function
                                            :costs=>costs,                      # costs for elevating/generator
                                            # :DQNParam=>dqn_param,
                                            :active_q=>active_q
                                            )
    end
    model.ResidentialAgent_Dict = ResidentialAgent_Dict
end

"""
    read_slr_exposure(input_struct)
reads slr exposure data for the slr scenario in input.csv
The slr exposure data was pre-computed and is a CSV file containing each guid (rows)
and year of exposure (columns)
"""
function read_slr_exposure(input_struct; order::Vector{String}, rng::AbstractRNG)
    sc = input_struct.slr_scenario
    ne = input_struct.slr_ne
    path_to_slr_scns = joinpath(pwd(), "input", "slr-scenarios")
    if (sc == "train")        # no scenario given, so sample scenario
        path_to_exposure = joinpath(path_to_slr_scns, "building-exposure")
        files = readdir(path_to_exposure)
        filter!(e->e≠".DS_Store",files)
        file = rand(rng, files)
        file_end = split(file, "years_")[2]
        path_to_exposure = joinpath(path_to_exposure, file)
        path_to_elec = joinpath(path_to_slr_scns, "electricity-access",  join(["nNoAccess_years_", file_end]))
        path_to_exit_tt = joinpath(path_to_slr_scns, "galveston-exit",  join(["nTTIncrease_years_", file_end]))
        path_to_utmb_tt = joinpath(path_to_slr_scns, "utmb-hospital",  join(["nTTIncrease_years_", file_end]))
    else
        path_to_exposure = joinpath(path_to_slr_scns, "building-exposure", "nTimesExp_years_sc$(sc)_ne$(ne).csv")
        path_to_elec = joinpath(path_to_slr_scns, "electricity-access", "nNoAccess_years_sc$(sc)_ne$(ne).csv")
        path_to_exit_tt = joinpath(path_to_slr_scns, "galveston-exit", "nTTIncrease_years_sc$(sc)_ne$(ne).csv")
        path_to_utmb_tt = joinpath(path_to_slr_scns, "utmb-hospital", "nTTIncrease_years_sc$(sc)_ne$(ne).csv")
    end

    slr_exposure = read_csv(path_to_exposure)
    slr_elec_acc = read_csv(path_to_elec)
    slr_exit_tt = read_csv(path_to_exit_tt)
    slr_utmb_tt = read_csv(path_to_utmb_tt)
    
    slr_exposure = slr_exposure[indexin(order, slr_exposure.guid),:]
    slr_elec_acc = slr_elec_acc[indexin(order, slr_elec_acc.guid),:]
    slr_exit_tt  = slr_exit_tt[indexin(order, slr_exit_tt.guid),:]
    slr_utmb_tt  = slr_utmb_tt[indexin(order, slr_utmb_tt.guid),:]

    # converting each column from Float64 to Int64
    convert_cols_int!(slr_exposure)
    convert_cols_int!(slr_elec_acc)
    convert_cols_int!(slr_exit_tt)
    convert_cols_int!(slr_utmb_tt)
    return slr_exposure, slr_elec_acc, slr_exit_tt, slr_utmb_tt
end



"""
    model_step!(model)
function for custom model step used during running (non-training).
checks time in model.
"""
function model_step!(model::ABM)
    if model.train==true
        model_train_step!(model)
    elseif model.train==false
        model_run_step!(model)
    end
end

"""
    model_train_step!(model)
function for custom model step used during training.
"""
function model_train_step!(model::ABM)
    if model.tick < model.n_years         # pre-hazard        
        AllAgentsUpdatePrevState!(model)    # Observe initial state; called previous state
        AllAgentsDecideAction_Train!(model)  # Decide action based on previous state 
        AllAgentsAct!(model)                # perform action based on previous state
        AllAgentsUpdateState!(model)        # oberve new state based on performing action
        AllAgentsEvaluateAction_Train!(model)     # evaluate prev_state, action, reward, state; save to replay buffer; train network

    else
        AllAgentsCloseStep!(model)
    end
    UpdateModelCounts!(model)
    model.tick += 1
    model.year += 1
    # next!(model.progress_bar)   # advancing progress bar;
end



"""
    model_run_step!(model::ABM)
function for custom model step used during running
"""
function model_run_step!(model::ABM)
    if model.tick < model.n_years
        AllAgentsUpdatePrevState!(model)    # updating agents previous state to state from last step
        AllAgentsDecideAction!(model)       # all agents first decide their action
        AllAgentsAct!(model)                # all agents act on above action
        AllAgentsUpdateState!(model)        # all agents upade their state; e.g., they see what their neighbors did
        AllAgentsEvaluateActionRun!(model)  # all agents evaluate their action; e.g., adding 
        # println()

    else
        AllAgentsCloseStep!(model)
    end
    UpdateModelCounts!(model)
    model.tick += 1
    model.year += 1
    # next!(model.progress_bar)
end


"""
    AllAgentsUpdatePrevState!(model::ABM)
all agents update their previous state to be the state from the prior step
"""
function AllAgentsUpdatePrevState!(model::ABM)
    Threads.@threads for agent in random_agents(model)
    # for agent in random_agents(model)
        update_prev_state!(agent, model)
    end
end

# """
#     AllAgentsDecideAction_Train!(model)
# all agents decide their action based on the previous state using episilon greedy strategy
# """
# function AllAgentsDecideAction_Train!(model::ABM)
#     # Threads.@threads for agent in random_agents(model)
#     for agent in random_agents(model)
#         if agent.agent_decision_type == :dqn
#             decide_action!(agent, model, model.ResidentialAgent_Dict; train=true)        # based on prev state; epsilon greedy
#         else
#             active_q = model.ResidentialAgent_Dict[agent.family][Symbol("ntwk_$(agent.ntwk_num)")][:active_q]
#             decide_action!(agent, active_q)
#             # utility_theory!(agent, model)
#         end
#         # check_age!(agent)
#     end
# end


"""
    AllAgentsDecideAction!(model::ABM)
all agents decide their action based on previous state; use trained network
"""
function AllAgentsDecideAction!(model::ABM)
    Threads.@threads for agent in random_agents(model)
    # for agent in random_agents(model)
        active_q = model.ResidentialAgent_Dict[agent.family][:active_q]
        decide_action!(agent, active_q) # deciding action based curent state; epsilon greedy
        # check_age!(agent)      # todo: temporary comment
    end
end

"""
    AllAgentsAct!(model::ABM)
all agents perform their decided action
"""
function AllAgentsAct!(model::ABM)
    for agent in random_agents(model)
        act!(agent, model)                  # perform action
    end
end

"""
    AllAgentsUpdateState!(model::ABM)
all agents update their state; see what their neighbors did
"""
function AllAgentsUpdateState!(model::ABM)
    Threads.@threads for agent in random_agents(model)
    # for agent in random_agents(model)
        update_state!(agent, model)         # observes new state; both current and neighbors 
    end
end

"""
    AllAgentsEvaluateAction_Train!(model::ABM)
all agents evaluate the action they took; in training, these get passed into a
  tuple that is (prev_state, action, reward, state).
"""
function AllAgentsEvaluateAction_Train!(model)
    for agent in random_agents(model)
        if agent.agent_decision_type == :dqn
            evaluate_action!(agent, model)       # evaluates actions and occasionally trains DQN
        else
            (agent.action==:leave) && (remove_agent!(agent, model))             # if agent has left using utility theory, remove from model
        end
    end    
end

"""
    AllAgentsEvaluateActionRun!(model::ABM)
all agents evaluate the action they took; records reward signal
"""
function AllAgentsEvaluateActionRun!(model::ABM)
    for agent in random_agents(model)
        evaluate_run_action!(agent, model)
    end
end


"""
    random_ids(model::ABM)
returns random ids from the model
"""
function random_ids(model::ABM)
    ids = collect(allids(model))         # getting ids of agents
    ids = shuffle(abmrng(model), ids)    # shuffling ids
    return ids
end

"""
    random_agents(model::ABM)
returns random agents from the model
"""
function random_agents(model::ABM)
    ids = collect(allids(model))
    ids = shuffle(abmrng(model), ids)
    agents = [model[id] for id in ids]
    return agents
end


"""
    age_calc(dist::Distribution, model::ABM)
returns an age for head of household; limited to be older than 18;
draws from distribution setup in model dictionary
"""
function age_calc(model::ABM)
    age_alpha = 15
    age_theta = 2.667
    dist = Gamma(age_alpha, age_theta)
    
    age = rand(abmrng(model), dist, 1)[1]
    age = convert(Int64,round(age))
    if age < 18
        age = 18
    end
    (age < 18) && (age=18)
    (age > 80) && (age=79)
    return age
end


"""
    UpdateModelCounts!(model::ABM)
updating counts in the model.
Inlcudes number of unoccupied and occupied parcels.
"""
function UpdateModelCounts!(model::ABM)
    model.n_occupied = count_occupied(model)
    model.n_unoccupied = count_unoccupied(model)
    
    model.n_elevated_occupied = count_elevated_occupied(model)
    model.n_elevated_unoccupied = count_elevated_unoccupied(model)

    model.n_generator_occupied = count_generator_occupied(model)
    model.n_generator_unoccupied = count_generator_unoccupied(model)

    model.n_elevated_generator_occupied = count_elevated_generator_occupied(model)
    model.n_elevated_generator_unoccupied = count_elevated_generator_unoccupied(model)

end


"""
    count_unoccupied(model::ABM)
    count_occupied(model::ABM)
    count_elevated(model::ABM)
    count_generator(model::ABM)
functions to count number of occupied/unoccupied/elevated/generator buildings
"""
count_occupied(model::ABM) = sum(Int64, model.prcl_df.occupied)
count_unoccupied(model::ABM) = sum(Int64, 1 .- model.prcl_df.occupied)

count_elevated_occupied(model::ABM) = sum(Int64, (model.prcl_df[!,:occupied].==1) .&& (model.prcl_df[!,:elevated].==1) .&& (model.prcl_df[!,:generator].==0))
count_elevated_unoccupied(model::ABM) = sum(Int64, (model.prcl_df[!,:occupied].==0) .&& (model.prcl_df[!,:elevated].==1) .&& (model.prcl_df[!,:generator].==0))

count_generator_occupied(model::ABM) = sum(Int64, (model.prcl_df[!,:occupied].==1) .&& (model.prcl_df[!,:elevated].==0) .&& (model.prcl_df[!,:generator].==1))
count_generator_unoccupied(model::ABM) = sum(Int64, (model.prcl_df[!,:occupied].==0) .&& (model.prcl_df[!,:elevated].==0) .&& (model.prcl_df[!,:generator].==1))

count_elevated_generator_occupied(model::ABM) = sum(Int64, (model.prcl_df[!,:occupied].==1) .&& (model.prcl_df[!,:elevated].==1) .&& (model.prcl_df[!,:generator].==1))
count_elevated_generator_unoccupied(model::ABM) = sum(Int64, (model.prcl_df[!,:occupied].==0) .&& (model.prcl_df[!,:elevated].==1) .&& (model.prcl_df[!,:generator].==1))



# """
#     hazard!(model::ABM)
# writes parcel dataframe to shapefile and runs pyincore
# this function is where python is called
# """
# function hazard!(model::ABM)
#     path_to_bldgs = joinpath("temp", "TEMP_bldgs.shp")
#     model.prcl_df.geometry = GeoDataFrames.reproject(model.prcl_df.geometry, GeoFormatTypes.EPSG(32615), GeoFormatTypes.EPSG(4326); order=:trad)
#     GeoDataFrames.write(path_to_bldgs, model.prcl_df, crs=GeoFormatTypes.EPSG(4326))

#     # py"py.pyincore_hazard_buildings1"(model.input_dir)
#     println("Running IN-CORE")
#     py"py.pyincore_hazard_buildings2"(model.input_dir)
# end



"""
    close_model!(model::ABM)
closes up the model;
perform final agent steps and model counts
"""
function close_model!(model::ABM, model_runname::String)
    return
end

function AllAgentsCloseStep!(model::ABM)
    ids = random_ids(model)
    for id in ids
        agent_close_step!(model[id], model)
    end
end

# function merge_dmg_with_prcl_df!(model::ABM)
#     dmg_results = read_csv(joinpath("temp", "BldgDmg-Wind.csv"))
#     model.prcl_df = innerjoin(model.prcl_df, dmg_results[:, [:guid, :DS_0, :DS_1, :DS_2, :DS_3]], on=:guid)
#     rename!(model.prcl_df, Dict(:DS_0=>:DS_0_Wind, :DS_1=>:DS_1_Wind, :DS_2=>:DS_2_Wind, :DS_3=>:DS_3_Wind))

#     dmg_results = read_csv(joinpath("temp", "BldgDmg-Flood.csv"))
#     model.prcl_df = innerjoin(model.prcl_df, dmg_results[:, [:guid, :DS_0, :DS_1, :DS_2, :DS_3]], on=:guid)
#     rename!(model.prcl_df, Dict(:DS_0=>:DS_0_Flood, :DS_1=>:DS_1_Flood, :DS_2=>:DS_2_Flood, :DS_3=>:DS_3_Flood))

#     dmg_results = read_csv(joinpath("temp", "BldgDmg-SurgeWave.csv"))
#     model.prcl_df = innerjoin(model.prcl_df, dmg_results[:, [:guid, :DS_0, :DS_1, :DS_2, :DS_3]], on=:guid)
#     rename!(model.prcl_df, Dict(:DS_0=>:DS_0_SurgeWave, :DS_1=>:DS_1_SurgeWave, :DS_2=>:DS_2_SurgeWave, :DS_3=>:DS_3_SurgeWave))
# end

# function read_input_file_key(input_df::DataFrame, key::String; dtype::DataType=Int64)
#     v = input_df[input_df[:,"Variable"] .== key, "Value"][1]
#     (dtype==Bool) && (return parse_bool(v))

#     if dtype == String
#         return v
#     else
#         v = parse(dtype, v)
#     end
#     return v
# end

function parse_bool(v)
    v = parse(Int64, v)
    v = Bool(v)
    return v
end

"""
    ProgressBar(i, iters, n_years)
prints status of model to terminal
"""
# function ProgressBar(i, iters, n_years; color=:cyan, description="Iteration")
#     p = Progress(n_years, 
#             desc="$(description): $(i)/$(iters)",
#             # desc="Iteration: $(i)/$(iters) | ",
#             barlen=30, 
#             color=color
#         )
#     return p
# end


function ProgressBar(iters; color=:cyan, description="Status")
    p = Progress(iters, 
            desc="$(description): ",
            # desc="Iteration: $(i)/$(iters) | ",
            barlen=30, 
            color=color
        )
    return p
end



