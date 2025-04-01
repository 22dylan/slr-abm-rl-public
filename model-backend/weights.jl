# Include the file where `get_agent_weights` is defined
# include(joinpath("model-backend", "ResidentialAgent.jl"))

# Constants
# const WEIGHTS_FILE = "weights.json"

# Define a mutable struct to hold run data
mutable struct RunData
    run_number::Int          # Run number
    weights::Vector{Float64} # Vector to store weights
end

# Function to read weights from the `get_agent_weights` function
function read_weights(input_struct)
    weights = get_agent_weights(input_struct, ResidentialAgent)  # Call `get_agent_weights`
    return weights
end

# Function to read costs from the get_agent_costs function
function read_costs(input_struct)
    costs = get_agent_costs(input_struct, ResidentialAgent)  # Call get_agent_costs
    return costs
end

# Function to read the run number from the weights JSON file
function read_run_number(run_number::String, WEIGHTS_FILE::String)
    if isfile(WEIGHTS_FILE)  # Check if weights file exists
        file = open(WEIGHTS_FILE, "r")  # Open file for reading
        data = JSON.parse(file)  # Parse JSON data from file
        close(file)  # Close file
        
        if length(data) > 0  # If data exists in JSON
            for entry in data
                if entry["run_number"] == run_number
                    error("Run number $run_number already exists.")
                end
            end
        end
    end
    return run_number  # Return the provided run number
end

# Function to re-order an entry as OrderedDict
function reorder_entry(entry)
    return OrderedDict(
        "run_number" => entry["run_number"],
        "weight1" => entry["weight1"],
        "weight2" => entry["weight2"],
        "weight3" => entry["weight3"],
        "weight4" => entry["weight4"],
        "weight5" => entry["weight5"],
        "cost_elevate" => entry["cost_elevate"],
        "cost_generator" => entry["cost_generator"]
    )
end

# Function to check if a run is a duplicate
function is_duplicate_run(new_entry, existing_data)
    for entry in existing_data
        if (entry["weight1"] == new_entry["weight1"]) &&
           (entry["weight2"] == new_entry["weight2"]) &&
           (entry["weight3"] == new_entry["weight3"]) &&
           (entry["weight4"] == new_entry["weight4"]) &&
           (entry["weight5"] == new_entry["weight5"]) &&
           (entry["cost_elevate"] == new_entry["cost_elevate"]) &&
           (entry["cost_generator"] == new_entry["cost_generator"])
            return true
        end
    end
    return false
end

# Function to save run data (run number and weights) to JSON file
function save_weights(run_number::String, weights::Vector{Float64}, costs::Vector{Float32}, WEIGHTS_FILE::String)
    # Define JSON entry with ordered keys
    new_entry = OrderedDict(
        "run_number" => run_number,
        "weight1" => weights[1],
        "weight2" => weights[2],
        "weight3" => weights[3],
        "weight4" => weights[4],
        "weight5" => weights[5],
        "cost_elevate" => costs[1],
        "cost_generator" => costs[2]
    )
    data = OrderedDict{String, Any}[]  # Initialize empty array for data
    
    if isfile(WEIGHTS_FILE)  # Check if weights file exists
        file = open(WEIGHTS_FILE, "r")  # Open file for reading
        existing_data = JSON.parse(file)  # Parse JSON data from file
        close(file)  # Close file
        
        # # Check for duplicates
        # if is_duplicate_run(new_entry, existing_data)
        #     error("Duplicate run found with identical weights and costs.")
        # end
        
        # Re-order existing entries and add them to data array
        for entry in existing_data
            push!(data, reorder_entry(entry))
        end
    end

    push!(data, new_entry)  # Add new entry to data array
    
    file = open(WEIGHTS_FILE, "w")  # Open file for writing
    JSON.print(file, data, 2)  # Write formatted JSON data to file (indentation level 2)
    close(file)  # Close file
end

# Main function to coordinate reading, saving, and printing run data

function update_weights_json(model::ABM, WEIGHTS_FILE::String="weights.json")
    input_struct = model.input_struct
    run_number = input_struct.sub_runname
    run_number = read_run_number(run_number, WEIGHTS_FILE)  # Read the provided run number
    weights = model.ResidentialAgent_Dict[:family1][:weights]
    costs = model.ResidentialAgent_Dict[:family1][:costs]

    # weights = read_weights(input_struct)  # Read weights from function
    # costs = read_costs(input_struct)  # Read costs from function
    save_weights(run_number, weights, costs, WEIGHTS_FILE)  # Save run number and weights to JSON file
    println("$run_number weights saved: ", weights)  # Print confirmation message
end

# Example call to the main function
# run_number = "run-5"
# get_sub_runname(run_number, WEIGHTS_FILE="weights.json")