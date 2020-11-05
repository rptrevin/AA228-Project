using DataFrames
using CSV
using Profile
using PProf
using Random
using JSON

include("mdp.jl")

const PROFILE_ONLY = false
const n_patients   = 5366
const n_states     = 750
const DS_PATH      = "../dataset_artifacts/"
STATE_DECEASED     = n_states + 1
STATE_SURVIVED     = n_states + 2

function load_trajectories()
    patients = []
    ds       = mdp_trajectory_dataset()

    for i = 1:n_patients
        p = JSON.parsefile(DS_PATH * "mdp_patient_" * string(i) * ".json")
        push!(patients, p) 
        rewards, actions, states = p["rewards"], p["actions"], p["trajectory"]
        for j = 1:length(actions)
            push!(ds.s, states[j])
            push!(ds.a, actions[j])
            push!(ds.r, rewards[j])
            push!(ds.sp, states[j + 1])
            push!(ds.tid, i)
        end

        if i % 1000 == 1 || i == n_patients
            println("loaded patient $(i) of $(n_patients)")
        end
    end
    return patients, ds
end

function solve()
    # input_data = CSV.read(infile)

    patients, dataset = load_trajectories()
    mdp = MDP(1, n_states + 2, 25)
    
    load!(mdp, dataset)

    U = value_iterator(mdp)
    π = optimal_policy(mdp, U)

    return U, π
end

#
# Main function that gets input data and stores output graph structure and graph image
#
function main()
    if PROFILE_ONLY
        println("WARNING: Profile only mode. Output will not be saved.")
        println("WARNING: Run from julia as include(value_iterator.jl)")
        @pprof solve()
    else
        U, π = @time solve()

        write_policy(π, "AI_phys.policy")
        write_policy(U, "AI_phys_state.utility")
    end

end

main() 