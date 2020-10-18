using JSON
using MultivariateStats
using RDatasets 
using Distributions
using Random
using PyPlot
using Clustering
using LinearAlgebra

# path to datasets
dpath = "../dataset_json/"

# number of patients (this corresponds to patient_999.json files)
n_patients  = 5366

# number of clusters (states) of patient
n_states    = 750

#
# Create list of patient states (each state is 49-vector)
# We return (49, m) matrix with columns being individual states
# and m is number of all possible states encountered for parients
# (say if patient each of 100 patients has trajectory of 200, then
# we return matrix of size (49, 100*200))
#
function load_states()
    states = []
    for i = 1:n_patients
        p     = JSON.parsefile(dpath * "patient_" * string(i) * ".json")
        τ     = p["trajectory"]
        τ_len = size(τ)[1] 
        for j = 1:τ_len
            push!(states, τ[j]) 
        end
    end
    # form state matrix, columns are 49-vectors of states
    S = hcat(states...)
    return S
end

function shuffle_columns()
