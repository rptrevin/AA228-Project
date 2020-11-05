using JSON
using MultivariateStats
using RDatasets 
using Distributions
using Random
using PyPlot
using Clustering
using LinearAlgebra
using Base.Threads

dpath = "../dataset_json/"
apath = "../dataset_artifacts/"

# number of patients
n_patients = 5366
n_states   = 750

STATE_DECEASED = n_states + 1
STATE_SURVIVED = n_states + 2

function save_state_vectors(filename, clusters)
    json_vc = JSON.json(clusters)
    open(filename, "w") do io
       println(io, json_vc)
    end
end

function restore_state_vectors(filename)
    s = ""
    open(filename, "r") do io
       s = read(io, String)  
    end
    dvc = JSON.parse(s)
    return hcat(dvc...)
end

function distance(p1, p2, dims)
    d = 0.0
    for i = 1:dims
        d += (p1[i] - p2[i])^2
    end
    return d
end

#
# K-mean++ initialization for better k-mean performance
# 
function k_means_pp(data, k)
    dim      = size(data)[1]
    m        = size(data)[2]

    centroids = []
    push!(centroids, data[1:dim, rand(1:m)])

    dist = []
    resize!(dist, m)
    
    for i = 1:(k - 1)
        @threads for s = 1:m
            p = data[:, s]
            d = Float64(Inf)

            for j = 1:size(centroids)[1]
                p_dist = distance(p, centroids[j], dim)
                d = min(d, p_dist)
            end
            dist[s] = d
        end
        
        # add next centroid
        push!(centroids, data[1:dim, argmax(dist)])
        
        if i % 10 == 1 || i == k - 1
            println("k_means_pp $(i) of $(k)")
        end
    end
    
    return centroids 
end

# reducing dimentionality of state space for 
# visualization
function partition_dataset(data; at = 0.999)
    n         = size(data, 2)
    idx       = shuffle(1:n)
    
    train_idx = view(idx, 1:floor(Int, at * n))    
    test_idx  = view(idx, (floor(Int, at * n) + 1):n)
    
    return data[:,train_idx], data[:,test_idx]
end

function create_pca_matrix(data; at = 0.999, dims = 3)
    M = fit(PCA, data; maxoutdim = dims)
    return M
end

function reduce_dims(M, data)
    return MultivariateStats.transform(M, data)  
end

function shuffle_cols(data)
    n = size(data, 2)
    v = view(shuffle(1:n), 1:n)
    return data[:, v]
end

# adding all states for all patients into 
# single array to find clusters
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

# return list of patients with their trajectories
function load_patients()
    patients = []
    for i = 1:n_patients
        p = JSON.parsefile(dpath * "patient_" * string(i) * ".json")
        push!(patients, p) 
    end
    return patients
end
