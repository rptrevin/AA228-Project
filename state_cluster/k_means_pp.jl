using Distributions
using Random
using PyPlot
using Clustering
using LinearAlgebra

dims      = 2
clusters  = 4 
n_samples = 50

function make_μ(dims)
    return [rand() * 20.0 for _ in 1:dims]
end

function make_Σ(dims)
    A = rand(Float64, (dims, dims))
    A = 0.5*(A + A')
    A = A + dims * Matrix{Float64}(I, dims, dims)
    return A
end

arr_μ = [make_μ(dims) for _ in 1:clusters]
arr_Σ = [make_Σ(dims) for _ in 1:clusters]

mvs  = [MvNormal(μ, Σ) for (μ, Σ) in zip(arr_μ, arr_Σ)]
data = hcat((rand(d, 50) for d in mvs)...)

function plot(data, centroids = missing)
    
    scatter(data[1,:], data[2,:], color = "gray")
    
    if !ismissing(centroids)
        scatter(centroids[1,:], centroids[2,:], color = "red")
    end
    
    legend()
    ylim(-50, 50)
    xlim(-50, 50)

    plt.show()
end

function distance(p1, p2) 
    return sum((p1 - p2).^2) 
end
   
function k_means_pp(data, k)
    dim      = size(data)[1]
    m        = size(data)[2]

    centroids = []
    push!(centroids, data[1:dim, rand(1:m)])
       
    for i = 1:(k - 1)
        dist = []

        for s = 1:m
            p = data[:, s]
            d = Float64(Inf)

            for j = 1:size(centroids)[1]
                p_dist = distance(p, centroids[j])
                d = min(d, p_dist)
            end
            push!(dist, d)
        end
        
        # add next centroid
        push!(centroids, data[1:dim, argmax(dist)])
    end
    
    return centroids 
end

centroids = k_means_pp(data, 4)
vc        = hcat(centroids...)

plot(data, vc) 

R = kmeans!(data, vc)
plot(data, vc) 