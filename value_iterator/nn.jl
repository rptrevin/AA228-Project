using Flux
using Flux: throttle
using CUDA

mutable struct neural_network_model
    n_input::Int64            # feature size
    n_output::Int64           # size of output 
    layer_config::Vector{Int} # sizes of hidden layers
    classify::Bool            # is it softmax classifier or regression
    model                     # Flux model for NN
    function neural_network_model(;n_input, n_output = 1, layer_config = [], classify = false)
        new(n_input, n_output, layer_config, classify, nothing)
    end
end

#
# Create NN model of specified configuration
#
# n:             feature size
# layers_sizes:  hidden layer sizes
#
# Function make_model(n, layers_sizes, outputs = 1, classifier = false)
#
function init!(nn::neural_network_model)
    n, layers_sizes, outputs, classifier = nn.n_input, nn.layer_config, nn.n_output, nn.classify
    inp_sz, layers = n, []
    for l in layers_sizes
        push!(layers, Dense(inp_sz, l, relu)) 
        inp_sz = l
    end
    push!(layers, Dense(inp_sz, outputs, identity))
    if classifier
        push!(layers, softmax)
    end
    nn.model = Chain(layers...) |> gpu
end

#
# Prepare minibatches of dataset for fit!() and send them to GPU
#
# X: is feature vector s.t. [sample_1, sample_2,...] and sample_i is array of features
# Y: is array of targets    [y_1, y_2,...]
#
# Example of making batches for classifier:
#   preprocess(img) = vec(Float64.(img))
#   X = [preprocess(img) for img in images]
#   Y = [Flux.onehot(l, 0:9) for l in labels]
#   B = make_batches(X, Y, 128);
#
function make_batches(X, Y, batch_size = 4096)
    m, batches = size(X)[1], []
    for s = 1:batch_size:m
        t = min(m, (s + batch_size - 1))
        X_batch = reduce(hcat, X[s:t])
        Y_batch = reduce(hcat, Y[s:t])
        push!(batches, (X_batch, Y_batch))
    end
    batches = batches |> gpu
    return batches
end

#
# Fit model to passed dataset
#
# D:     is dataset created with make_data
# model: is NN model created with make_model
#
function fit!(nn::neural_network_model, batches; iters = 10, η = 0.001, show_loss = true)
    model      = nn.model
    loss(x, y) = !nn.classify ? Flux.mse(model(x), y) : Flux.logitcrossentropy(model(x), y)
    B, evalcb  = batches, nothing
    opt        = ADAM(η)

    if show_loss
        evalcb = () -> @show(loss(B[1][1], B[1][2]))
    else
        evalcb = () -> print(".");
    end
    for i = 1:iters
        Flux.train!(loss, params(model), batches, opt, cb = throttle(evalcb, 1000)) 
    end

    println()
end

function predict(model, X)
    X_gpu = reduce(hcat, X) |> gpu
    Y_gpu = model(X_gpu)
    Y = Y_gpu |> cpu
    return Y
end