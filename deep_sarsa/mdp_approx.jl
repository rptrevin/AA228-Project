import DataStructures
using Flux, CUDA
using Random

include("regression_b.jl")

mutable struct approximate_value
    φ     # function that converts (state id, action) pair to feature vector
    Qθ    # Q value predictor function
    Q     # actual Q value 
    model # NN model
    target_factor
    function approximate_value(n, φ, layers_sizes, factor)
        model = make_model(n, layers_sizes)
        function q_model(s, a) 
            return predict(model, [φ(s, a)])[1]
        end
        new(φ, q_model, DefaultDict(0.0), model, factor)
    end
end

#
# Bellman backup for passed state we return 
#     R(s, a) + γ∑[T(s'|s,a)U(s')]
#       Where:
#          U(s') = max_a[Q(s', a)]
function U_opt(m::MDP, avi::approximate_value, s; approximate = false)
    a_opt, q_opt, q_sa = 1, Float64(-Inf), 0
    for a = 1:m.A
        if approximate
            q_sa = avi.Qθ(s, a)
        else
            q_sa = avi.Q[(s, a)]
        end
        if q_sa > q_opt
            q_opt, a_opt = q_sa, a
        end
    end
    return q_opt, a_opt
end

function Q_opt(m::MDP, avi::approximate_value, s, a; approximate = false)
    R, γ, T, U, ∑ = m.R, m.γ, m.T(s, a), (s) -> U_opt(m, avi, s, approximate = approximate), sum
    r = 0.0
    for sᵢ in nonzeroinds(T)
        r += T[sᵢ] * U(sᵢ)[1]
    end
    return R(s, a) + γ * r
end

function fit!(avi::approximate_value, sa, Qsa; η = 0.01, iters = 20, show_loss = true)
    φ, Y = avi.φ, Qsa
    Φ = [φ(s, a) for (s, a) in sa]
    B = make_batches(Φ, Y)
    fit!(avi.model, B, iters = iters, η = η, show_loss = show_loss)
end

#
# Same value iterator, except we use Q(s,a) value function
# approximator with DNN
#
function solve(m::MDP, avi::approximate_value; iters = 1600, η = 0.001, ln_iters = 20)

    println("Running DNN value iterator solver. Please wait...")
    
    S = [s for s in m.S_used]             # array of used states (ids of states)
    shuffle!(S)

    n    = size(S)[1]                        # number of used states
    sa   = [(S[i], a) for i=1:n for a=1:m.A] # array of (state, action) pairs
    spin = SpinLock()

    for j = 1:n
        for a = 1:m.A
            avi.Q[(S[j], a)] = 0
        end
    end

    for i = 1:iters
        @threads for j = 1:n
            s = S[j]
            for a = 1:m.A
                Qsa = Q_opt(m, avi, s, a)
                lock(spin)
                avi.Q[(s, a)] = Qsa
                unlock(spin)
            end
        end
        if i % 10 == 1 || i == iters
            println("Iteration $(i)")
        end
    end

    π = [1 for s = 1:m.S]
    U = [0.0 for s = 1:m.S]

    println("Fitting Q(s,a) value function approximator...")
    factor = avi.target_factor
    qsa = [avi.Q[(S[i], a)] / factor for i=1:n for a=1:m.A]
    fit!(avi, sa, qsa, η = η, iters = ln_iters)
    for s = 1:m.S
        U[s], π[s] = U_opt(m, avi, s, approximate = true)
        U[s] *= factor
    end

    # override results for observed states
    for j = 1:n
        s = S[j]
        U[s], π[s] = U_opt(m, avi, s)
    end

    return U, π
end
