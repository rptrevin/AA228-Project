using LightGraphs
using DataFrames
using DataStructures
using SparseArrays
using Printf
import SparseArrays.nonzeroinds
using Base.Threads

const EPSILON           = 1e-18  # convergence value
const CONVERGENCE_ITERS = 100    # num iteration to check for convergence

mutable struct MDP
    γ       # discount factor
    S       # number of states
    A       # number of actions
    T       # transition function T(s, a) -> SparseArray 
    R       # reward function R(s, a)
    S_used  # array of actually used states
    function MDP(γ, S, A)
        new(γ, S, A, nothing, nothing, nothing)
    end
end

function converged(U0, U1, S_used, epsilon = EPSILON)
    sz = size(U0)[1]
    for i in S_used
        if abs(U0[i] - U1[i]) > epsilon
            return false
        end
    end
    return true
end

#
# compute lookahead utility for passed state s and action a
#
function lookahead(m::MDP, U::SparseVector, s, a)
    T, R, u_sp = m.T(s, a), m.R, 0.0
    for s in nonzeroinds(T)
        u_sp += T[s] * U[s]
    end
    r = R(s, a)
    res = r + m.γ * u_sp
    if isnothing(res) || res > 1.0
        println("invalid reward:$(res) for ($(s),$(a))")
        @assert false
    end
    return res
end

#
# iteratively compute ulitity for each state for passed policy
#
function value_iterator(mdp::MDP, iters = 1000)
    S, R, T, γ, A, S_used = mdp.S, mdp.R, mdp.T, mdp.γ, mdp.A, mdp.S_used
    U = spzeros(Float64, S) # value function
    spin = SpinLock()

    println("Running Value Iterator. Please wait...")

    for i = 1:iters

        # copy value array to check
        if i % CONVERGENCE_ITERS == 0
            Up = copy(U)
        end

        for s in S_used
            a_best, u_best = -1, Float64(-Inf)
            for a = 1:A   
                u = lookahead(mdp, U, s, a)
                if u > u_best
                    a_best, u_best = a, u
                end
            end
            lock(spin)
            U[s] = u_best    
            unlock(spin)
        end

        if i % 10 == 0
            println(i)
        end

        # check for convergence
        if i % CONVERGENCE_ITERS == 0
            if converged(Up, U, S_used)
                println("Converged at $(i) iteration")
                break
            end
            println("Not yet converged")
        end        

    end
    return U
end

#
# given state utility function (represented by vector)
# compute optimal policy 
#
function optimal_policy(m::MDP, U::SparseVector)
    A     = m.A
    Q     = (s, a) -> lookahead(m, U, s, a)
    opt_a = (s) -> argmax([Q(s, a) for a = 1:A])

    π = [1 for i = 1:m.S]
    @threads for s in m.S_used
        π[s] = opt_a(s)
    end

    return π  
end

mutable struct mdp_trajectory_dataset
    s::Vector{Int64}     # from state
    a::Vector{Int64}     # action
    r::Vector{Float64}   # reward
    sp::Vector{Int64}    # to state 
    tid::Vector{Int64}   # trajectory id
    function mdp_trajectory_dataset()
        new([], [], [], [], [])
    end
    
end

#
# Load data to discrete state discrete action MDP 
# Parameters:
#       mdp   - output mdp
#       sarsp - DataFrame with (state, action, reward, next state)
#               tuples (names are "s", "a", "r", "sp") 
#
function load!(mdp::MDP, sarsp)
    
    println("Loading MDP from dataset. Please wait...")

    S, A = mdp.S, mdp.A
    s, a, r, sp = sarsp.s, sarsp.a, sarsp.r, sarsp.sp
    m = size(s)[1]
    states = Set()
    r_max, r_min = Float64(-Inf), Float64(Inf)

    # we keep Transitions in dictionary
    td = Dict()
    T = function (s, a)
        if !haskey(td, (s, a))
            td[(s, a)] = spzeros(Float64, mdp.S)
        end
        return td[(s, a)]
    end

    R   = spzeros(Float64, mdp.S, mdp.A)
    R_C = spzeros(Int64, mdp.S, mdp.A)

    open("dataset.csv", "w") do io
        println(io, "s,a,r,sp")
        for i = 1:m
            s_i, a_i, r_i, sp_i = s[i], a[i], r[i], sp[i]

            println(io, "$(s_i),$(a_i),$(r_i),$(sp_i)")

            t_sa           = T(s_i, a_i)
            t_sa[sp_i]    += 1.0
            R[s_i, a_i]   += r_i
            R_C[s_i, a_i] += 1

            push!(states, s_i)
            push!(states, sp_i)

            r_max = max(r_max, r_i)
            r_min = min(r_min, r_i)
        end 
    end

    for s in states
        for a = 1:A
            if R_C[s, a] != 0
                R[s, a] /= R_C[s, a]
            end
            if sum(T(s, a)) != 0
                t_sa = T(s, a) / (sum(T(s, a)) + EPSILON)
                td[(s, a)] = t_sa
            end
        end
    end
    
    mdp.T = T
    mdp.R = (s, a) -> R[s, a]
    mdp.S_used = [s for s in states]

    write_transition(mdp)
    write_rewards(mdp)

    println("Dataset loaded. Actual number of states $(length(states))")
    println("Max reward $(r_max). Min reward $(r_min)")
end

function write_transition(mdp)
    open("AI_phys.transitions", "w") do io
        for s in mdp.S_used
            println(io, s)
            for a = 1:mdp.A
                t = mdp.T(s, a)
                println(io, "($(s),$(a))->\n$(t)")
            end
        end
    end
end

function write_rewards(mdp)
    open("AI_phys.rewards", "w") do io
        for s in mdp.S_used
            println(io, s)
            for a = 1:mdp.A
                r = mdp.R(s, a)
                println(io, "($(s),$(a))->$(r)")
            end
        end
    end
end


#
# Write policy to file
#
function write_policy(policy, filename)
    open(filename, "w") do io
        for i=1:length(policy)
            @printf(io, "%s\n", policy[i])
        end
    end
end
