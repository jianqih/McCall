begin
    using Plots
    using BenchmarkTools
    using LaTeXStrings
    using Parameters, CSV, Random, QuantEcon
    using Distributions
    using LinearAlgebra, Roots, Interpolations, Dierckx
    using Printf
    using DataFrames
end

function jf(θ, η_L)
    return θ/(1+θ^(η_L))^(1/η_L)
end

#vacancy filling probability
function vf(θ, η_L)
    return 1/(1+θ^η_L)^(1/η_L)
end

function theta_invert(q, η_L)
    θ = (1/q^(η_L) - 1)^(1/η_L)
    return θ
end

function columns(M)
    return (view(M, :, i) for i in 1:size(M, 2))
end

"""
    grid_cons_grow(n, left, right, g)
Create n+1 gridpoints with growing distance on interval [a, b]
according to the formula
x_i = a + (b-a)/((1+g)^n-1)*((1+g)^i-1) for i=0,1,...n 
"""
function grid_cons_grow(n, left, right, g)
    x = zeros(n)
    for i in 0:(n-1)
        x[i+1] = @. left + (right-left)/((1+g)^(n-1)-1)*((1+g)^i-1)
    end
    return x
end


function calibrate(;β=exp(-4.0/1200.0), α_L=0.5, z=0.71, η_L=1.25,
    s=0.035, u_target=0.058, ρ_x=0.95^(1/3), stdx=0.00625)

    r = 1/β - 1.0
    function err(x, k)
        return k/vf(x, η_L) - ((1-α_L)*(1-z)-α_L*k*x)/(r+s)
    end

    function loss(k)
        θ = find_zero(x -> err(x, k), (0.0, 10.0), Bisection())
        f = jf(θ, η_L)
        return s/(s+f) - u_target
    end

    k = find_zero(loss, (0.1, 1), Bisection())
    CalibratedParameters = (k=k, α_L=α_L, z=z, s=s, η_L=η_L, β=β, ρ_x=ρ_x, stdx=stdx)
    return CalibratedParameters
end


function steady_state(para)
    @unpack k, α_L, z, s, η_L, β, ρ_x, stdx = para
    x = 1.0
    r = 1/β - 1.0

    function err(x, k)
        return k/vf(x, η_L) - ((1-α_L)*(1-z)-α_L*k*x)/(r+s)
    end

    θ = find_zero(x ->err(x, k), (0.01, 10), Bisection())
    f = jf(θ, η_L)
    q = f/θ
    u = s/(s+f)
    w = α_L*(x+k*θ) + (1-α_L)*z
    v = θ*u
    # Average hiring cost/ expected value of a filled job
    E = k/q

    J = x-w + (1-s)*k/q
    M = (1-u)*J
    Y = (1-u)*x

    SteadyState = (E=E, θ=θ, q=q, f=f, u=u, w=w, v=v, M=M, Y=Y)
    return SteadyState
end


@with_kw struct Para{T1, T2, T3, T4}
    # model parameters
    k::Float64 
    α_L::Float64
    z::Float64
    s::Float64
    η_L::Float64
    β::Float64
    ρ_x::Float64
    stdx::Float64

    # numerical parameter
    u_low::Float64 = 0.02
    u_high::Float64 = 0.35
    max_iter::Int64 = 1000
    NU::Int64 = 50
    NS::Int64 = 20
    T::Float64 = 1e5
    mc::T1 = rouwenhorst(NS, ρ_x, stdx, 0)
    P::T2 = mc.p
    A::T3 = exp.(mc.state_values)
    u_grid::T4 = grid_cons_grow(NU, u_low, u_high, 0.02)
end


function get_policies(E, u, x, para)
    " obtain policies from right-hand side of Euler equation "

    @unpack k, α_L, z, s, η_L, β, NU, NS, u_grid, A, u_low, u_high = para
    q = k/E
    q = min(q, 1.0)
    θ = theta_invert(q, η_L)
    f = θ*q
    v = θ*u 
    w = α_L*(x+k*θ) + (1-α_L)*z
    u_p = s*(1-u) + (1-f)*u
    # enforce bounds
    u_p = min(u_p, u_high)
    u_p = max(u_p, u_low)

    J = x-w + (1-s)*k/q
    M = (1-u)*J
    Y = (1-u)*x
    return q, θ, f, v, w, M, Y, u_p
end

function rhs_jcc(E_pol, iu, ix, para)
    # Right-hand side for particular state
    @unpack k, α_L, z, s, η_L, β, NU, NS, u_grid, A, P = para
    # Reconstruct right-hand side of Euler
    u = u_grid[iu]
    x = A[ix]
    # current right-hand side
    E = E_pol(u, ix)
    # extract policies and next-period unemployment
    q, θ, f, v, w, M, Y, u_p= get_policies(E, u, x, para)
    E_new = 0.0
    @inbounds @simd for ix_p in 1:NS
        x_p = A[ix_p]
        E_p = E_pol(u_p, ix_p)
        q_p, θ_p, f_p, v_p, w_p, M_p, Y_p, u_p2 = get_policies(E_p, u_p, x_p, para)
        # add job surplus under realization ix_p
        E_new += P[ix, ix_p]*β*(x_p-w_p + (1-s)*(k/q_p))
    end
    return E_new
end


function solve_model_time_iter(E_mat, para; tol=1e-7, max_iter=1000, verbose=true, 
                                print_skip=25, ω=0.7)
    # Set up loop 
    @unpack k, α_L, z, s, η_L, β, NU, NS, u_grid, A, P = para
    
    err = 1
    iter = 1
    while (iter < max_iter) && (err > tol)
        E_pol(u, ix) = interpolate(u_grid, @view(E_mat[:, ix]), extrapolate=:reflect)(u)
        # interpolate given grid on EE
        E_new = zeros(NU, NS)
        @inbounds for (iu, u) in collect(enumerate(u_grid))
            for ix in 1:NS
                # new right-hand side of EE given iu, ix
                E_new[iu, ix] = rhs_jcc(E_pol, iu, ix, para)
            end
        end
        E_new .= ω*E_new +(1-ω)*E_mat
        err = maximum(abs.(E_new-E_mat)/max.(abs.(E_mat), 1e-10))
        if verbose && iter % print_skip == 0
            print("Error at iteration $iter is $err.")
        end
        iter += 1
        # update grid of rhs of EE
        E_mat = E_new
    end
    itpu = linear_interpolation((u_grid), @view(E_mat[:, ix]),extrapolation_bc=Line())
    E_pol(u, ix) = itpu(u)
    # Get convergence level
    if iter == max_iter
        print("Failed to converge!")
    end

    if verbose && (iter < max_iter)
        print("Converged in $iter iterations")
    end
    # Get remainding variables
    # Productivity on (NS, NU) grid
    X = repeat(A, 1, NU)'
    q = k./E_mat
    q = min.(q, 1.0)
    θ = theta_invert.(q, η_L)
    f = θ.*q
    v = θ.*u_grid
    w =  α_L.*(X.+k.*θ) .+ (1-α_L)*z

    return E_mat, E_pol, X, q, f, v, w
end


   
# calibrate model
cal = calibrate(stdx=0.015)
@unpack k, α_L, z, s, η_L, β, ρ_x, stdx = cal

ss = steady_state(cal)
# Stock market cap
ss.M/ss.Y 

# form struct using calibrated parameters
para = Para(k=k, α_L=α_L, z=z, s=s, η_L=η_L, β=β, ρ_x=ρ_x, stdx=stdx)
@unpack NU, NS, u_grid = para

# initial guess of rhs jcc
E_mat = zeros(NU, NS)
E_mat .= k/ss.q

# Solve model: iterating on the job creation condition
E_mat, E_pol, X, q, f, v, w = solve_model_time_iter(E_mat, para)
