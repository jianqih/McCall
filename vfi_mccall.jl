using Distributions,Expectations
using NLsolve, Roots, Random, LaTeXStrings
using Plots, LinearAlgebra, Statistics
begin
    β = 0.99
    nw = 100
    w_min = 50;
    w_max = 150;
    c = 25;
    w_vec = collect(range(w_min,stop=w_max,length=nw+1))
end

# plot the distribution of wages
begin
    using StatsPlots
    dist = BetaBinomial(nw, 200, 100) # probability distribution
    @show support(dist)
    plot(w_vec,pdf.(dist,support(dist)),xlabel="wages",ylabel="probabilities",legend=false)
end

begin
    V = fill(0.0, nw)
    tol = 1e-6
end


begin
    using LinearAlgebra
    function update_V(V, w_vec, c, β)
        V_new = fill(0.0, nw+1)
        for (i, w_val) in enumerate(w_vec)
            acc = w_val/(1-β)
            rej = c + β * dot(pdf.(dist,support(dist)),V);
            V_new[i] = max(acc, rej)
        end
        return V_new
    end
end

begin
    num_plots = 10
    V_init = w_vec ./(1-β)
    vs = zeros(nw+1,num_plots)
    vs[:,1] = V_init
    for col in 2:num_plots
        V_last = vs[:,col-1]
        vs[:,col] .= update_V(V_last, w_vec, c, β)
    end
    plot(vs)
end

begin
    E = expectation(dist) # expectation operator
    using FixedPoint
    # finding the reservation wage
    T(v) = max.(w_vec/(1-β),c+β*E*v)
    v_iv = collect(w_vec ./ (1 - β))
    sol = afps(T, v_iv, iters = 4000)
    v_star = sol.x
    R = (1-β) *(c+β*E*v_star)
end

grid_size = 25
R = zeros(grid_size, grid_size)

c_vals = range(10.0, 30.0, length = grid_size)
beta_vals = range(0.9, 0.99, length = grid_size)
