
function simulate_series(E_mat, para, burn_in=200, capT=10000)

    @unpack k, α_L, z, s, η_L, β, ρ_x, stdx, NU, NS, A, mc, u_grid = para
    capT = capT + burn_in + 1

    # bivariate interpolation
    E_pol = Spline2D(u_grid, A, E_mat)

    # draw shocks
    η_x = randn(capT).*stdx
    x = zeros(capT)
    for t in 1:(capT-1)
        x[t+1] = ρ_x*x[t] + η_x[t+1]
    end
    x = exp.(x)

    u = ones(capT+1)*0.05
    vars = ones(capT, 8)
    f, θ, v, w, M, Y, E = columns(vars)
    
    for t in 1:capT
        # interpolate E
        E[t] = E_pol(u[t], x[t])
        # recover policies and next-period unemployment from interpolation
        __, θ[t], f[t], v[t], w[t], M[t], Y[t], u[t+1] = get_policies(E[t], u[t], x[t], para)
    end
    # remove last element of u
    pop!(u)
    # remove burn-in
    out = [θ f v w u x M Y][(burn_in+1):end, :]
    θ, f, v, w, u, x, M, Y = columns(out)
    labor_share = w./x

    Simulation = (θ=θ, f=f, v=v, w=w, u=u, x=x, M=M, Y=Y, labor_share=labor_share)
    return Simulation
end


function impulse_response(E_mat, para; u_init=0.07, irf_length=60, scale=1.0)

    @unpack k, α_L, z, s, η_L, β, ρ_x, stdx, NU, NS, A, mc, u_grid = para

    # bivariate interpolation
    E_pol = Spline2D(u_grid, A, E_mat)

    # draw shocks
    x = zeros(irf_length)
    x[1] = stdx*scale
    for t in 1:(irf_length-1)
        x[t+1] = ρ_x*x[t]
    end
    x = exp.(x)
    # counterfactual
    x_bas = ones(irf_length)

    function impulse(x_series)
        u = ones(irf_length+1)
        u[1] = u_init
        vars = ones(irf_length, 7)
        f, θ, v, w, M, Y, E = columns(vars)
        
        for t in 1:irf_length
            # interpolate E
            E[t] = E_pol(u[t], x_series[t])
            # recover policies and next-period unemployment from interpolation
            __, θ[t], f[t], v[t], w[t], M[t], Y[t], u[t+1] = get_policies(E[t], u[t], x_series[t], para)
        end
        # remove last element of u
        pop!(u)
        labor_share = w./x_series

        out = [θ f v w u x_series M Y labor_share]
        return out
    end

    out_imp = impulse(x)
    out_bas = impulse(x_bas)

    irf_res = similar(out_imp)
    @. irf_res = 100*log(out_imp/out_bas)
    θ, f, v, w, u, x, M, Y, labor_share = columns(irf_res)
    irf = (θ=θ, f=f, v=v, w=w, u=u, x=x, M=M, Y=Y, labor_share=labor_share)
    return irf
end



# Simulate model
simul = simulate_series(E_mat, para)

fields = [:x, :u, :θ, :Y, :M, :labor_share]
# Log deviations from stationary mean
out = reduce(hcat, [100 .*log.(getfield(simul, x)./mean(getfield(simul,x))) for x in fields])

# Create time series object
# df = time_series_object(out, fields);

# # convert to quarterly
# df_q = collapse(df, eoq(df.index), fun=mean);
# #cycle = mapcols(col -> hamilton_filter(col, h=8), DataFrames.DataFrame(df_q.values,:auto))
# cycle = mapcols(col -> hp_filter(col, 10^5)[1], DataFrames.DataFrame(df_q.values,:auto))

# DataFrames.rename!(cycle, fields)

# #Extract moments
# mom_mod = moments(cycle, :x, fields, var_names=fields)
# pprint(mom_mod)



begin
    fig, ax = subplots(1, 3, figsize=(20, 5))
    t = 250:1000
    ax[1].plot(t, simul.x[t], label="x")
    ax[1].plot(t, simul.w[t], label="w")
    ax[1].set_title("Subplot a: Productivity and wages")
    ax[1].legend()

    ax[2].plot(t, simul.f[t], label="f")
    ax[2].plot(t, simul.θ[t], label="θ")
    ax[2].set_title("Subplot b: Job finding rate and tightness")
    ax[2].legend()

    ax[3].plot(t, simul.u[t], label="u")
    ax[3].set_title("Subplot c: Unemployment")
    ax[3].legend()
    display(fig)
    PyPlot.savefig("simulations.pdf")

    " Impulse responses "
    irf =impulse_response(E_mat, para, u_init=ss.u)

    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    ax[1].plot(irf.θ, label="θ")
    ax[1].plot(irf.u, label="u")
    ax[1].legend(loc="best")
    ax[1].set_title("Market tightness and vacancies")
    
    ax[2].plot(irf.M, label="M")
    ax[2].plot(irf.Y, label="Y")
    ax[2].legend(loc="best")
    ax[2].set_title("Output and the stock market cap")
    
    ax[3].plot(irf.x, label="x")
    ax[3].plot(irf.w, label="w")
    ax[3].plot(irf.labor_share, label="labor_share")
    ax[3].legend(loc="best")
    ax[3].set_title("Productivity shock, wages, and the labor share")
    tight_layout()
    display(fig)
    PyPlot.savefig("irfs.pdf")
end

simul = DataFrame(simul)

# "Beveridge curve"
begin   
    Plots.scatter(simul.u,simul.θ, alpha = 0.3, colorbar = :none, legend = false, markersize = 2)
    title!("Beveridge curve")
    xlabel!("u")
    ylabel!("θ")
    # savefig("Beveridge_curve.pdf")
end

