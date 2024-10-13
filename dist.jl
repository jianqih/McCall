using Distributions
ν = 5
d = Chisq(ν)     # Chi-squared distribution with ν degrees of freedom

params(d)    # Get the parameters, i.e. (ν,)
dof(d)       # Get the degrees of freedom, i.e. ν

α = 2
β = 2
BetaPrime()        # equivalent to BetaPrime(1, 1)
BetaPrime(α)       # equivalent to BetaPrime(α, α)
BetaPrime(α, β)    # Beta prime distribution with shape parameters α and β

params(d)          # Get the parameters, i.e. (α, β)