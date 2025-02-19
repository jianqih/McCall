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



# Interpolations
using Interpolations,Plots
xs = 1:0.2:5
A = log.(xs)
interp_linear = linear_interpolation(xs, A)
int = interp_linear(xs)
plot(xs, A, label="data")
plot!(xs,int, label="linear interpolation")


# Performant Example Usage
# the LinearInterpolation is actually a short hand for composition of interpolate, scale, and extrapolate. 
xs
interp_linear = extrapolate(scale(interpolate(A, BSpline(Linear())), xs))
# if we don't need to extrapolate, we can use the following
scaled_itp = scale(interpolate(A, BSpline(Linear())), xs)

itp = interpolate(xs, BSpline(Linear()))
plot(itp)
itp[1.5]

f(x) =log(x);
xs = 1:0.2:5;
A = [f(x) for x in xs];
extrap = linear_interpolation(xs, A ,extrapolation_bc=Line())


extrap(1-0.2) ≈ f(1) - (f(1.2)- f(1))

extrap(5+0.2) ≈ f(5) + (f(5) - f(4.8))

extrap = linear_interpolation(xs, A, extrapolation_bc = NaN)

isnan(extrap(5.2))




# example with plots
a = 1.0
b = 10.0

x = a:1.0:b

y = @. cos(x^2/9.0)

itp_linear = linear_interpolation(x, y)
itp_cubic = cubic_spline_interpolation(x, y)

f_linear(x) = itp_linear(x)
f_cubic(x) = itp_cubic(x)  

width, height = 900, 400
x_new = a:0.1:b

scatter(x, y, label="data", legend=:topleft)
plot!(f_linear, x_new, w = 3, label = "Linear interpolation")
plot!(f_cubic, x_new, w = 3, label = "Cubic interpolation")
plot!(size = (width, height))
plot!(legend = :bottomleft)