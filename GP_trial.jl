using GaussianProcesses
using Random
using Plots
plotlyjs()
using Statistics
using KernelFunctions
import GaussianProcesses.Noise
##
F(x,y) = x^2 + y^2
function df(x,y)::Tuple{Float64, Float64}
    Fx, Fy = -2*x, -2*y
    return Fx,Fy
end
x=-5:0.01:5
y=-5:0.01:5
##
X = rand(x,100)
Y = rand(y,100)
V = F.(X,Y)
Force=Array{Float64}(undef,100,2)
for i in 1:100
    fx,fy=df(X[i],Y[i])
    Force[i,1]=fx
    Force[i,2]=fy
end
##
Pred = hcat(V,Force)
##
mZero = MeanZero()
kern  = SE(0.0,0.0) #+ Matern(5/2,[0.0,0.0],0.0)
logObsNoise = -1.0
Input = Array{Float64}(undef,(2,100))
for i in 1:100
    Input[1,i] = X[i]
    Input[2,i] = Y[i]
end
##
#gp3dv = GP(Input, V,mZero, kern)#, logObsNoise )
#gp3dfx = GP(Input, Force[:,1], mZero, kern, logObsNoise)
#gp3dfy = GP(Input, Force[:,2], mZero, kern, logObsNoise)
gp3dv = GP(Input, V,MeanConst(mean(V)), kern)#, logObsNoise )
##
optimize!(gp3dv;kern=false)
##
surface(gp3dv)
##
#=d, n = 2, 50;         #Dimension and number of observations
x = 2π * rand(d, n);                               #Predictors
y = vec(sin.(x[1,:]).*sin.(x[2,:])) + 0.05*rand(n);  #Responses
mZero = MeanZero()                             # Zero mean function
kern =  SE(0.0,0.0) #+ Matern(5/2,[0.0,0.0],0.0)
gp = GP(x,y,mZero,kern,-2.0)
plot(gp)=#
##
A=Array{Float64}(undef,(2,1000))
A[1,:] = copy(rand(x,1000))
A[2,:] = copy(rand(y,1000))
vpred2, er2 = predict_y(gp3dv,A)
##
sqerr = sum((vpred2 .- F.(A[1,:],A[2,:])).^2)/100
print(sqerr^0.5)
