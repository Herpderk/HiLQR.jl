using JuMP
using DiffOpt
using OSQP

n = 2 # variable dimension
m = 1; # no of inequality constraints

Q = [4.0 1.0; 1.0 2.0]
q = [1.0; 1.0]
G = [1.0 1.0;]
h = [-1.0;]   # initial values set

model = Model(OSQP.Optimizer)
@variable(model, x[1:2])

@constraint(model, cons[j in 1:1], sum(G[j, i] * x[i] for i in 1:2) <= h[j])

@objective(
    model,
    Min,
    1 / 2 * sum(Q[j, i] * x[i] * x[j] for i in 1:2, j in 1:2) +
    sum(q[i] * x[i] for i in 1:2)
)

optimize!(model)

