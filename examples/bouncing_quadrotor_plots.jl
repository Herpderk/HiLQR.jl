using Plots

trnss = vcat([
    10
    09
    10
    11
    10
    10
    09
    09
    09
    09
], 9*ones(11))

Jss = vcat([
    6.39e-01
    5.00e-01
    3.47e-01
    2.51e-01
    2.27e-01
    2.09e-01
    2.06e-01
    2.06e-01
    2.06e-01
    2.06e-01
], 2.06e-01ones(11))

trnms = [
    015
    001
    001
    001
    001
    001
    001
    001
    001
    001
    001
    001
    001
    001
    001
    001
    001
    001
    001
    001
    001
]

Jms = [
    5.51e-01
    3.35e-01
    2.94e-01
    2.67e-01
    2.52e-01
    2.21e-01
    2.02e-01
    1.89e-01
    1.78e-01
    1.63e-01
    1.56e-01
    1.50e-01
    1.47e-01
    1.45e-01
    1.40e-01
    1.38e-01
    1.30e-01
    1.29e-01
    1.27e-01
    1.26e-01
    1.24e-01
]


p = plot(1:21, trnss, label="SS-HiLQR", linewidth=4, layout=(2,1))
plot!(1:21, trnms, label="MS-HiLQR",  ylabel="Number of Transitions", linewidth=4, subplot=1, xformatter=_->"")

plot!(1:21, Jss, label="SS-HiLQR", xlabel="HiLQR Iteration", ylabel="Trajectory Cost", linewidth=4, subplot=2)
plot!(1:21, Jms, label="MS-HiLQR", xlabel="HiLQR Iteration", ylabel="Trajectory Cost", linewidth=4, yaxis=:log, subplot=2, link=:xaxis)
