using Plots

τss = reshape(vcat([
    000
    001
    001
    001
    001
    001
    000
    001
    001
    003
    003
    003
    003
    003
    003
    003
    005
    006
    005
    005
    005
    005
    005
    007
    004
    005
    004
    004
    004
    004
    004
    004
    004
    004
    004
    004
], 4*ones(5,1)), 41)
τms = reshape([
    002
    001
    001
    000
    002
    002
    002
    002
    002
    002
    002
    002
    002
    002
    002
    002
    002
    002
    002
    001
    001
    002
    000
    001
    000
    001
    000
    001
    000
    000
    000
    001
    001
    000
    001
    000
    000
    000
    000
    000
    000
], 41)

p = plot(1:41, τss, label="SS-HiLQR", linewidth=4, layout=(2,1))
plot!(1:41, τms, label="MS-HiLQR",  ylabel="Number of Transitions", linewidth=4, subplot=1, xformatter=_->"")

Jss = reshape(vcat([
    3.74e+00
    3.32e+00
    2.86e+00
    2.50e+00
    1.97e+00
    1.53e+00
    8.10e-01
    6.44e-01
    5.41e-01
    4.56e-01
    4.44e-01
    4.39e-01
    4.38e-01
    4.38e-01
    4.38e-01
    4.38e-01
    2.38e+00
    5.85e-01
    4.26e-01
    3.88e-01
    3.66e-01
    3.61e-01
    3.58e-01
    3.50e-01
    3.31e-01
    3.29e-01
    3.27e-01
    3.24e-01
    3.23e-01
    3.23e-01
    3.23e-01
    3.23e-01
    3.23e-01
    3.23e-01
    3.23e-01
    3.23e-01
], 3.23e-01*ones(5)), 41)
Jms = reshape([
    4.47e+00
    4.10e+00
    3.96e+00
    2.73e+00
    1.54e+00
    7.35e-01
    6.34e-01
    4.14e-01
    3.63e-01
    2.91e-01
    2.77e-01
    2.73e-01
    2.72e-01
    2.71e-01
    2.71e-01
    2.71e-01
    2.71e-01
    2.71e-01
    4.21e+00
    2.17e+00
    1.24e+00
    8.40e-01
    5.75e-01
    5.69e-01
    4.82e-01
    3.85e-01
    3.69e-01
    2.73e-01
    2.48e-01
    2.43e-01
    2.41e-01
    2.37e-01
    2.07e-01
    1.96e-01
    1.86e-01
    1.86e-01
    1.86e-01
    1.86e-01
    1.86e-01
    1.86e-01
    1.86e-01
], 41)

plot!(1:41, Jss, label="SS-HiLQR", xlabel="HiLQR Iteration", ylabel="Trajectory Cost", linewidth=4, subplot=2)
plot!(1:41, Jms, label="MS-HiLQR", xlabel="HiLQR Iteration", ylabel="Trajectory Cost", linewidth=4, yaxis=:log, subplot=2, link=:xaxis)
