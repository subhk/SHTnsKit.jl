#!/usr/bin/env julia
# Pure mathematical test of DH weights - no SHTnsKit dependencies

println("=== DH Weights Mathematical Verification ===\n")

# Manually compute DH weights based on SHTOOLS formula
function compute_dh_weights(n::Int)
    @assert iseven(n) "n must be even"

    w = zeros(Float64, n)
    norm_factor = sqrt(8.0) / n

    for j in 0:(n-1)
        sum1 = 0.0
        for l in 0:(n÷2-1)
            sum1 += sin((2*l + 1) * π * j / n) / (2*l + 1)
        end
        w[j+1] = norm_factor * sin(π * j / n) * sum1
    end

    return w
end

# Test with n=18 (lmax=8 case)
n = 18
println("Computing DH weights for n = $n")
w = compute_dh_weights(n)

println("\nResults:")
println("  Length: $(length(w))")
println("  Sum: $(sum(w))")
println("  Expected sum: $(2*sqrt(2)) = $(round(2*sqrt(2), digits=10))")
println("  Error in sum: $(abs(sum(w) - 2*sqrt(2)))")
println()

println("First 10 weights:")
for i in 1:10
    println("  w[$i] = $(w[i])")
end
println()

println("Last 5 weights:")
for i in (n-4):n
    println("  w[$i] = $(w[i])")
end
println()

println("Properties:")
println("  w[1] (north pole): $(w[1])")
println("  w[1] ≈ 0? $(abs(w[1]) < 1e-14)")
println("  w[end]: $(w[end])")
println("  w[end] ≈ 0? $(abs(w[end]) < 1e-14)")
println("  Max weight: $(maximum(w))")
println("  Min weight: $(minimum(w))")
println("  All non-negative? $(all(w .>= 0))")
println()

# Verify symmetry property (if it exists)
println("Symmetry check:")
for i in 2:(n÷2)
    w_sym = w[n - i + 2]  # Symmetric point
    diff = abs(w[i] - w_sym)
    if diff > 1e-10
        println("  w[$i] vs w[$(n-i+2)]: difference = $diff")
    end
end
println()

# Check if weights integrate to correct value
# ∫₀^π sin(θ) dθ = 2, so with proper normalization we should get 2√2
integral = sum(w)
expected = 2 * sqrt(2)
println("Integration check:")
println("  Σ w[j] = $integral")
println("  Expected (2√2) = $expected")
println("  Relative error: $(abs(integral - expected) / expected)")

if abs(integral - expected) < 1e-10
    println("  ✅ PASS: Weights sum correctly!")
else
    println("  ❌ FAIL: Weights do not sum correctly!")
    println()
    println("This suggests a bug in the DH weights formula.")
end
println()

# Test with smaller n to see pattern
println("Testing smaller n values:")
for test_n in [4, 6, 8, 10]
    w_test = compute_dh_weights(test_n)
    sum_test = sum(w_test)
    err_test = abs(sum_test - 2*sqrt(2))
    println("  n=$test_n: sum=$(round(sum_test, digits=8)), error=$(err_test)")
end

println("\n=== Verification Complete ===")
