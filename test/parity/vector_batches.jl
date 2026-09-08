"""Compare resident vector/QST batches with independent unbatched CPU calls."""
function run_gpu_vector_batch_parity(adapter::VectorParityAdapter)
    @testset "vector/QST batch synthesis parity" begin
        for T in (Float32, Float64), mres in (1, 2), real_output in (false, true)
            cfg = create_gauss_config(3, 8; nlon=10, mres)
            S = zeros(Complex{T}, 4, 4, 2)
            S[2, 1, 1] = T(0.25)
            S[3, 3, 2] = Complex{T}(0.1, 0.2)
            for nonzero in (true, false)
                nonzero || fill!(S, 0)
                Q, Tlm = 3S, 2S
                device_Q = vector_place(adapter, cfg, Q, :coefficients)
                device_S = vector_place(adapter, cfg, S, :coefficients)
                device_T = vector_place(adapter, cfg, Tlm, :coefficients)
                vector = synthesis_sphtor_batch(cfg, device_S, device_T; real_output)
                qst = synthesis_qst_batch(cfg, device_Q, device_S, device_T; real_output)
                for component in (vector..., qst...)
                    vector_resident(adapter, component)
                end
                vector_host = map(value -> vector_collect(adapter, value, cfg), vector)
                qst_host = map(value -> vector_collect(adapter, value, cfg), qst)
                tolerance = T === Float32 ? 3e-5 : 2e-12
                for k in axes(S, 3)
                    expected = synthesis_qst(CPU(), cfg, Q[:, :, k], S[:, :, k], Tlm[:, :, k]; real_output)
                    for component in 1:2
                        @test vector_host[component][:, :, k] ≈ expected[component + 1] atol=tolerance rtol=tolerance
                    end
                    for component in 1:3
                        @test qst_host[component][:, :, k] ≈ expected[component] atol=tolerance rtol=tolerance
                    end
                end
            end
        end
    end
end
