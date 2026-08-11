module GPUCommon

using KernelAbstractions

export laplacian_kernel!

# Device-neutral kernels live here. Vendor extensions own array placement,
# FFT libraries, synchronization, device selection, and runtime inspection.
@kernel function laplacian_kernel!(output, input, lmax, mmax)
    l, m = @index(Global, NTuple)
    if l <= lmax + 1 && m <= mmax + 1
        l_val = l - 1
        m_val = m - 1
        if l_val >= m_val
            output[l, m] = -l_val * (l_val + 1) * input[l, m]
        end
    end
end

end # module GPUCommon
