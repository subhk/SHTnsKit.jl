module SHTns37TestCapabilities

const BACKENDS = (
    :cpu, :cuda, :amdgpu, :mpi_cpu, :mpi_cuda, :mpi_amdgpu,
)

const CAPABILITIES = (
    :scalar_real_full, :scalar_complex_full, :scalar_l, :scalar_ml,
    :scalar_batch, :packed_storage, :sphtor_full, :sphtor_l,
    :sphtor_ml, :sphtor_batch, :qst_full, :qst_l, :qst_ml, :qst_batch,
    :point, :point_complex, :latitude, :latitude_complex,
    :qst_point, :qst_latitude, :gradient_point, :operators, :rotations,
)

const TESTFILES = Dict(
    :cpu => "test/parity/runtests_cpu.jl",
    :cuda => "test/gpu/cuda/runtests.jl",
    :amdgpu => "test/gpu/amdgpu/runtests.jl",
    :mpi_cpu => "test/parity/runtests_mpi.jl",
    :mpi_cuda => "test/gpu/cuda/mpi_runtests.jl",
    :mpi_amdgpu => "test/gpu/amdgpu/mpi_runtests.jl",
)

const ENTRYPOINTS = (
    :analysis, :synthesis, :synthesis_cplx,
    :analysis_batch, :synthesis_batch, :synthesis_batch_cplx,
    :analysis_packed, :synthesis_packed,
    :analysis_packed_l, :synthesis_packed_l,
    :analysis_packed_ml, :synthesis_packed_ml,
    :analysis_packed_cplx, :synthesis_packed_cplx,
    :analysis_axisym, :synthesis_axisym,
    :analysis_axisym_l, :synthesis_axisym_l,
    :synthesis_point, :synthesis_point_cplx,
    :analysis_sphtor, :synthesis_sphtor,
    :analysis_sphtor_cplx, :synthesis_sphtor_cplx,
    :analysis_sphtor_l, :synthesis_sphtor_l,
    :analysis_sphtor_ml, :synthesis_sphtor_ml,
    :analysis_sphtor_batch, :synthesis_sphtor_batch,
    :analysis_qst, :synthesis_qst,
    :analysis_qst_cplx, :synthesis_qst_cplx,
    :analysis_qst_l, :synthesis_qst_l,
    :analysis_qst_ml, :synthesis_qst_ml,
    :analysis_qst_batch, :synthesis_qst_batch,
    :SH_to_lat, :SH_to_lat_cplx,
    :SHqst_to_lat, :SHqst_to_point, :SH_to_grad_point,
    :synthesis_grad, :synthesis_grad_l, :synthesis_grad_ml,
    :mul_ct_matrix, :st_dt_matrix, :SH_mul_mx,
    :SH_Zrotate, :SH_Yrotate, :SH_Yrotate90, :SH_Xrotate90,
    :shtns_rotation_apply_cplx, :shtns_rotation_apply_real,
)

end
