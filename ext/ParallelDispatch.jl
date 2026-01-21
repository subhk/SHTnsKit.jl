##########
# Unified dispatch helpers for PencilArray inputs
##########

function SHTnsKit.synthesis(cfg::SHTnsKit.SHTConfig, Alm::PencilArray; 
                    prototype_θφ::PencilArray, real_output::Bool=true)

    return SHTnsKit.dist_synthesis(cfg, Alm; prototype_θφ, real_output)
end

function SHTnsKit.analysis(cfg::SHTnsKit.SHTConfig, fθφ::PencilArray;
                           use_rfft::Bool=false)
    return SHTnsKit.dist_analysis(cfg, fθφ; use_rfft)
end

##########
# Vector/QST dispatch for PencilArrays
##########

function SHTnsKit.spat_to_SHsphtor(cfg::SHTnsKit.SHTConfig, Vtθφ::PencilArray, Vpθφ::PencilArray; use_tables=cfg.use_plm_tables)
    return SHTnsKit.dist_spat_to_SHsphtor(cfg, Vtθφ, Vpθφ; use_tables)
end

function SHTnsKit.spat_to_SHqst(cfg::SHTnsKit.SHTConfig, Vrθφ::PencilArray, Vtθφ::PencilArray, Vpθφ::PencilArray)
    return SHTnsKit.dist_spat_to_SHqst(cfg, Vrθφ, Vtθφ, Vpθφ)
end
