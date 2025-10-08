"""
Device taxonomy for SHTnsKit.

Defines the `Device` enumeration and helpers that convert between symbolic
representations used across the configuration and GPU extension layers.
"""

@enum Device begin
    CPU
    GPU
end

const _DEVICE_FROM_SYMBOL = Dict{Symbol, Device}(:cpu => CPU, :cuda => GPU, :gpu => GPU)
const _SYMBOL_FROM_DEVICE = Dict{Device, Symbol}(CPU => :cpu, GPU => :cuda)

"""Return the canonical backend symbol for a `Device`."""
device_symbol(device::Device) = _SYMBOL_FROM_DEVICE[device]

"""Derive the logical `Device` from a backend symbol."""
device_from_symbol(sym::Symbol) = get(_DEVICE_FROM_SYMBOL, sym, CPU)

_normalize_device_entry(entry::Device) = device_symbol(entry)
_normalize_device_entry(entry::Symbol) = entry

"""Check whether a functional CUDA backend is available."""
function cuda_available()
    isdefined(Main, :CUDA) && try
        Main.CUDA.functional()
    catch
        false
    end
end
