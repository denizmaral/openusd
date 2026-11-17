//! WebAssembly bindings that let the TopoJSON viewer read and write USD.
//!
//! [`wasm::parse_usd_meshes`] reads one USDC or USDA layer and returns its meshes as JSON;
//! [`wasm::build_usdc`] writes a USDC layer from the viewer's export JSON.

mod compat;
mod export;
pub mod wasm;

/// The layer access and value shapes the mesh extraction code reads through.
pub mod sdf {
    pub use crate::compat::{AbstractData, Layer, ListOp, Value};
    pub use openusd::sdf::{path, Path};
}
