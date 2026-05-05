//! Minimal USDC (binary crate) writer for mesh + material export.
//!
//! Produces a valid `.usdc` file that can be read by the existing reader
//! and by USD tools (usdview, Blender, etc.).

use std::collections::HashMap;

use anyhow::{ensure, Result};
use bytemuck::bytes_of;

use super::coding;
use super::layout::{self, Bootstrap, Section, Type};
use crate::sdf;

/// Target USDC version: 0.8.0
/// - Supports compressed structural sections (>= 0.4.0)
/// - Array element counts as u64 (>= 0.7.0)
/// - Payload layer offsets (>= 0.8.0)
const WRITE_VERSION: layout::Version = layout::version(0, 8, 0);

/// A minimal USDC binary writer.
///
/// Usage:
/// 1. Register all tokens, fields, fieldsets, paths, and specs
/// 2. Call `serialize()` to produce the binary output
pub struct CrateWriter {
    /// Deduplicated token table.
    tokens: Vec<String>,
    token_map: HashMap<String, usize>,

    /// String table (indices into tokens).
    strings: Vec<u32>,

    /// Fields: (token_index, value_rep_u64).
    fields: Vec<(u32, u64)>,

    /// Fieldsets: groups of field indices, None = terminator.
    fieldsets: Vec<Option<u32>>,

    /// Ordered paths for the path table.
    paths: Vec<sdf::Path>,
    path_map: HashMap<String, usize>,

    /// Specs: (path_index, fieldset_index, spec_type).
    specs: Vec<(u32, u32, sdf::SpecType)>,

    /// Out-of-line data buffer. Written right after the bootstrap header.
    data_buf: Vec<u8>,
}

impl CrateWriter {
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            token_map: HashMap::new(),
            strings: Vec::new(),
            fields: Vec::new(),
            fieldsets: Vec::new(),
            paths: Vec::new(),
            path_map: HashMap::new(),
            specs: Vec::new(),
            data_buf: Vec::new(),
        }
    }

    // ── Token management ──

    /// Intern a token string, returning its index. Deduplicates.
    pub fn intern_token(&mut self, s: &str) -> usize {
        if let Some(&idx) = self.token_map.get(s) {
            return idx;
        }
        let idx = self.tokens.len();
        self.tokens.push(s.to_string());
        self.token_map.insert(s.to_string(), idx);
        idx
    }

    /// Intern a string into the strings table (returns string index).
    pub fn intern_string(&mut self, s: &str) -> u32 {
        let token_idx = self.intern_token(s) as u32;
        // Check if already in strings table
        for (i, &ti) in self.strings.iter().enumerate() {
            if ti == token_idx {
                return i as u32;
            }
        }
        let idx = self.strings.len() as u32;
        self.strings.push(token_idx);
        idx
    }

    // ── Path management ──

    /// Register a path, returning its index.
    pub fn register_path(&mut self, path: &sdf::Path) -> usize {
        let key = path.as_str().to_string();
        if let Some(&idx) = self.path_map.get(&key) {
            return idx;
        }
        let idx = self.paths.len();
        self.paths.push(path.clone());
        self.path_map.insert(key, idx);
        idx
    }

    // ── Field management ──

    /// Add a field (name + value_rep), returning its index.
    pub fn add_field(&mut self, name: &str, value_rep: u64) -> usize {
        let token_idx = self.intern_token(name) as u32;
        let idx = self.fields.len();
        self.fields.push((token_idx, value_rep));
        idx
    }

    // ── Fieldset management ──

    /// Start a new fieldset. Returns the starting index in the fieldsets array.
    /// Call `add_field_to_fieldset` for each field, then `end_fieldset`.
    pub fn begin_fieldset(&mut self) -> usize {
        self.fieldsets.len()
    }

    /// Add a field index to the current fieldset.
    pub fn add_field_to_fieldset(&mut self, field_idx: usize) {
        self.fieldsets.push(Some(field_idx as u32));
    }

    /// Terminate the current fieldset with a None sentinel.
    pub fn end_fieldset(&mut self) {
        self.fieldsets.push(None);
    }

    // ── Spec management ──

    /// Register a spec.
    pub fn add_spec(&mut self, path_idx: usize, fieldset_idx: usize, spec_type: sdf::SpecType) {
        self.specs.push((path_idx as u32, fieldset_idx as u32, spec_type));
    }

    // ── ValueRep construction ──

    const ARRAY_BIT: u64 = 1 << 63;
    const INLINED_BIT: u64 = 1 << 62;
    const COMPRESSED_BIT: u64 = 1 << 61;
    const PAYLOAD_MASK: u64 = (1 << 48) - 1;

    /// Construct a ValueRep with the given type and payload, marked as inlined.
    pub fn make_inline(ty: Type, payload: u64) -> u64 {
        let type_bits = ((ty as u64) & 0xFF) << 48;
        type_bits | Self::INLINED_BIT | (payload & Self::PAYLOAD_MASK)
    }

    /// Construct a ValueRep pointing to out-of-line data at the given offset.
    pub fn make_offset(ty: Type, is_array: bool, offset: u64) -> u64 {
        let type_bits = ((ty as u64) & 0xFF) << 48;
        let mut bits = type_bits | (offset & Self::PAYLOAD_MASK);
        if is_array {
            bits |= Self::ARRAY_BIT;
        }
        bits
    }

    /// Construct a compressed array ValueRep.
    pub fn make_compressed_offset(ty: Type, offset: u64) -> u64 {
        let type_bits = ((ty as u64) & 0xFF) << 48;
        type_bits | Self::ARRAY_BIT | Self::COMPRESSED_BIT | (offset & Self::PAYLOAD_MASK)
    }

    // ── Data buffer helpers (out-of-line values) ──

    /// The data buffer starts right after the 32-byte bootstrap header.
    const DATA_START: u64 = 32;

    /// Current absolute file offset for the next data write.
    fn data_offset(&self) -> u64 {
        Self::DATA_START + self.data_buf.len() as u64
    }

    /// Write an inline token value (stores token index as payload).
    pub fn write_token_value(&mut self, field_name: &str, token_value: &str) -> usize {
        let token_idx = self.intern_token(token_value) as u64;
        self.add_field(field_name, Self::make_inline(Type::Token, token_idx))
    }

    /// Write an inline bool value.
    pub fn write_bool_value(&mut self, field_name: &str, val: bool) -> usize {
        self.add_field(field_name, Self::make_inline(Type::Bool, val as u64))
    }

    /// Write an inline i32 specifier value.
    pub fn write_specifier_value(&mut self, field_name: &str, spec: sdf::Specifier) -> usize {
        self.add_field(field_name, Self::make_inline(Type::Specifier, spec as u64))
    }

    /// Write an inline variability value.
    pub fn write_variability_value(&mut self, field_name: &str, var: sdf::Variability) -> usize {
        self.add_field(field_name, Self::make_inline(Type::Variability, var as u64))
    }

    /// Write an inline float value.
    pub fn write_float_value(&mut self, field_name: &str, val: f32) -> usize {
        let bits = val.to_bits() as u64;
        self.add_field(field_name, Self::make_inline(Type::Float, bits))
    }

    /// Write an inline double value (stored out-of-line since it's 8 bytes).
    pub fn write_double_value(&mut self, field_name: &str, val: f64) -> usize {
        let offset = self.data_offset();
        self.data_buf.extend_from_slice(&val.to_le_bytes());
        self.add_field(field_name, Self::make_offset(Type::Double, false, offset))
    }

    /// Write a token vector (out-of-line: u64 count + u32[] token indices).
    pub fn write_token_vec_value(&mut self, field_name: &str, tokens: &[&str]) -> usize {
        let offset = self.data_offset();
        // Count
        self.data_buf.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
        // Token indices
        for t in tokens {
            let idx = self.intern_token(t) as u32;
            self.data_buf.extend_from_slice(&idx.to_le_bytes());
        }
        self.add_field(field_name, Self::make_offset(Type::TokenVector, false, offset))
    }

    /// Minimum element count to enable compression (matches USD reader).
    const MIN_COMPRESSED_ARRAY_SIZE: usize = 4;

    /// Write an array of Vec3f (out-of-line, uncompressed).
    /// USD doesn't support compression for vector array types (Vec3f, Vec2f).
    pub fn write_vec3f_array(&mut self, field_name: &str, data: &[f32]) -> usize {
        let offset = self.data_offset();
        let count = data.len() / 3;
        self.data_buf.extend_from_slice(&(count as u64).to_le_bytes());
        for &v in data { self.data_buf.extend_from_slice(&v.to_le_bytes()); }
        self.add_field(field_name, Self::make_offset(Type::Vec3f, true, offset))
    }

    /// Write an array of Vec2f (out-of-line, uncompressed).
    pub fn write_vec2f_array(&mut self, field_name: &str, data: &[f32]) -> usize {
        let offset = self.data_offset();
        let count = data.len() / 2;
        self.data_buf.extend_from_slice(&(count as u64).to_le_bytes());
        for &v in data { self.data_buf.extend_from_slice(&v.to_le_bytes()); }
        self.add_field(field_name, Self::make_offset(Type::Vec2f, true, offset))
    }

    /// Write an i32 array with USD integer coding + LZ4 compression.
    /// Format: u64 count + encoded_compressed(i32[] data).
    pub fn write_int_array(&mut self, field_name: &str, data: &[i32]) -> usize {
        if data.len() < Self::MIN_COMPRESSED_ARRAY_SIZE {
            let offset = self.data_offset();
            self.data_buf.extend_from_slice(&(data.len() as u64).to_le_bytes());
            for &v in data { self.data_buf.extend_from_slice(&v.to_le_bytes()); }
            return self.add_field(field_name, Self::make_offset(Type::Int, true, offset));
        }
        let offset = self.data_offset();
        self.data_buf.extend_from_slice(&(data.len() as u64).to_le_bytes());
        Self::write_encoded_ints(&mut self.data_buf, data);
        self.add_field(field_name, Self::make_compressed_offset(Type::Int, offset))
    }

    /// Write a Vec3f scalar (out-of-line: 3 floats).
    pub fn write_vec3f_scalar(&mut self, field_name: &str, rgb: [f32; 3]) -> usize {
        let offset = self.data_offset();
        for &v in &rgb {
            self.data_buf.extend_from_slice(&v.to_le_bytes());
        }
        self.add_field(field_name, Self::make_offset(Type::Vec3f, false, offset))
    }

    /// Write an asset path value (out-of-line: token index as u32).
    /// Write an inline string value (stored as string table index).
    pub fn write_string_value(&mut self, field_name: &str, val: &str) -> usize {
        let string_idx = self.intern_string(val) as u64;
        self.add_field(field_name, Self::make_inline(Type::String, string_idx))
    }

    pub fn write_asset_path_value(&mut self, field_name: &str, path: &str) -> usize {
        let offset = self.data_offset();
        let token_idx = self.intern_token(path) as u64;
        self.data_buf.extend_from_slice(&token_idx.to_le_bytes());
        self.add_field(field_name, Self::make_offset(Type::AssetPath, false, offset))
    }

    /// Write token-valued time samples (e.g. visibility: "invisible"/"inherited").
    /// Each sample is (time_code, token_value).
    pub fn write_token_time_samples(&mut self, field_name: &str, samples: &[(f64, &str)]) -> usize {
        // 1. Write the times as a Double array in data_buf
        let times_data_offset = self.data_offset();
        self.data_buf.extend_from_slice(&(samples.len() as u64).to_le_bytes());
        for &(t, _) in samples {
            self.data_buf.extend_from_slice(&t.to_le_bytes());
        }
        let times_rep = Self::make_offset(Type::Double, true, times_data_offset);

        // 2. Build inlined Token ValueReps for each value
        let value_reps: Vec<u64> = samples.iter().map(|&(_, tok)| {
            let idx = self.intern_token(tok) as u64;
            Self::make_inline(Type::Token, idx)
        }).collect();

        // 3. Write the TimeSamples structure:
        //    i64(8) → ValueRep(times) → i64(8) → u64(count) → ValueRep[count]
        let ts_offset = self.data_offset();
        self.data_buf.extend_from_slice(&8i64.to_le_bytes());       // recursive offset to times rep
        self.data_buf.extend_from_slice(&times_rep.to_le_bytes());  // times ValueRep
        self.data_buf.extend_from_slice(&8i64.to_le_bytes());       // recursive offset to values
        self.data_buf.extend_from_slice(&(samples.len() as u64).to_le_bytes()); // count
        for &rep in &value_reps {
            self.data_buf.extend_from_slice(&rep.to_le_bytes());    // value ValueReps
        }

        self.add_field(field_name, Self::make_offset(Type::TimeSamples, false, ts_offset))
    }

    /// Write a PathListOp with explicit items (for material:binding and connectionPaths).
    pub fn write_path_list_op_explicit(&mut self, field_name: &str, path_indices: &[u32]) -> usize {
        let offset = self.data_offset();
        // ListOpHeader: IS_EXPLICIT | HAS_EXPLICIT_ITEMS = 0b11 = 3
        self.data_buf.push(3u8);
        // Write path vector: count + indices
        self.data_buf.extend_from_slice(&(path_indices.len() as u64).to_le_bytes());
        for &idx in path_indices {
            self.data_buf.extend_from_slice(&idx.to_le_bytes());
        }
        self.add_field(field_name, Self::make_offset(Type::PathListOp, false, offset))
    }

    // ── Path tree encoding ──

    /// Build the compressed path arrays for serialization.
    /// Returns (path_indexes, element_token_indexes, jumps) ready for integer encoding.
    fn build_path_tree(&self) -> (Vec<u32>, Vec<i32>, Vec<i32>) {
        // Build a tree from the flat path list.
        // Each path is either "/" (root) or has a parent + element name.

        #[derive(Debug)]
        struct PathNode {
            path_index: usize,     // index into self.paths
            element_token: i32,    // token index (negative = property)
            children: Vec<usize>,  // indices into nodes vec
        }

        let mut nodes: Vec<PathNode> = Vec::new();
        let mut path_to_node: HashMap<String, usize> = HashMap::new();

        // First, sort paths in a deterministic order (DFS-compatible).
        // Root always first.
        let mut ordered_paths: Vec<usize> = Vec::new();

        // Find the root path ("/")
        let root_idx = self.paths.iter().position(|p| p.as_str() == "/");
        if let Some(ri) = root_idx {
            ordered_paths.push(ri);
        }

        // Build parent-child relationships
        // For each non-root path, find parent and element name
        struct PathInfo {
            path_idx: usize,
            parent: String,
            element: String,
            is_property: bool,
        }

        let mut infos: Vec<PathInfo> = Vec::new();
        for (i, path) in self.paths.iter().enumerate() {
            if path.as_str() == "/" {
                continue;
            }

            let s = path.as_str();

            // Determine if this is a property path.
            // A property path has a '.' after the last '/'.
            // Examples: /Root/Mesh.points, /Root/Mesh.primvars:displayColor
            let last_slash = s.rfind('/').unwrap_or(0);
            let after_last_slash = &s[last_slash..];
            let is_property = after_last_slash.contains('.');

            if is_property {
                // Split at the first '.' after the last '/'
                let dot_pos = last_slash + after_last_slash.find('.').unwrap();
                let parent = &s[..dot_pos];
                let element = &s[dot_pos + 1..];
                infos.push(PathInfo {
                    path_idx: i,
                    parent: parent.to_string(),
                    element: element.to_string(),
                    is_property: true,
                });
            } else {
                // Prim path - split at last '/'
                let slash_pos = last_slash;
                let parent = if slash_pos == 0 { "/" } else { &s[..slash_pos] };
                let element = &s[slash_pos + 1..];
                infos.push(PathInfo {
                    path_idx: i,
                    parent: parent.to_string(),
                    element: element.to_string(),
                    is_property: false,
                });
            }
        }

        // Build tree using DFS order
        // Root node
        if let Some(ri) = root_idx {
            let node_idx = nodes.len();
            nodes.push(PathNode {
                path_index: ri,
                element_token: 0, // root has no element token
                children: Vec::new(),
            });
            path_to_node.insert("/".to_string(), node_idx);
        }

        // Add all paths - we need multiple passes since parents may not be added yet
        let mut remaining = infos;
        let mut max_iterations = remaining.len() + 1;
        while !remaining.is_empty() && max_iterations > 0 {
            max_iterations -= 1;
            let mut next_remaining = Vec::new();
            for info in remaining {
                if path_to_node.contains_key(&info.parent) {
                    let parent_node_idx = path_to_node[&info.parent];
                    let token_idx = self.token_map.get(&info.element).copied().unwrap_or(0);
                    let element_token = if info.is_property {
                        -(token_idx as i32)
                    } else {
                        token_idx as i32
                    };
                    let node_idx = nodes.len();
                    nodes.push(PathNode {
                        path_index: info.path_idx,
                        element_token,
                        children: Vec::new(),
                    });
                    nodes[parent_node_idx].children.push(node_idx);
                    let path_str = self.paths[info.path_idx].as_str().to_string();
                    path_to_node.insert(path_str, node_idx);
                } else {
                    next_remaining.push(info);
                }
            }
            remaining = next_remaining;
        }

        // DFS traversal to produce the three arrays.
        //
        // The reader's algorithm processes entries sequentially in a loop:
        // - Entry at index I is the current node
        // - If has_child, the FIRST child is always at index I+1 (next loop iteration)
        // - If has_sibling, the sibling is at index I+jump (handled by recursion)
        //
        // Jump semantics:
        //   jump > 0: has child AND has sibling (sibling at I+jump)
        //   jump == -1: has child, NO sibling
        //   jump < -1 (e.g. -2): NO child, NO sibling => loop breaks
        //   jump == 0: NO child, has sibling (sibling at I+0 — shouldn't happen normally)
        //
        // For a node with children [C0, C1, C2]:
        //   - Node is written, jump = -1 (has child, no sibling — at this level)
        //   - C0 is written next. C0's jump encodes both its own children AND its sibling C1
        //   - C1 is written after C0's subtree. C1's jump encodes children + sibling C2
        //   - C2 is last sibling: jump = -1 if it has children, -2 if leaf

        let mut path_indexes: Vec<u32> = Vec::new();
        let mut element_token_indexes: Vec<i32> = Vec::new();
        let mut jumps: Vec<i32> = Vec::new();

        /// Write a node and all its descendants.
        /// The caller is responsible for setting the jump of this node
        /// (to encode sibling relationships).
        /// Returns the position of this node in the output arrays.
        fn write_subtree(
            node_idx: usize,
            nodes: &[PathNode],
            path_indexes: &mut Vec<u32>,
            element_token_indexes: &mut Vec<i32>,
            jumps: &mut Vec<i32>,
        ) -> usize {
            let node = &nodes[node_idx];
            let this_pos = path_indexes.len();

            path_indexes.push(node.path_index as u32);
            element_token_indexes.push(node.element_token);
            jumps.push(0); // placeholder, set below or by caller

            let children = &node.children;

            if children.is_empty() {
                // Leaf node: no child, no sibling (sibling is handled by caller)
                jumps[this_pos] = -2;
            } else {
                // Has children. First child is at this_pos + 1.
                // Process children as a sibling chain.
                for (ci, &child_idx) in children.iter().enumerate() {
                    let child_pos = write_subtree(child_idx, nodes, path_indexes, element_token_indexes, jumps);

                    let is_last_sibling = ci + 1 >= children.len();
                    let child_has_children = !nodes[child_idx].children.is_empty();

                    if !is_last_sibling {
                        if child_has_children {
                            // Has child AND has sibling: jump > 0 (offset to sibling)
                            let next_sibling_pos = path_indexes.len();
                            let offset = (next_sibling_pos - child_pos) as i32;
                            jumps[child_pos] = offset;
                        } else {
                            // Leaf with sibling: jump = 0.
                            // Reader: has_child=false (0 is not >0 and not -1),
                            //         has_sibling=true (0 >= 0).
                            // The loop continues to current_index (next entry)
                            // with the SAME parent_path — exactly what we want.
                            // The next sibling MUST be written at child_pos+1.
                            jumps[child_pos] = 0;
                        }
                    } else {
                        // Last sibling: no more siblings
                        if child_has_children {
                            jumps[child_pos] = -1; // has child, no sibling
                        } else {
                            jumps[child_pos] = -2; // no child, no sibling => break
                        }
                    }
                }

                // This node has children: jump = -1 (has child, no sibling at this level)
                // The caller may override this if this node itself has a sibling.
                jumps[this_pos] = -1;
            }

            this_pos
        }

        if !nodes.is_empty() {
            write_subtree(0, &nodes, &mut path_indexes, &mut element_token_indexes, &mut jumps);
        }

        (path_indexes, element_token_indexes, jumps)
    }

    // ── LZ4 compression ──

    /// Compress data using TfFastCompression format (chunk_count=0 + LZ4 block).
    fn compress_lz4(data: &[u8]) -> Vec<u8> {
        let compressed = lz4_flex::compress(data);
        let mut output = Vec::with_capacity(1 + compressed.len());
        output.push(0u8); // 0 chunks = single block
        output.extend_from_slice(&compressed);
        output
    }

    /// Write compressed data: u64 compressed_size + compressed_bytes.
    fn write_compressed(out: &mut Vec<u8>, data: &[u8]) {
        let compressed = Self::compress_lz4(data);
        out.extend_from_slice(&(compressed.len() as u64).to_le_bytes());
        out.extend_from_slice(&compressed);
    }

    /// Encode integers and write them as compressed data.
    fn write_encoded_ints<T: num_traits::PrimInt + 'static>(out: &mut Vec<u8>, values: &[T])
    where
        T: num_traits::AsPrimitive<i64>,
    {
        let encoded = coding::encode_ints(values);
        Self::write_compressed(out, &encoded);
    }

    // ── Serialization ──

    /// Serialize the entire crate to a binary USDC file.
    pub fn serialize(&self) -> Result<Vec<u8>> {
        ensure!(!self.paths.is_empty(), "No paths registered");

        let mut output = Vec::new();

        // 1. Write bootstrap header (32 bytes) with placeholder TOC offset
        let mut bootstrap = Bootstrap::default();
        bootstrap.ident = *b"PXR-USDC";
        bootstrap.version[0] = WRITE_VERSION.major;
        bootstrap.version[1] = WRITE_VERSION.minor;
        bootstrap.version[2] = WRITE_VERSION.patch;
        bootstrap.toc_offset = 0; // patched later

        output.extend_from_slice(bytes_of(&bootstrap));
        debug_assert_eq!(output.len(), 32);

        // 2. Write out-of-line data buffer
        output.extend_from_slice(&self.data_buf);

        // 3. Write sections, tracking their offsets
        let mut sections: Vec<Section> = Vec::new();

        // ── TOKENS section ──
        {
            let section_start = output.len() as u64;

            // Token count
            output.extend_from_slice(&(self.tokens.len() as u64).to_le_bytes());

            // Build null-separated token string
            let mut token_bytes = Vec::new();
            for t in &self.tokens {
                token_bytes.extend_from_slice(t.as_bytes());
                token_bytes.push(0);
            }

            // Uncompressed size
            output.extend_from_slice(&(token_bytes.len() as u64).to_le_bytes());

            // LZ4-compressed token data
            Self::write_compressed(&mut output, &token_bytes);

            let section_size = output.len() as u64 - section_start;
            sections.push(make_section("TOKENS", section_start, section_size));
        }

        // ── STRINGS section ──
        {
            let section_start = output.len() as u64;

            // Count
            output.extend_from_slice(&(self.strings.len() as u64).to_le_bytes());
            // u32 indices
            for &idx in &self.strings {
                output.extend_from_slice(&idx.to_le_bytes());
            }

            let section_size = output.len() as u64 - section_start;
            sections.push(make_section("STRINGS", section_start, section_size));
        }

        // ── FIELDS section ──
        {
            let section_start = output.len() as u64;

            let field_count = self.fields.len();
            output.extend_from_slice(&(field_count as u64).to_le_bytes());

            // Compressed token indices
            let indices: Vec<u32> = self.fields.iter().map(|(ti, _)| *ti).collect();
            Self::write_encoded_ints(&mut output, &indices);

            // Compressed value reps (as raw u64 bytes)
            let reps_bytes: Vec<u8> = self.fields
                .iter()
                .flat_map(|(_, vr)| vr.to_le_bytes())
                .collect();
            Self::write_compressed(&mut output, &reps_bytes);

            let section_size = output.len() as u64 - section_start;
            sections.push(make_section("FIELDS", section_start, section_size));
        }

        // ── FIELDSETS section ──
        {
            let section_start = output.len() as u64;

            let count = self.fieldsets.len();
            output.extend_from_slice(&(count as u64).to_le_bytes());

            let values: Vec<u32> = self.fieldsets
                .iter()
                .map(|opt| opt.unwrap_or(u32::MAX))
                .collect();
            Self::write_encoded_ints(&mut output, &values);

            let section_size = output.len() as u64 - section_start;
            sections.push(make_section("FIELDSETS", section_start, section_size));
        }

        // ── PATHS section ──
        {
            let section_start = output.len() as u64;

            let path_count = self.paths.len();
            output.extend_from_slice(&(path_count as u64).to_le_bytes());

            // Build compressed path tree
            let (path_indexes, element_token_indexes, jumps) = self.build_path_tree();
            let encoded_count = path_indexes.len();
            output.extend_from_slice(&(encoded_count as u64).to_le_bytes());

            Self::write_encoded_ints(&mut output, &path_indexes);
            Self::write_encoded_ints(&mut output, &element_token_indexes);
            Self::write_encoded_ints(&mut output, &jumps);

            let section_size = output.len() as u64 - section_start;
            sections.push(make_section("PATHS", section_start, section_size));
        }

        // ── SPECS section ──
        {
            let section_start = output.len() as u64;

            let spec_count = self.specs.len();
            output.extend_from_slice(&(spec_count as u64).to_le_bytes());

            let path_indices: Vec<u32> = self.specs.iter().map(|(p, _, _)| *p).collect();
            let fieldset_indices: Vec<u32> = self.specs.iter().map(|(_, f, _)| *f).collect();
            let spec_types: Vec<u32> = self.specs.iter().map(|(_, _, t)| *t as u32).collect();

            Self::write_encoded_ints(&mut output, &path_indices);
            Self::write_encoded_ints(&mut output, &fieldset_indices);
            Self::write_encoded_ints(&mut output, &spec_types);

            let section_size = output.len() as u64 - section_start;
            sections.push(make_section("SPECS", section_start, section_size));
        }

        // ── Table of Contents ──
        let toc_offset = output.len() as u64;
        // Section count
        output.extend_from_slice(&(sections.len() as u64).to_le_bytes());
        // Section entries
        for section in &sections {
            output.extend_from_slice(bytes_of(section));
        }

        // Patch TOC offset in bootstrap header
        let toc_bytes = toc_offset.to_le_bytes();
        output[16..24].copy_from_slice(&toc_bytes);

        Ok(output)
    }
}

/// Create a Section struct with the given name, start, and size.
fn make_section(name: &str, start: u64, size: u64) -> Section {
    let mut sec = Section::default();
    let name_bytes = name.as_bytes();
    let len = name_bytes.len().min(15);
    // Safety: Section has a fixed 16-byte name field
    let sec_bytes: &mut [u8] = bytemuck::bytes_of_mut(&mut sec);
    sec_bytes[..len].copy_from_slice(&name_bytes[..len]);
    // start is at offset 16, size at offset 24
    sec_bytes[16..24].copy_from_slice(&start.to_le_bytes());
    sec_bytes[24..32].copy_from_slice(&size.to_le_bytes());
    sec
}

// ── High-level scene builder ──

use serde::Deserialize;

#[derive(Deserialize)]
pub struct ExportScene {
    pub meshes: Vec<ExportMesh>,
    #[serde(default)]
    pub materials: Vec<ExportMaterial>,
    /// Frames per second for time samples (default 24.0).
    #[serde(default)]
    pub time_codes_per_second: Option<f64>,
    /// Start time code for animation range.
    #[serde(default)]
    pub start_time_code: Option<f64>,
    /// End time code for animation range.
    #[serde(default)]
    pub end_time_code: Option<f64>,
}

#[derive(Deserialize)]
pub struct ExportMesh {
    pub name: String,
    #[serde(default = "default_group")]
    pub group: String,
    pub points: Vec<f32>,       // flat [x,y,z, ...]
    pub indices: Vec<i32>,
    #[serde(default)]
    pub uvs: Option<Vec<f32>>,  // flat [u,v, ...]
    #[serde(default)]
    pub display_color: Option<[f32; 3]>,
    #[serde(default)]
    pub material_id: Option<usize>,
    /// Custom string attributes (e.g. IFC properties, TopoJSON fields).
    #[serde(default)]
    pub attributes: Option<HashMap<String, String>>,
    /// Visibility time samples: (time_code, "invisible"|"inherited").
    #[serde(default)]
    pub visibility_times: Option<Vec<(f64, String)>>,
}

fn default_group() -> String {
    "Elements".to_string()
}

#[derive(Deserialize)]
pub struct ExportMaterial {
    pub id: usize,
    #[serde(default)]
    pub diffuse_color: Option<[f32; 3]>,
    #[serde(default)]
    pub metallic: Option<f32>,
    #[serde(default)]
    pub roughness: Option<f32>,
    #[serde(default)]
    pub textures: HashMap<String, String>, // channel -> filename
}

/// Build a complete USDC binary from an ExportScene.
pub fn build_usdc_scene(scene: &ExportScene) -> Result<Vec<u8>> {
    let mut w = CrateWriter::new();

    // Pre-intern common tokens
    w.intern_token(""); // index 0

    // ── Register all paths first ──
    // Root "/"
    let root_path = sdf::Path::abs_root();
    let root_path_idx = w.register_path(&root_path);

    // Materials container
    let has_materials = !scene.materials.is_empty();

    // Collect all material paths
    struct MatPaths {
        mat_path_idx: usize,
        shader_path_idx: usize,
        mat_output_surface_prop_idx: usize,   // /Materials/Material_N.outputs:surface
        shader_output_prop_idx: usize,        // /Materials/Material_N/PBRShader.outputs:surface
        shader_info_id_prop_idx: usize,       // /Materials/Material_N/PBRShader.info:id
        tex_entries: Vec<TexPrimPaths>,
        st_reader_path_idx: Option<usize>,
        st_reader_id_prop_idx: Option<usize>,
        st_reader_varname_prop_idx: Option<usize>,
        st_reader_result_prop_idx: Option<usize>,
        // Connection targets: channel -> output_rgb_prop_idx of texture prim
        tex_output_rgb_by_channel: HashMap<String, usize>,
        // Additional PBRShader input property indices for normal/emissive/occlusion
        shader_normal_prop_idx: Option<usize>,
        shader_emissive_prop_idx: Option<usize>,
        shader_occlusion_prop_idx: Option<usize>,
    }
    struct TexPrimPaths {
        prim_path_idx: usize,
        id_prop_idx: usize,
        file_prop_idx: usize,
        st_prop_idx: usize,
        output_rgb_prop_idx: usize,
        wrap_s_prop_idx: usize,
        wrap_t_prop_idx: usize,
        _channel: String,
    }

    let materials_path_idx = if has_materials {
        let p = sdf::path("/Materials")?;
        Some(w.register_path(&p))
    } else {
        None
    };

    let mut mat_paths_vec: Vec<MatPaths> = Vec::new();

    for mat in &scene.materials {
        let mat_name = format!("Material_{}", mat.id);
        let mat_path = sdf::path(&format!("/Materials/{}", mat_name))?;
        let mat_path_idx = w.register_path(&mat_path);

        let shader_path = sdf::path(&format!("/Materials/{}/PBRShader", mat_name))?;
        let shader_path_idx = w.register_path(&shader_path);

        let shader_id_prop = mat_path.clone().append_property("outputs:surface")?;
        let shader_id_prop_idx = w.register_path(&shader_id_prop);

        let shader_output_prop = shader_path.append_property("outputs:surface")?;
        let shader_output_prop_idx = w.register_path(&shader_output_prop);

        let mat_output_prop = sdf::path(&format!("/Materials/{}/PBRShader.outputs:surface", mat_name))?;
        let _mat_output_prop_idx = w.register_path(&mat_output_prop);

        // info:id property on shader
        let shader_info_id = sdf::path(&format!("/Materials/{}/PBRShader.info:id", mat_name))?;
        let shader_info_id_idx = w.register_path(&shader_info_id);

        // Texture prims
        let tex_channels = [
            ("diffuse", "DiffuseTexture"),
            ("normal", "NormalTexture"),
            ("mr", "MRTexture"),
            ("emissive", "EmissiveTexture"),
            ("ao", "AOTexture"),
        ];

        let mut tex_entries = Vec::new();
        let mut has_any_tex = false;

        for (channel, prim_name) in &tex_channels {
            if !mat.textures.contains_key(*channel) {
                continue;
            }
            has_any_tex = true;

            let tex_path = sdf::path(&format!("/Materials/{}/{}", mat_name, prim_name))?;
            let tex_path_idx = w.register_path(&tex_path);

            let id_prop = tex_path.append_property("info:id")?;
            let id_prop_idx = w.register_path(&id_prop);

            let file_prop = tex_path.append_property("inputs:file")?;
            let file_prop_idx = w.register_path(&file_prop);

            let st_prop = tex_path.append_property("inputs:st")?;
            let st_prop_idx = w.register_path(&st_prop);

            let output_rgb_prop = tex_path.append_property("outputs:rgb")?;
            let output_rgb_prop_idx = w.register_path(&output_rgb_prop);

            let wrap_s_prop = tex_path.append_property("inputs:wrapS")?;
            let wrap_s_prop_idx = w.register_path(&wrap_s_prop);

            let wrap_t_prop = tex_path.append_property("inputs:wrapT")?;
            let wrap_t_prop_idx = w.register_path(&wrap_t_prop);

            tex_entries.push(TexPrimPaths {
                prim_path_idx: tex_path_idx,
                id_prop_idx,
                file_prop_idx,
                st_prop_idx,
                output_rgb_prop_idx,
                wrap_s_prop_idx,
                wrap_t_prop_idx,
                _channel: channel.to_string(),
            });
        }

        // ST reader
        let (st_reader_path_idx, st_reader_id_prop_idx, st_reader_varname_prop_idx, st_reader_result_prop_idx) = if has_any_tex {
            let st_path = sdf::path(&format!("/Materials/{}/Primvar_st", mat_name))?;
            let st_path_idx = w.register_path(&st_path);
            let st_id = st_path.append_property("info:id")?;
            let st_id_idx = w.register_path(&st_id);
            let st_var = st_path.append_property("inputs:varname")?;
            let st_var_idx = w.register_path(&st_var);
            let st_result = st_path.append_property("outputs:result")?;
            let st_result_idx = w.register_path(&st_result);
            (Some(st_path_idx), Some(st_id_idx), Some(st_var_idx), Some(st_result_idx))
        } else {
            (None, None, None, None)
        };

        // Build channel -> output_rgb_prop_idx map for texture connections
        let mut tex_output_rgb_by_channel: HashMap<String, usize> = HashMap::new();
        for tex in &tex_entries {
            tex_output_rgb_by_channel.insert(tex._channel.clone(), tex.output_rgb_prop_idx);
        }

        // Additional shader properties for connections
        {
            let p = sdf::path(&format!("/Materials/{}/PBRShader.inputs:diffuseColor", mat_name))?;
            w.register_path(&p);
        }
        {
            let p = sdf::path(&format!("/Materials/{}/PBRShader.inputs:metallic", mat_name))?;
            w.register_path(&p);
        }
        {
            let p = sdf::path(&format!("/Materials/{}/PBRShader.inputs:roughness", mat_name))?;
            w.register_path(&p);
        }
        // Register normal/emissive/occlusion input paths when those textures exist
        let shader_normal_prop_idx = if mat.textures.contains_key("normal") {
            let p = sdf::path(&format!("/Materials/{}/PBRShader.inputs:normal", mat_name))?;
            Some(w.register_path(&p))
        } else { None };
        let shader_emissive_prop_idx = if mat.textures.contains_key("emissive") {
            let p = sdf::path(&format!("/Materials/{}/PBRShader.inputs:emissiveColor", mat_name))?;
            Some(w.register_path(&p))
        } else { None };
        let shader_occlusion_prop_idx = if mat.textures.contains_key("ao") {
            let p = sdf::path(&format!("/Materials/{}/PBRShader.inputs:occlusion", mat_name))?;
            Some(w.register_path(&p))
        } else { None };

        mat_paths_vec.push(MatPaths {
            mat_path_idx,
            shader_path_idx,
            mat_output_surface_prop_idx: shader_id_prop_idx,  // /Materials/Material_N.outputs:surface
            shader_output_prop_idx,                            // /Materials/Material_N/PBRShader.outputs:surface
            shader_info_id_prop_idx: shader_info_id_idx,       // /Materials/Material_N/PBRShader.info:id
            tex_entries,
            st_reader_path_idx,
            st_reader_id_prop_idx,
            st_reader_varname_prop_idx,
            st_reader_result_prop_idx,
            tex_output_rgb_by_channel,
            shader_normal_prop_idx,
            shader_emissive_prop_idx,
            shader_occlusion_prop_idx,
        });
    }

    // Root Xform
    let root_xform_path = sdf::path("/Root")?;
    let root_xform_idx = w.register_path(&root_xform_path);

    // Group Xforms + Mesh prims
    struct MeshPathInfo {
        _group_path_idx: usize,
        mesh_path_idx: usize,
    }

    let mut group_indices: HashMap<String, usize> = HashMap::new();
    let mut mesh_path_infos: Vec<MeshPathInfo> = Vec::new();

    for (i, mesh) in scene.meshes.iter().enumerate() {
        let group = &mesh.group;
        let group_path_idx = if let Some(&idx) = group_indices.get(group) {
            idx
        } else {
            let safe_group = usd_safe_name(group);
            let p = sdf::path(&format!("/Root/{}", safe_group))?;
            let idx = w.register_path(&p);
            group_indices.insert(group.clone(), idx);
            idx
        };

        let safe_name = format!("{}_{}", usd_safe_name(&mesh.name), i);
        let group_name = usd_safe_name(group);
        let mesh_path = sdf::path(&format!("/Root/{}/{}", group_name, safe_name))?;
        let mesh_path_idx = w.register_path(&mesh_path);

        // Register property paths for mesh attributes
        let props = ["points", "faceVertexIndices", "faceVertexCounts",
                      "subdivisionScheme", "doubleSided", "primvars:displayColor"];
        for prop in &props {
            let p = mesh_path.append_property(prop)?;
            w.register_path(&p);
        }

        if mesh.uvs.is_some() {
            let p = mesh_path.append_property("primvars:st")?;
            w.register_path(&p);
        }

        if mesh.material_id.is_some() {
            let p = mesh_path.append_property("material:binding")?;
            w.register_path(&p);
        }

        // Register custom attribute paths
        if let Some(ref attrs) = mesh.attributes {
            for key in attrs.keys() {
                let safe_key = usd_safe_name(key);
                let p = mesh_path.append_property(&safe_key)?;
                w.register_path(&p);
            }
        }

        // Register visibility property path if time samples exist
        if mesh.visibility_times.is_some() {
            let p = mesh_path.append_property("visibility")?;
            w.register_path(&p);
        }

        mesh_path_infos.push(MeshPathInfo {
            _group_path_idx: group_path_idx,
            mesh_path_idx,
        });
    }

    // ── Now build fields, fieldsets, and specs ──

    // PseudoRoot spec
    {
        let fs_start = w.begin_fieldset();
        let f = w.write_token_value("defaultPrim", "Root");
        w.add_field_to_fieldset(f);
        let f = w.write_double_value("metersPerUnit", 1.0);
        w.add_field_to_fieldset(f);
        let f = w.write_token_value("upAxis", "Y");
        w.add_field_to_fieldset(f);

        // Animation metadata
        if let Some(fps) = scene.time_codes_per_second {
            let f = w.write_double_value("timeCodesPerSecond", fps);
            w.add_field_to_fieldset(f);
            let f = w.write_double_value("framesPerSecond", fps);
            w.add_field_to_fieldset(f);
        }
        if let Some(start) = scene.start_time_code {
            let f = w.write_double_value("startTimeCode", start);
            w.add_field_to_fieldset(f);
        }
        if let Some(end) = scene.end_time_code {
            let f = w.write_double_value("endTimeCode", end);
            w.add_field_to_fieldset(f);
        }

        // primChildren: list top-level prims under root
        let mut children: Vec<&str> = Vec::new();
        if has_materials { children.push("Materials"); }
        children.push("Root");
        let f = w.write_token_vec_value("primChildren", &children);
        w.add_field_to_fieldset(f);

        w.end_fieldset();
        w.add_spec(root_path_idx, fs_start, sdf::SpecType::PseudoRoot);
    }

    // Materials container prim
    if has_materials {
        let fs_start = w.begin_fieldset();
        let f = w.write_specifier_value("specifier", sdf::Specifier::Def);
        w.add_field_to_fieldset(f);

        let mat_child_names: Vec<String> = scene.materials.iter()
            .map(|m| format!("Material_{}", m.id))
            .collect();
        let mat_child_refs: Vec<&str> = mat_child_names.iter().map(|s| s.as_str()).collect();
        let f = w.write_token_vec_value("primChildren", &mat_child_refs);
        w.add_field_to_fieldset(f);

        w.end_fieldset();
        w.add_spec(materials_path_idx.unwrap(), fs_start, sdf::SpecType::Prim);
    }

    // Material prims
    for (mi, mat) in scene.materials.iter().enumerate() {
        let mp = &mat_paths_vec[mi];

        // Material prim spec
        {
            let fs_start = w.begin_fieldset();
            let f = w.write_specifier_value("specifier", sdf::Specifier::Def);
            w.add_field_to_fieldset(f);
            let f = w.write_token_value("typeName", "Material");
            w.add_field_to_fieldset(f);

            // Children
            let tex_names: Vec<String> = mp.tex_entries.iter()
                .map(|t| {
                    // Extract prim name from path
                    let path_str = w.paths[t.prim_path_idx].as_str().to_string();
                    path_str.rsplit('/').next().unwrap_or("").to_string()
                })
                .collect();
            let mut all_children: Vec<String> = vec!["PBRShader".to_string()];
            all_children.extend(tex_names);
            if mp.st_reader_path_idx.is_some() {
                all_children.push("Primvar_st".to_string());
            }
            let child_refs: Vec<&str> = all_children.iter().map(|s| s.as_str()).collect();
            let f = w.write_token_vec_value("primChildren", &child_refs);
            w.add_field_to_fieldset(f);

            // Properties
            let f = w.write_token_vec_value("properties", &["outputs:surface"]);
            w.add_field_to_fieldset(f);

            w.end_fieldset();
            w.add_spec(mp.mat_path_idx, fs_start, sdf::SpecType::Prim);
        }

        // outputs:surface attribute on Material
        {
            let fs_start = w.begin_fieldset();
            let f = w.write_token_value("typeName", "token");
            w.add_field_to_fieldset(f);
            let f = w.write_variability_value("variability", sdf::Variability::Uniform);
            w.add_field_to_fieldset(f);

            // connectionPaths to PBRShader.outputs:surface
            let shader_output_path_idx = mp.shader_output_prop_idx as u32;
            let f = w.write_path_list_op_explicit("connectionPaths", &[shader_output_path_idx]);
            w.add_field_to_fieldset(f);

            w.end_fieldset();
            w.add_spec(mp.mat_output_surface_prop_idx, fs_start, sdf::SpecType::Attribute);
        }

        // PBRShader prim
        {
            let fs_start = w.begin_fieldset();
            let f = w.write_specifier_value("specifier", sdf::Specifier::Def);
            w.add_field_to_fieldset(f);
            let f = w.write_token_value("typeName", "Shader");
            w.add_field_to_fieldset(f);

            // Properties list — include additional inputs when textures exist
            let mut props: Vec<&str> = vec!["info:id", "inputs:diffuseColor", "inputs:metallic", "inputs:roughness", "outputs:surface"];
            if mp.shader_emissive_prop_idx.is_some() { props.push("inputs:emissiveColor"); }
            if mp.shader_normal_prop_idx.is_some() { props.push("inputs:normal"); }
            if mp.shader_occlusion_prop_idx.is_some() { props.push("inputs:occlusion"); }
            let f = w.write_token_vec_value("properties", &props);
            w.add_field_to_fieldset(f);

            w.end_fieldset();
            w.add_spec(mp.shader_path_idx, fs_start, sdf::SpecType::Prim);
        }

        // PBRShader.info:id attribute
        {
            let fs_start = w.begin_fieldset();
            let f = w.write_token_value("typeName", "token");
            w.add_field_to_fieldset(f);
            let f = w.write_variability_value("variability", sdf::Variability::Uniform);
            w.add_field_to_fieldset(f);
            let f = w.write_token_value("default", "UsdPreviewSurface");
            w.add_field_to_fieldset(f);
            w.end_fieldset();
            w.add_spec(mp.shader_info_id_prop_idx, fs_start, sdf::SpecType::Attribute);
        }

        // PBRShader.inputs:diffuseColor — with connectionPaths to DiffuseTexture if present
        {
            let prop_path = sdf::path(&format!("/Materials/Material_{}/PBRShader.inputs:diffuseColor", mat.id))?;
            let prop_idx = w.path_map[prop_path.as_str()];

            let fs_start = w.begin_fieldset();
            let f = w.write_token_value("typeName", "color3f");
            w.add_field_to_fieldset(f);
            if let Some(&tex_rgb_idx) = mp.tex_output_rgb_by_channel.get("diffuse") {
                let f = w.write_path_list_op_explicit("connectionPaths", &[tex_rgb_idx as u32]);
                w.add_field_to_fieldset(f);
            } else if let Some(color) = mat.diffuse_color {
                let f = w.write_vec3f_scalar("default", color);
                w.add_field_to_fieldset(f);
            }
            w.end_fieldset();
            w.add_spec(prop_idx, fs_start, sdf::SpecType::Attribute);
        }

        // PBRShader.inputs:metallic — with connectionPaths to MRTexture if present
        {
            let prop_path = sdf::path(&format!("/Materials/Material_{}/PBRShader.inputs:metallic", mat.id))?;
            let prop_idx = w.path_map[prop_path.as_str()];

            let fs_start = w.begin_fieldset();
            let f = w.write_token_value("typeName", "float");
            w.add_field_to_fieldset(f);
            if let Some(&tex_rgb_idx) = mp.tex_output_rgb_by_channel.get("mr") {
                let f = w.write_path_list_op_explicit("connectionPaths", &[tex_rgb_idx as u32]);
                w.add_field_to_fieldset(f);
            } else {
                let f = w.write_float_value("default", mat.metallic.unwrap_or(0.1));
                w.add_field_to_fieldset(f);
            }
            w.end_fieldset();
            w.add_spec(prop_idx, fs_start, sdf::SpecType::Attribute);
        }

        // PBRShader.inputs:roughness — with connectionPaths to MRTexture if present
        {
            let prop_path = sdf::path(&format!("/Materials/Material_{}/PBRShader.inputs:roughness", mat.id))?;
            let prop_idx = w.path_map[prop_path.as_str()];

            let fs_start = w.begin_fieldset();
            let f = w.write_token_value("typeName", "float");
            w.add_field_to_fieldset(f);
            if let Some(&tex_rgb_idx) = mp.tex_output_rgb_by_channel.get("mr") {
                let f = w.write_path_list_op_explicit("connectionPaths", &[tex_rgb_idx as u32]);
                w.add_field_to_fieldset(f);
            } else {
                let f = w.write_float_value("default", mat.roughness.unwrap_or(0.8));
                w.add_field_to_fieldset(f);
            }
            w.end_fieldset();
            w.add_spec(prop_idx, fs_start, sdf::SpecType::Attribute);
        }

        // PBRShader.inputs:normal (only when normal texture exists)
        if let Some(prop_idx) = mp.shader_normal_prop_idx {
            if let Some(&tex_rgb_idx) = mp.tex_output_rgb_by_channel.get("normal") {
                let fs_start = w.begin_fieldset();
                let f = w.write_token_value("typeName", "normal3f");
                w.add_field_to_fieldset(f);
                let f = w.write_path_list_op_explicit("connectionPaths", &[tex_rgb_idx as u32]);
                w.add_field_to_fieldset(f);
                w.end_fieldset();
                w.add_spec(prop_idx, fs_start, sdf::SpecType::Attribute);
            }
        }

        // PBRShader.inputs:emissiveColor (only when emissive texture exists)
        if let Some(prop_idx) = mp.shader_emissive_prop_idx {
            if let Some(&tex_rgb_idx) = mp.tex_output_rgb_by_channel.get("emissive") {
                let fs_start = w.begin_fieldset();
                let f = w.write_token_value("typeName", "color3f");
                w.add_field_to_fieldset(f);
                let f = w.write_path_list_op_explicit("connectionPaths", &[tex_rgb_idx as u32]);
                w.add_field_to_fieldset(f);
                w.end_fieldset();
                w.add_spec(prop_idx, fs_start, sdf::SpecType::Attribute);
            }
        }

        // PBRShader.inputs:occlusion (only when ao texture exists)
        if let Some(prop_idx) = mp.shader_occlusion_prop_idx {
            if let Some(&tex_rgb_idx) = mp.tex_output_rgb_by_channel.get("ao") {
                let fs_start = w.begin_fieldset();
                let f = w.write_token_value("typeName", "float");
                w.add_field_to_fieldset(f);
                let f = w.write_path_list_op_explicit("connectionPaths", &[tex_rgb_idx as u32]);
                w.add_field_to_fieldset(f);
                w.end_fieldset();
                w.add_spec(prop_idx, fs_start, sdf::SpecType::Attribute);
            }
        }

        // PBRShader.outputs:surface
        {
            let fs_start = w.begin_fieldset();
            let f = w.write_token_value("typeName", "token");
            w.add_field_to_fieldset(f);
            w.end_fieldset();
            w.add_spec(mp.shader_output_prop_idx, fs_start, sdf::SpecType::Attribute);
        }

        // Texture prims
        for tex in &mp.tex_entries {
            // Texture prim
            {
                let fs_start = w.begin_fieldset();
                let f = w.write_specifier_value("specifier", sdf::Specifier::Def);
                w.add_field_to_fieldset(f);
                let f = w.write_token_value("typeName", "Shader");
                w.add_field_to_fieldset(f);
                let f = w.write_token_vec_value("properties", &[
                    "info:id", "inputs:file", "inputs:st", "inputs:wrapS", "inputs:wrapT", "outputs:rgb"
                ]);
                w.add_field_to_fieldset(f);
                w.end_fieldset();
                w.add_spec(tex.prim_path_idx, fs_start, sdf::SpecType::Prim);
            }

            // info:id
            {
                let fs_start = w.begin_fieldset();
                let f = w.write_token_value("typeName", "token");
                w.add_field_to_fieldset(f);
                let f = w.write_variability_value("variability", sdf::Variability::Uniform);
                w.add_field_to_fieldset(f);
                let f = w.write_token_value("default", "UsdUVTexture");
                w.add_field_to_fieldset(f);
                w.end_fieldset();
                w.add_spec(tex.id_prop_idx, fs_start, sdf::SpecType::Attribute);
            }

            // inputs:file
            {
                let filename = mat.textures.get(&tex._channel).unwrap();
                let fs_start = w.begin_fieldset();
                let f = w.write_token_value("typeName", "asset");
                w.add_field_to_fieldset(f);
                let f = w.write_asset_path_value("default", filename);
                w.add_field_to_fieldset(f);
                w.end_fieldset();
                w.add_spec(tex.file_prop_idx, fs_start, sdf::SpecType::Attribute);
            }

            // inputs:wrapS
            {
                let fs_start = w.begin_fieldset();
                let f = w.write_token_value("typeName", "token");
                w.add_field_to_fieldset(f);
                let f = w.write_token_value("default", "repeat");
                w.add_field_to_fieldset(f);
                w.end_fieldset();
                w.add_spec(tex.wrap_s_prop_idx, fs_start, sdf::SpecType::Attribute);
            }

            // inputs:wrapT
            {
                let fs_start = w.begin_fieldset();
                let f = w.write_token_value("typeName", "token");
                w.add_field_to_fieldset(f);
                let f = w.write_token_value("default", "repeat");
                w.add_field_to_fieldset(f);
                w.end_fieldset();
                w.add_spec(tex.wrap_t_prop_idx, fs_start, sdf::SpecType::Attribute);
            }

            // inputs:st (connection to Primvar_st)
            if let Some(st_result_idx) = mp.st_reader_result_prop_idx {
                let fs_start = w.begin_fieldset();
                let f = w.write_token_value("typeName", "float2");
                w.add_field_to_fieldset(f);
                let f = w.write_path_list_op_explicit("connectionPaths", &[st_result_idx as u32]);
                w.add_field_to_fieldset(f);
                w.end_fieldset();
                w.add_spec(tex.st_prop_idx, fs_start, sdf::SpecType::Attribute);
            }

            // outputs:rgb
            {
                let fs_start = w.begin_fieldset();
                let f = w.write_token_value("typeName", "float3");
                w.add_field_to_fieldset(f);
                w.end_fieldset();
                w.add_spec(tex.output_rgb_prop_idx, fs_start, sdf::SpecType::Attribute);
            }
        }

        // Primvar_st reader
        if let (Some(st_path_idx), Some(st_id_idx), Some(st_var_idx), Some(st_result_idx)) = (
            mp.st_reader_path_idx,
            mp.st_reader_id_prop_idx,
            mp.st_reader_varname_prop_idx,
            mp.st_reader_result_prop_idx,
        ) {
            // Prim
            {
                let fs_start = w.begin_fieldset();
                let f = w.write_specifier_value("specifier", sdf::Specifier::Def);
                w.add_field_to_fieldset(f);
                let f = w.write_token_value("typeName", "Shader");
                w.add_field_to_fieldset(f);
                let f = w.write_token_vec_value("properties", &["info:id", "inputs:varname", "outputs:result"]);
                w.add_field_to_fieldset(f);
                w.end_fieldset();
                w.add_spec(st_path_idx, fs_start, sdf::SpecType::Prim);
            }
            // info:id
            {
                let fs_start = w.begin_fieldset();
                let f = w.write_token_value("typeName", "token");
                w.add_field_to_fieldset(f);
                let f = w.write_variability_value("variability", sdf::Variability::Uniform);
                w.add_field_to_fieldset(f);
                let f = w.write_token_value("default", "UsdPrimvarReader_float2");
                w.add_field_to_fieldset(f);
                w.end_fieldset();
                w.add_spec(st_id_idx, fs_start, sdf::SpecType::Attribute);
            }
            // inputs:varname
            {
                let fs_start = w.begin_fieldset();
                let f = w.write_token_value("typeName", "string");
                w.add_field_to_fieldset(f);
                let string_idx = w.intern_string("st");
                let f = w.add_field("default", CrateWriter::make_inline(Type::String, string_idx as u64));
                w.add_field_to_fieldset(f);
                w.end_fieldset();
                w.add_spec(st_var_idx, fs_start, sdf::SpecType::Attribute);
            }
            // outputs:result
            {
                let fs_start = w.begin_fieldset();
                let f = w.write_token_value("typeName", "float2");
                w.add_field_to_fieldset(f);
                w.end_fieldset();
                w.add_spec(st_result_idx, fs_start, sdf::SpecType::Attribute);
            }
        }
    }

    // Root Xform prim
    {
        let fs_start = w.begin_fieldset();
        let f = w.write_specifier_value("specifier", sdf::Specifier::Def);
        w.add_field_to_fieldset(f);
        let f = w.write_token_value("typeName", "Xform");
        w.add_field_to_fieldset(f);

        // Children: group names
        let mut group_names: Vec<String> = Vec::new();
        let mut seen_groups: HashMap<String, bool> = HashMap::new();
        for mesh in &scene.meshes {
            let g = usd_safe_name(&mesh.group);
            if !seen_groups.contains_key(&g) {
                seen_groups.insert(g.clone(), true);
                group_names.push(g);
            }
        }
        let group_refs: Vec<&str> = group_names.iter().map(|s| s.as_str()).collect();
        let f = w.write_token_vec_value("primChildren", &group_refs);
        w.add_field_to_fieldset(f);

        w.end_fieldset();
        w.add_spec(root_xform_idx, fs_start, sdf::SpecType::Prim);
    }

    // Group Xform prims
    let mut group_mesh_names: HashMap<String, Vec<String>> = HashMap::new();
    for (i, mesh) in scene.meshes.iter().enumerate() {
        let safe_name = format!("{}_{}", usd_safe_name(&mesh.name), i);
        let group = usd_safe_name(&mesh.group);
        group_mesh_names.entry(group).or_default().push(safe_name);
    }

    for (group_name, mesh_names) in &group_mesh_names {
        let group_path = sdf::path(&format!("/Root/{}", group_name))?;
        let group_idx = w.path_map[group_path.as_str()];

        let fs_start = w.begin_fieldset();
        let f = w.write_specifier_value("specifier", sdf::Specifier::Def);
        w.add_field_to_fieldset(f);
        let f = w.write_token_value("typeName", "Xform");
        w.add_field_to_fieldset(f);

        let child_refs: Vec<&str> = mesh_names.iter().map(|s| s.as_str()).collect();
        let f = w.write_token_vec_value("primChildren", &child_refs);
        w.add_field_to_fieldset(f);

        w.end_fieldset();
        w.add_spec(group_idx, fs_start, sdf::SpecType::Prim);
    }

    // Mesh prims
    for (i, mesh) in scene.meshes.iter().enumerate() {
        let mpi = &mesh_path_infos[i];
        let mesh_path = &w.paths[mpi.mesh_path_idx].clone();

        // Mesh prim spec
        let fs_start = w.begin_fieldset();
        let f = w.write_specifier_value("specifier", sdf::Specifier::Def);
        w.add_field_to_fieldset(f);
        let f = w.write_token_value("typeName", "Mesh");
        w.add_field_to_fieldset(f);

        // Properties list
        let mut prop_names_owned: Vec<String> = vec![
            "doubleSided".into(), "faceVertexCounts".into(), "faceVertexIndices".into(),
            "points".into(), "primvars:displayColor".into(), "subdivisionScheme".into(),
        ];
        if mesh.uvs.is_some() {
            prop_names_owned.push("primvars:st".into());
        }
        if mesh.material_id.is_some() {
            prop_names_owned.push("material:binding".into());
        }
        if let Some(ref attrs) = mesh.attributes {
            for key in attrs.keys() {
                prop_names_owned.push(usd_safe_name(key));
            }
        }
        if mesh.visibility_times.is_some() {
            prop_names_owned.push("visibility".into());
        }
        prop_names_owned.sort();
        let prop_refs: Vec<&str> = prop_names_owned.iter().map(|s| s.as_str()).collect();
        let f = w.write_token_vec_value("properties", &prop_refs);
        w.add_field_to_fieldset(f);

        w.end_fieldset();
        w.add_spec(mpi.mesh_path_idx, fs_start, sdf::SpecType::Prim);

        // points attribute
        {
            let prop_path = mesh_path.append_property("points")?;
            let prop_idx = w.path_map[prop_path.as_str()];

            let fs_start = w.begin_fieldset();
            let f = w.write_token_value("typeName", "point3f[]");
            w.add_field_to_fieldset(f);
            let f = w.write_vec3f_array("default", &mesh.points);
            w.add_field_to_fieldset(f);
            w.end_fieldset();
            w.add_spec(prop_idx, fs_start, sdf::SpecType::Attribute);
        }

        // faceVertexIndices
        {
            let prop_path = mesh_path.append_property("faceVertexIndices")?;
            let prop_idx = w.path_map[prop_path.as_str()];

            let fs_start = w.begin_fieldset();
            let f = w.write_token_value("typeName", "int[]");
            w.add_field_to_fieldset(f);
            let f = w.write_int_array("default", &mesh.indices);
            w.add_field_to_fieldset(f);
            w.end_fieldset();
            w.add_spec(prop_idx, fs_start, sdf::SpecType::Attribute);
        }

        // faceVertexCounts
        {
            let prop_path = mesh_path.append_property("faceVertexCounts")?;
            let prop_idx = w.path_map[prop_path.as_str()];

            let tri_count = mesh.indices.len() / 3;
            let counts: Vec<i32> = vec![3; tri_count];

            let fs_start = w.begin_fieldset();
            let f = w.write_token_value("typeName", "int[]");
            w.add_field_to_fieldset(f);
            let f = w.write_int_array("default", &counts);
            w.add_field_to_fieldset(f);
            w.end_fieldset();
            w.add_spec(prop_idx, fs_start, sdf::SpecType::Attribute);
        }

        // subdivisionScheme
        {
            let prop_path = mesh_path.append_property("subdivisionScheme")?;
            let prop_idx = w.path_map[prop_path.as_str()];

            let fs_start = w.begin_fieldset();
            let f = w.write_token_value("typeName", "token");
            w.add_field_to_fieldset(f);
            let f = w.write_variability_value("variability", sdf::Variability::Uniform);
            w.add_field_to_fieldset(f);
            let f = w.write_token_value("default", "none");
            w.add_field_to_fieldset(f);
            w.end_fieldset();
            w.add_spec(prop_idx, fs_start, sdf::SpecType::Attribute);
        }

        // doubleSided
        {
            let prop_path = mesh_path.append_property("doubleSided")?;
            let prop_idx = w.path_map[prop_path.as_str()];

            let fs_start = w.begin_fieldset();
            let f = w.write_token_value("typeName", "bool");
            w.add_field_to_fieldset(f);
            let f = w.write_variability_value("variability", sdf::Variability::Uniform);
            w.add_field_to_fieldset(f);
            let f = w.write_bool_value("default", true);
            w.add_field_to_fieldset(f);
            w.end_fieldset();
            w.add_spec(prop_idx, fs_start, sdf::SpecType::Attribute);
        }

        // primvars:displayColor
        {
            let prop_path = mesh_path.append_property("primvars:displayColor")?;
            let prop_idx = w.path_map[prop_path.as_str()];

            let color = mesh.display_color.unwrap_or([0.5, 0.5, 0.5]);

            let fs_start = w.begin_fieldset();
            let f = w.write_token_value("typeName", "color3f[]");
            w.add_field_to_fieldset(f);
            let f = w.write_vec3f_array("default", &color);
            w.add_field_to_fieldset(f);
            let f = w.write_token_value("interpolation", "constant");
            w.add_field_to_fieldset(f);
            w.end_fieldset();
            w.add_spec(prop_idx, fs_start, sdf::SpecType::Attribute);
        }

        // primvars:st (UVs)
        if let Some(ref uvs) = mesh.uvs {
            let prop_path = mesh_path.append_property("primvars:st")?;
            let prop_idx = w.path_map[prop_path.as_str()];

            let fs_start = w.begin_fieldset();
            let f = w.write_token_value("typeName", "texCoord2f[]");
            w.add_field_to_fieldset(f);
            let f = w.write_vec2f_array("default", uvs);
            w.add_field_to_fieldset(f);
            let f = w.write_token_value("interpolation", "vertex");
            w.add_field_to_fieldset(f);
            w.end_fieldset();
            w.add_spec(prop_idx, fs_start, sdf::SpecType::Attribute);
        }

        // material:binding
        if let Some(mat_id) = mesh.material_id {
            let prop_path = mesh_path.append_property("material:binding")?;
            let prop_idx = w.path_map[prop_path.as_str()];

            // Find the material path index
            let mat_path = sdf::path(&format!("/Materials/Material_{}", mat_id))?;
            if let Some(&mat_path_idx) = w.path_map.get(mat_path.as_str()) {
                let fs_start = w.begin_fieldset();
                let f = w.write_path_list_op_explicit("default", &[mat_path_idx as u32]);
                w.add_field_to_fieldset(f);
                w.end_fieldset();
                w.add_spec(prop_idx, fs_start, sdf::SpecType::Relationship);
            }
        }

        // Custom string attributes
        if let Some(ref attrs) = mesh.attributes {
            for (key, value) in attrs {
                let safe_key = usd_safe_name(key);
                let prop_path = mesh_path.append_property(&safe_key)?;
                let prop_idx = w.path_map[prop_path.as_str()];

                let fs_start = w.begin_fieldset();
                let f = w.write_token_value("typeName", "string");
                w.add_field_to_fieldset(f);
                let f = w.write_string_value("default", value);
                w.add_field_to_fieldset(f);
                let f = w.write_bool_value("custom", true);
                w.add_field_to_fieldset(f);
                w.end_fieldset();
                w.add_spec(prop_idx, fs_start, sdf::SpecType::Attribute);
            }
        }

        // Visibility time samples
        if let Some(ref vis_times) = mesh.visibility_times {
            let prop_path = mesh_path.append_property("visibility")?;
            let prop_idx = w.path_map[prop_path.as_str()];

            let samples: Vec<(f64, &str)> = vis_times.iter()
                .map(|(t, v)| (*t, v.as_str()))
                .collect();

            let fs_start = w.begin_fieldset();
            let f = w.write_token_value("typeName", "token");
            w.add_field_to_fieldset(f);
            let f = w.write_token_time_samples("timeSamples", &samples);
            w.add_field_to_fieldset(f);
            w.end_fieldset();
            w.add_spec(prop_idx, fs_start, sdf::SpecType::Attribute);
        }
    }

    w.serialize()
}

/// Sanitize a name for USD prim naming (alphanumeric + underscore).
fn usd_safe_name(name: &str) -> String {
    let mut safe: String = name
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' })
        .collect();

    if safe.is_empty() {
        safe = "_unnamed".to_string();
    } else if safe.starts_with(|c: char| c.is_ascii_digit()) {
        safe.insert(0, '_');
    }

    safe
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdf::AbstractData;
    use crate::usdc::CrateData;
    use std::io::Cursor;

    #[test]
    fn test_minimal_usdc_roundtrip() -> Result<()> {
        let scene = ExportScene {
            meshes: vec![ExportMesh {
                name: "TestCube".to_string(),
                group: "Elements".to_string(),
                // Simple triangle
                points: vec![
                    0.0, 0.0, 0.0,
                    1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                ],
                indices: vec![0, 1, 2],
                uvs: None,
                display_color: Some([1.0, 0.0, 0.0]),
                material_id: None,
                attributes: None,
                visibility_times: None,
            }],
            materials: vec![],
            time_codes_per_second: None,
            start_time_code: None,
            end_time_code: None,
        };

        let usdc_bytes = build_usdc_scene(&scene)?;

        // Verify header
        assert_eq!(&usdc_bytes[0..8], b"PXR-USDC");

        // Read it back with the existing reader
        let cursor = Cursor::new(&usdc_bytes);
        let mut data = CrateData::open(cursor, true)?;

        // Check layer metadata
        let default_prim = data.get(&sdf::Path::abs_root(), "defaultPrim")?;
        assert_eq!(default_prim.try_as_token_ref().unwrap(), "Root");

        let up_axis = data.get(&sdf::Path::abs_root(), "upAxis")?;
        assert_eq!(up_axis.try_as_token_ref().unwrap(), "Y");

        // Check primChildren
        let children = data.get(&sdf::Path::abs_root(), "primChildren")?
            .into_owned()
            .try_as_token_vec()
            .unwrap();
        assert_eq!(children, vec!["Root"]);

        Ok(())
    }

    #[test]
    fn test_usdc_with_mesh_data() -> Result<()> {
        let scene = ExportScene {
            meshes: vec![ExportMesh {
                name: "Tri".to_string(),
                group: "Geometry".to_string(),
                points: vec![
                    0.0, 0.0, 0.0,
                    1.0, 0.0, 0.0,
                    0.5, 1.0, 0.0,
                ],
                indices: vec![0, 1, 2],
                uvs: Some(vec![0.0, 0.0, 1.0, 0.0, 0.5, 1.0]),
                display_color: Some([0.8, 0.2, 0.1]),
                material_id: None,
                attributes: Some(HashMap::from([
                    ("IfcGUID".to_string(), "2O2Fr$t4X7Z8f0Q3MC_25F".to_string()),
                    ("level".to_string(), "Level 1".to_string()),
                ])),
                visibility_times: None,
            }],
            materials: vec![],
            time_codes_per_second: None,
            start_time_code: None,
            end_time_code: None,
        };

        let usdc_bytes = build_usdc_scene(&scene)?;

        // Read back
        let cursor = Cursor::new(&usdc_bytes);
        let mut data = CrateData::open(cursor, true)?;

        // Verify points exist on the mesh prim
        let points = data.get(&sdf::path("/Root/Geometry/Tri_0.points")?, "default")?;
        let pts = points.try_as_vec_3f_ref().unwrap();
        assert_eq!(pts.len(), 9); // 3 vertices * 3 components
        assert_eq!(pts[0], 0.0);
        assert_eq!(pts[3], 1.0);

        // Verify face vertex indices
        let fvi = data.get(&sdf::path("/Root/Geometry/Tri_0.faceVertexIndices")?, "default")?;
        let indices = fvi.into_owned().try_as_int_vec().unwrap();
        assert_eq!(indices, vec![0, 1, 2]);

        // Verify face vertex counts
        let fvc = data.get(&sdf::path("/Root/Geometry/Tri_0.faceVertexCounts")?, "default")?;
        let counts = fvc.into_owned().try_as_int_vec().unwrap();
        assert_eq!(counts, vec![3]);

        // Verify custom attributes
        let guid = data.get(&sdf::path("/Root/Geometry/Tri_0.IfcGUID")?, "default")?;
        assert_eq!(guid.try_as_string_ref().unwrap(), "2O2Fr$t4X7Z8f0Q3MC_25F");

        let level = data.get(&sdf::path("/Root/Geometry/Tri_0.level")?, "default")?;
        assert_eq!(level.try_as_string_ref().unwrap(), "Level 1");

        Ok(())
    }

    #[test]
    fn test_compressed_arrays_roundtrip() -> Result<()> {
        // Build a mesh with enough vertices/indices to trigger compression (>= 4 elements)
        let num_verts = 100;
        let mut points = Vec::new();
        for i in 0..num_verts {
            let f = i as f32;
            points.push(f * 0.1);
            points.push(f * 0.2);
            points.push(f * 0.3);
        }
        let mut indices = Vec::new();
        for i in (0..num_verts as i32 - 2).step_by(3) {
            indices.push(i);
            indices.push(i + 1);
            indices.push(i + 2);
        }
        let tri_count = indices.len() / 3;

        let scene = ExportScene {
            meshes: vec![ExportMesh {
                name: "CompressedMesh".to_string(),
                group: "Test".to_string(),
                points: points.clone(),
                indices: indices.clone(),
                uvs: None,
                display_color: Some([0.5, 0.5, 0.5]),
                material_id: None,
                attributes: None,
                visibility_times: None,
            }],
            materials: vec![],
            time_codes_per_second: None,
            start_time_code: None,
            end_time_code: None,
        };

        let usdc_bytes = build_usdc_scene(&scene)?;

        // Read back
        let cursor = Cursor::new(&usdc_bytes);
        let mut data = CrateData::open(cursor, true)?;

        // Verify points
        let pts_val = data.get(&sdf::path("/Root/Test/CompressedMesh_0.points")?, "default")?;
        let pts = pts_val.try_as_vec_3f_ref().unwrap();
        assert_eq!(pts.len(), num_verts * 3);
        for i in 0..pts.len() {
            assert!((pts[i] - points[i]).abs() < 1e-6,
                "Point mismatch at {}: got {} expected {}", i, pts[i], points[i]);
        }

        // Verify indices
        let idx_val = data.get(&sdf::path("/Root/Test/CompressedMesh_0.faceVertexIndices")?, "default")?;
        let idx = idx_val.into_owned().try_as_int_vec().unwrap();
        assert_eq!(idx, indices);

        // Verify face counts
        let fvc_val = data.get(&sdf::path("/Root/Test/CompressedMesh_0.faceVertexCounts")?, "default")?;
        let fvc = fvc_val.into_owned().try_as_int_vec().unwrap();
        assert_eq!(fvc.len(), tri_count);
        assert!(fvc.iter().all(|&c| c == 3));

        Ok(())
    }

    #[test]
    fn test_time_samples_roundtrip() -> Result<()> {
        let scene = ExportScene {
            meshes: vec![ExportMesh {
                name: "AnimMesh".to_string(),
                group: "Elements".to_string(),
                points: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                indices: vec![0, 1, 2],
                uvs: None,
                display_color: Some([1.0, 0.0, 0.0]),
                material_id: None,
                attributes: None,
                visibility_times: Some(vec![
                    (0.0, "invisible".to_string()),
                    (24.0, "inherited".to_string()),
                    (72.0, "invisible".to_string()),
                ]),
            }],
            materials: vec![],
            time_codes_per_second: Some(24.0),
            start_time_code: Some(0.0),
            end_time_code: Some(72.0),
        };

        let usdc_bytes = build_usdc_scene(&scene)?;

        // Read back
        let cursor = Cursor::new(&usdc_bytes);
        let mut data = CrateData::open(cursor, true)?;

        // Verify animation metadata
        let fps = data.get(&sdf::Path::abs_root(), "timeCodesPerSecond")?;
        assert_eq!(fps.into_owned().try_as_double().unwrap(), 24.0);

        let start = data.get(&sdf::Path::abs_root(), "startTimeCode")?;
        assert_eq!(start.into_owned().try_as_double().unwrap(), 0.0);

        let end = data.get(&sdf::Path::abs_root(), "endTimeCode")?;
        assert_eq!(end.into_owned().try_as_double().unwrap(), 72.0);

        // Verify visibility time samples
        let vis = data.get(&sdf::path("/Root/Elements/AnimMesh_0.visibility")?, "timeSamples")?;
        let samples = vis.into_owned().try_as_time_samples().unwrap();
        assert_eq!(samples.len(), 3);

        assert_eq!(samples[0].0, 0.0);
        assert_eq!(samples[0].1.try_as_token_ref().unwrap(), "invisible");

        assert_eq!(samples[1].0, 24.0);
        assert_eq!(samples[1].1.try_as_token_ref().unwrap(), "inherited");

        assert_eq!(samples[2].0, 72.0);
        assert_eq!(samples[2].1.try_as_token_ref().unwrap(), "invisible");

        Ok(())
    }

    #[test]
    fn test_usdc_material_no_texture_roundtrip() -> Result<()> {
        // Simple material with color only, no textures
        let scene = ExportScene {
            meshes: vec![ExportMesh {
                name: "ColoredMesh".to_string(),
                group: "Elements".to_string(),
                points: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                indices: vec![0, 1, 2],
                uvs: None,
                display_color: Some([1.0, 0.0, 0.0]),
                material_id: Some(0),
                attributes: None,
                visibility_times: None,
            }],
            materials: vec![ExportMaterial {
                id: 0,
                diffuse_color: Some([1.0, 0.0, 0.0]),
                metallic: Some(0.0),
                roughness: Some(0.8),
                textures: HashMap::new(),
            }],
            time_codes_per_second: None,
            start_time_code: None,
            end_time_code: None,
        };

        let usdc_bytes = build_usdc_scene(&scene)?;
        let cursor = Cursor::new(&usdc_bytes);
        let _data = CrateData::open(cursor, true)?;
        println!("Material-no-texture roundtrip OK, USDC size={}", usdc_bytes.len());
        Ok(())
    }

    #[test]
    fn test_usdc_texture_roundtrip() -> Result<()> {
        // Build a scene with textured materials and verify the reader can follow
        // PBRShader.inputs:diffuseColor → connectionPaths → DiffuseTexture.outputs:rgb → inputs:file
        let mut textures = HashMap::new();
        textures.insert("diffuse".to_string(), "tex_0_diffuse.png".to_string());
        textures.insert("normal".to_string(), "tex_0_normal.png".to_string());

        let scene = ExportScene {
            meshes: vec![ExportMesh {
                name: "TexturedMesh".to_string(),
                group: "Elements".to_string(),
                points: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                indices: vec![0, 1, 2],
                uvs: Some(vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]),
                display_color: Some([1.0, 1.0, 1.0]),
                material_id: Some(0),
                attributes: None,
                visibility_times: None,
            }],
            materials: vec![ExportMaterial {
                id: 0,
                diffuse_color: Some([1.0, 1.0, 1.0]),
                metallic: Some(0.0),
                roughness: Some(0.8),
                textures,
            }],
            time_codes_per_second: None,
            start_time_code: None,
            end_time_code: None,
        };

        let usdc_bytes = build_usdc_scene(&scene)?;

        // Parse back with the reader
        let result = crate::wasm::parse_usd_meshes_inner(&usdc_bytes)?;
        assert!(result.error.is_none(), "Parse error: {:?}", result.error);
        assert_eq!(result.meshes.len(), 1);

        let m = &result.meshes[0];
        println!("Round-trip diffuse={:?} normal={:?}", m.diffuse_texture, m.normal_texture);
        assert_eq!(m.diffuse_texture.as_deref(), Some("tex_0_diffuse.png"),
            "Diffuse texture path not found in round-trip. Got: {:?}", m.diffuse_texture);
        assert_eq!(m.normal_texture.as_deref(), Some("tex_0_normal.png"),
            "Normal texture path not found in round-trip. Got: {:?}", m.normal_texture);

        // Verify UVs survived
        assert!(m.texcoords.is_some(), "UVs should survive round-trip");
        assert_eq!(m.texcoords.as_ref().unwrap().len(), 6); // 3 verts * 2 components

        Ok(())
    }
}
