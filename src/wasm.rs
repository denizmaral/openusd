//! WebAssembly bindings for USD mesh extraction.
//!
//! Provides `parse_usd_meshes` which takes raw USD file bytes and returns
//! JSON-encoded mesh data (points, indices, colors, transforms, names)
//! ready for Three.js consumption.

use std::io::Cursor;

use anyhow::{Context, Result};
use serde::Serialize;
use wasm_bindgen::prelude::*;

use crate::sdf::{self, AbstractData, Value};
use crate::usda;
use crate::usdc;

#[derive(Serialize)]
struct MeshData {
    name: String,
    points: Vec<f32>,       // flat [x,y,z, x,y,z, ...]
    indices: Vec<i32>,       // triangle indices
    #[serde(skip_serializing_if = "Option::is_none")]
    display_color: Option<[f32; 3]>,
    double_sided: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    transform: Option<Vec<f64>>, // 4x4 matrix, 16 elements
    #[serde(skip_serializing_if = "Option::is_none")]
    diffuse_texture: Option<String>, // texture asset path for diffuse color
    #[serde(skip_serializing_if = "Option::is_none")]
    texcoords: Option<Vec<f32>>, // flat [u,v, u,v, ...] UV coordinates
}

#[derive(Serialize)]
struct ParseResult {
    meshes: Vec<MeshData>,
    up_axis: String,
    meters_per_unit: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

/// Parse USD binary (USDC) or text (USDA) file bytes and extract mesh data.
///
/// Returns JSON string with mesh data ready for Three.js.
#[wasm_bindgen]
pub fn parse_usd_meshes(data: &[u8]) -> String {
    match parse_usd_meshes_inner(data) {
        Ok(result) => serde_json::to_string(&result).unwrap_or_else(|e| {
            format!(r#"{{"meshes":[],"up_axis":"Y","meters_per_unit":1,"error":"JSON error: {}"}}"#, e)
        }),
        Err(e) => {
            let result = ParseResult {
                meshes: vec![],
                up_axis: "Y".to_string(),
                meters_per_unit: 1.0,
                error: Some(format!("{:#}", e)),
            };
            serde_json::to_string(&result).unwrap_or_else(|_| {
                format!(r#"{{"meshes":[],"up_axis":"Y","meters_per_unit":1,"error":"{}"}}"#, e)
            })
        }
    }
}

fn parse_usd_meshes_inner(data: &[u8]) -> Result<ParseResult> {
    // Detect format: USDC binary starts with "PXR-USDC", USDA is text
    let is_usdc = data.len() >= 8 && &data[0..8] == b"PXR-USDC";

    let mut abstract_data: Box<dyn AbstractData> = if is_usdc {
        let cursor = Cursor::new(data);
        let crate_data = usdc::CrateData::open(cursor, true)
            .context("Failed to parse USDC binary")?;
        Box::new(crate_data)
    } else {
        // Try as USDA text (strip UTF-8 BOM if present)
        let text = std::str::from_utf8(data).context("Invalid UTF-8 in USDA file")?;
        let text = text.strip_prefix('\u{FEFF}').unwrap_or(text);
        let mut parser = usda::parser::Parser::new(text);
        let parsed = parser.parse().context("Failed to parse USDA text")?;
        Box::new(usda::TextReader::from_data(parsed))
    };

    // Read stage metadata
    let root = sdf::Path::abs_root();
    let up_axis = get_string_field(&mut *abstract_data, &root, "upAxis")
        .unwrap_or_else(|| "Y".to_string());
    let meters_per_unit = get_double_field(&mut *abstract_data, &root, "metersPerUnit")
        .unwrap_or(1.0);

    // Walk prim hierarchy and extract meshes
    let mut meshes = Vec::new();
    let prim_children = get_token_vec_field(&mut *abstract_data, &root, "primChildren")
        .unwrap_or_default();

    for child_name in &prim_children {
        let child_path = sdf::path(&format!("/{child_name}"))
            .unwrap_or_else(|_| sdf::Path::abs_root());
        walk_prims(&mut *abstract_data, &child_path, child_name, None, &mut meshes);
    }

    Ok(ParseResult {
        meshes,
        up_axis,
        meters_per_unit,
        error: None,
    })
}

/// Recursively walk the prim hierarchy extracting mesh data.
fn walk_prims(
    data: &mut dyn AbstractData,
    path: &sdf::Path,
    name: &str,
    parent_transform: Option<&[f64; 16]>,
    meshes: &mut Vec<MeshData>,
) {
    // Read prim type
    let prim_type = get_string_field(data, path, "typeName").unwrap_or_default();

    // Check if this prim is a Mesh (has points + faceVertexIndices)
    let points_path = path.append_property("points").ok();
    let has_mesh = points_path
        .as_ref()
        .map(|p| data.has_spec(p))
        .unwrap_or(false);

    // Read transform if present
    let local_transform = read_xform_ops(data, path);
    let world_transform = match (&local_transform, parent_transform) {
        (Some(local), Some(parent)) => Some(multiply_4x4(parent, local)),
        (Some(local), None) => Some(*local),
        (None, Some(parent)) => Some(*parent),
        (None, None) => None,
    };

    if has_mesh {
        if let Some(mesh) = extract_mesh(data, path, name, world_transform.as_ref()) {
            meshes.push(mesh);
        }
    } else {
        // Check for parametric primitives
        match prim_type.as_str() {
            "Cube" | "Sphere" | "Cylinder" | "Cone" | "Capsule" => {
                if let Some(mesh) = extract_parametric(data, path, name, &prim_type, world_transform.as_ref()) {
                    meshes.push(mesh);
                }
            }
            _ => {}
        }
    }

    // Recurse into children
    let children = get_token_vec_field(data, path, "primChildren")
        .unwrap_or_default();

    for child_name in &children {
        let child_path_str = format!("{}/{}", path, child_name);
        if let Ok(child_path) = sdf::path(&child_path_str) {
            walk_prims(data, &child_path, child_name, world_transform.as_ref(), meshes);
        }
    }
}

/// Extract mesh data from a Mesh prim.
fn extract_mesh(
    data: &mut dyn AbstractData,
    path: &sdf::Path,
    name: &str,
    transform: Option<&[f64; 16]>,
) -> Option<MeshData> {
    // Read points (point3f[] or float3[])
    let points = read_points(data, path)?;

    // Read face vertex indices
    let face_vertex_indices = read_int_array(data, path, "faceVertexIndices")?;

    // Read face vertex counts (for triangulation)
    let face_vertex_counts = read_int_array(data, path, "faceVertexCounts");

    // Read UV coordinates and indices BEFORE triangulation
    let texcoords_raw = read_texcoords(data, path);
    let uv_indices_raw = read_texcoord_indices(data, path);

    // Build face-vertex UV mapping (one UV index per face vertex, pre-triangulation)
    // This array parallels face_vertex_indices: fv_uv_indices[i] = UV index for face vertex i
    let fv_uv_indices: Option<Vec<i32>> = if let Some(ref uv_data) = texcoords_raw {
        let uv_count = uv_data.len() / 2;
        let vert_count = points.len() / 3;
        if uv_count != vert_count && uv_count > 0 {
            // Face-varying: build UV index per face vertex
            Some(if let Some(ref idx) = uv_indices_raw {
                // Explicit UV index array
                idx.clone()
            } else {
                // No separate index array: UVs are in face-vertex order (0,1,2,3,...)
                (0..face_vertex_indices.len() as i32).collect()
            })
        } else {
            None // vertex-interpolated, no expansion needed
        }
    } else {
        None
    };

    // Triangulate vertex indices (and UV face-vertex indices in parallel)
    let (indices, tri_uv_indices) = match face_vertex_counts {
        Some(ref counts) if counts.iter().any(|&c| c != 3) => {
            let tri_idx = triangulate(&face_vertex_indices, counts);
            let tri_uv = fv_uv_indices.as_ref().map(|uv_idx| triangulate(uv_idx, counts));
            (tri_idx, tri_uv)
        }
        _ => {
            (face_vertex_indices, fv_uv_indices)
        }
    };

    // Read display color
    let display_color = read_display_color(data, path);

    // Read doubleSided
    let double_sided_path = path.append_property("doubleSided").ok();
    let double_sided = double_sided_path
        .and_then(|p| data.get(&p, "default").ok())
        .and_then(|v| match v.into_owned() {
            Value::Bool(b) => Some(b),
            _ => None,
        })
        .unwrap_or(false);

    // Read material binding to find diffuse texture
    let diffuse_texture = read_material_texture(data, path);

    // Check if this is a skinned mesh (has skel:skeleton binding)
    let is_skinned = path.append_property("skel:skeleton").ok()
        .map(|p| data.has_spec(&p))
        .unwrap_or(false)
        || path.append_property("skel:joints").ok()
            .map(|p| data.has_spec(&p))
            .unwrap_or(false);

    // Bake world transform into vertex positions.
    let points = if !is_skinned {
        if let Some(t) = transform {
            transform_points(&points, t)
        } else {
            points
        }
    } else {
        points
    };

    // Expand face-varying UVs: create one vertex per face-vertex occurrence
    let (final_points, final_indices, final_texcoords) =
        if let (Some(ref uv_data), Some(ref uv_tri_idx)) = (&texcoords_raw, &tri_uv_indices) {
            expand_face_varying(&points, &indices, uv_data, uv_tri_idx)
        } else {
            (points, indices, texcoords_raw)
        };

    Some(MeshData {
        name: name.to_string(),
        points: final_points,
        indices: final_indices,
        display_color,
        double_sided,
        transform: None,
        diffuse_texture,
        texcoords: final_texcoords,
    })
}

/// Read point3f[] points from a mesh prim.
fn read_points(data: &mut dyn AbstractData, prim_path: &sdf::Path) -> Option<Vec<f32>> {
    let prop_path = prim_path.append_property("points").ok()?;
    let value = data.get(&prop_path, "default").ok()?;
    match value.into_owned() {
        Value::Vec3f(v) => Some(v),
        Value::Vec3d(v) => Some(v.iter().map(|&d| d as f32).collect()),
        Value::Vec3h(v) => Some(v.iter().map(|h| h.to_f32()).collect()),
        Value::FloatVec(v) => Some(v),
        Value::DoubleVec(v) => Some(v.iter().map(|&d| d as f32).collect()),
        _ => None,
    }
}

/// Read int[] array from a mesh property.
fn read_int_array(data: &mut dyn AbstractData, prim_path: &sdf::Path, prop_name: &str) -> Option<Vec<i32>> {
    let prop_path = prim_path.append_property(prop_name).ok()?;
    let value = data.get(&prop_path, "default").ok()?;
    match value.into_owned() {
        Value::IntVec(v) => Some(v),
        Value::Int(v) => Some(vec![v]),
        _ => None,
    }
}

/// Read primvars:displayColor from a mesh prim.
fn read_display_color(data: &mut dyn AbstractData, prim_path: &sdf::Path) -> Option<[f32; 3]> {
    let prop_path = prim_path.append_property("primvars:displayColor").ok()?;
    let value = data.get(&prop_path, "default").ok()?;
    match value.into_owned() {
        Value::Vec3f(v) if v.len() >= 3 => Some([v[0], v[1], v[2]]),
        Value::Vec3d(v) if v.len() >= 3 => Some([v[0] as f32, v[1] as f32, v[2] as f32]),
        _ => None,
    }
}

/// Read primvars:st (texture coordinates) from a mesh prim.
/// Returns flat [u,v, u,v, ...] Vec<f32>.
fn read_texcoords(data: &mut dyn AbstractData, prim_path: &sdf::Path) -> Option<Vec<f32>> {
    // Try common UV primvar names (2-component)
    for prop_name in &["primvars:st", "primvars:st0", "primvars:UVMap", "primvars:UVW"] {
        if let Ok(prop_path) = prim_path.append_property(prop_name) {
            if let Ok(value) = data.get(&prop_path, "default") {
                let coords = match value.into_owned() {
                    Value::Vec2f(v) => Some(v),
                    Value::Vec2d(v) => Some(v.iter().map(|&d| d as f32).collect()),
                    Value::Vec2h(v) => Some(v.iter().map(|h| h.to_f32()).collect()),
                    _ => None,
                };
                if coords.is_some() {
                    return coords;
                }
            }
        }
    }
    // Try 3-component UV names (primvars:UVW) — extract only U,V
    for prop_name in &["primvars:UVW", "primvars:uvw"] {
        if let Ok(prop_path) = prim_path.append_property(prop_name) {
            if let Ok(value) = data.get(&prop_path, "default") {
                let uvw = match value.into_owned() {
                    Value::Vec3f(v) => Some(v),
                    Value::Vec3d(v) => Some(v.iter().map(|&d| d as f32).collect::<Vec<f32>>()),
                    _ => None,
                };
                if let Some(uvw) = uvw {
                    // Extract U,V from [u,v,w, u,v,w, ...] → [u,v, u,v, ...]
                    let mut uv = Vec::with_capacity(uvw.len() / 3 * 2);
                    for chunk in uvw.chunks_exact(3) {
                        uv.push(chunk[0]);
                        uv.push(chunk[1]);
                    }
                    return Some(uv);
                }
            }
        }
    }
    None
}

/// Read UV index array (primvars:st:indices, primvars:UVW:indices, etc.)
fn read_texcoord_indices(data: &mut dyn AbstractData, prim_path: &sdf::Path) -> Option<Vec<i32>> {
    for prop_name in &["primvars:st:indices", "primvars:st0:indices", "primvars:UVMap:indices", "primvars:UVW:indices"] {
        if let Some(idx) = read_int_array(data, prim_path, prop_name) {
            return Some(idx);
        }
    }
    None
}

/// Expand indexed geometry with face-varying UVs into non-indexed per-face-vertex data.
/// `uv_tri_indices` must be triangulated in parallel with `indices` so they correspond 1:1.
fn expand_face_varying(
    points: &[f32],
    indices: &[i32],
    uv_data: &[f32],
    uv_tri_indices: &[i32],
) -> (Vec<f32>, Vec<i32>, Option<Vec<f32>>) {
    let mut new_points = Vec::with_capacity(indices.len() * 3);
    let mut new_uvs = Vec::with_capacity(indices.len() * 2);
    let mut new_indices = Vec::with_capacity(indices.len());

    for (fi, (&pt_idx, &uv_idx)) in indices.iter().zip(uv_tri_indices.iter()).enumerate() {
        let vi = pt_idx as usize;
        // Copy position
        if vi * 3 + 2 < points.len() {
            new_points.push(points[vi * 3]);
            new_points.push(points[vi * 3 + 1]);
            new_points.push(points[vi * 3 + 2]);
        } else {
            new_points.extend_from_slice(&[0.0, 0.0, 0.0]);
        }
        // Copy UV
        let ui = uv_idx as usize;
        if ui * 2 + 1 < uv_data.len() {
            new_uvs.push(uv_data[ui * 2]);
            new_uvs.push(uv_data[ui * 2 + 1]);
        } else {
            new_uvs.extend_from_slice(&[0.0, 0.0]);
        }
        new_indices.push(fi as i32);
    }

    (new_points, new_indices, Some(new_uvs))
}

/// Follow material:binding → material → PreviewSurface → diffuseColor texture.
/// Only returns textures connected to inputs:diffuseColor, not normal maps etc.
fn read_material_texture(data: &mut dyn AbstractData, prim_path: &sdf::Path) -> Option<String> {
    // Read material:binding relationship target
    let binding_path = prim_path.append_property("material:binding").ok()?;
    let target_value = data.get(&binding_path, "targetPaths").ok()?;
    let material_path_str = match target_value.into_owned() {
        Value::PathListOp(list_op) => {
            let items = if !list_op.explicit_items.is_empty() {
                list_op.explicit_items
            } else if !list_op.prepended_items.is_empty() {
                list_op.prepended_items
            } else {
                return None;
            };
            items.first()?.as_str().to_string()
        }
        _ => return None,
    };

    let mat_path = sdf::path(&material_path_str).ok()?;

    // Find shader children of the material
    let shader_children = get_token_vec_field(data, &mat_path, "primChildren")
        .unwrap_or_default();

    // First pass: find the UsdPreviewSurface shader and check if diffuseColor is connected
    let mut diffuse_texture_path: Option<String> = None;

    for shader_name in &shader_children {
        let shader_path_str = format!("{}/{}", mat_path, shader_name);
        let shader_path = match sdf::path(&shader_path_str) {
            Ok(p) => p,
            Err(_) => continue,
        };

        // Check if this is a UsdPreviewSurface
        let info_id = read_token_prop(data, &shader_path, "info:id");
        if info_id.as_deref() == Some("UsdPreviewSurface") {
            // Check if inputs:diffuseColor has a connection
            if let Ok(dc_path) = shader_path.append_property("inputs:diffuseColor") {
                if let Ok(conn_val) = data.get(&dc_path, "connectionPaths") {
                    // Has a connection — extract the target path
                    if let Value::PathListOp(list_op) = conn_val.into_owned() {
                        let items = if !list_op.explicit_items.is_empty() {
                            list_op.explicit_items
                        } else if !list_op.prepended_items.is_empty() {
                            list_op.prepended_items
                        } else {
                            continue;
                        };
                        if let Some(target) = items.first() {
                            // Target is like /Material/DiffuseTexture.outputs:rgb
                            // Extract the prim path (before the dot)
                            let target_str = target.as_str();
                            let prim_part = target_str.split('.').next().unwrap_or(target_str);
                            if let Ok(tex_prim_path) = sdf::path(prim_part) {
                                diffuse_texture_path = read_asset_path(data, &tex_prim_path, "inputs:file");
                            }
                        }
                    }
                }
            }
            // If no connection, diffuseColor is a plain color — no texture
            break;
        }
    }

    if diffuse_texture_path.is_some() {
        return diffuse_texture_path;
    }

    // Fallback: return any texture found in shader children.
    // For visualization, showing any texture (even normal maps) is better than none.
    for shader_name in &shader_children {
        let shader_path_str = format!("{}/{}", mat_path, shader_name);
        let shader_path = match sdf::path(&shader_path_str) {
            Ok(p) => p,
            Err(_) => continue,
        };

        if let Some(tex) = read_asset_path(data, &shader_path, "inputs:file") {
            return Some(tex);
        }
    }

    None
}

/// Read a token/string property (like info:id) from a prim.
fn read_token_prop(data: &mut dyn AbstractData, prim_path: &sdf::Path, prop_name: &str) -> Option<String> {
    let prop_path = prim_path.append_property(prop_name).ok()?;
    let value = data.get(&prop_path, "default").ok()?;
    match value.into_owned() {
        Value::Token(s) | Value::String(s) => Some(s),
        _ => None,
    }
}

/// Read an asset path property value.
fn read_asset_path(data: &mut dyn AbstractData, prim_path: &sdf::Path, prop_name: &str) -> Option<String> {
    let prop_path = prim_path.append_property(prop_name).ok()?;
    let value = data.get(&prop_path, "default").ok()?;
    match value.into_owned() {
        Value::AssetPath(s) => {
            if s.is_empty() { None } else { Some(s) }
        }
        _ => None,
    }
}

/// Read xformOp:transform (matrix4d) from a prim.
/// USD stores matrices in row-major with row-vector convention (translation in last row).
/// We transpose to column-vector convention (translation in last column) for transform_points.
fn read_transform(data: &mut dyn AbstractData, prim_path: &sdf::Path) -> Option<[f64; 16]> {
    let prop_path = prim_path.append_property("xformOp:transform").ok()?;
    if let Ok(value) = data.get(&prop_path, "default") {
        if let Value::Matrix4d(m) = value.into_owned() {
            if m.len() == 16 {
                // Transpose from row-vector to column-vector convention
                return Some([
                    m[0], m[4], m[8],  m[12],
                    m[1], m[5], m[9],  m[13],
                    m[2], m[6], m[10], m[14],
                    m[3], m[7], m[11], m[15],
                ]);
            }
        }
    }
    None
}

/// Read xformOps from xformOpOrder and compose into a 4x4 matrix (column-vector convention).
fn read_xform_ops(data: &mut dyn AbstractData, prim_path: &sdf::Path) -> Option<[f64; 16]> {
    // Read xformOpOrder to get the ordered list of operations
    let op_order = get_token_vec_field(data, prim_path, "xformOpOrder");

    if let Some(ops) = op_order {
        if ops.is_empty() { return None; }
        let identity: [f64; 16] = [1.0,0.0,0.0,0.0, 0.0,1.0,0.0,0.0, 0.0,0.0,1.0,0.0, 0.0,0.0,0.0,1.0];
        let mut result = identity;

        for op_name in &ops {
            let (prop_name, invert) = if let Some(stripped) = op_name.strip_prefix("!invert!") {
                (stripped, true)
            } else {
                (op_name.as_str(), false)
            };

            let op_matrix = read_single_xform_op(data, prim_path, prop_name);
            if let Some(mut m) = op_matrix {
                if invert {
                    if let Some(inv) = invert_4x4(&m) { m = inv; }
                }
                result = multiply_4x4(&result, &m);
            }
        }

        if result != identity { return Some(result); }
        return None;
    }

    // Fallback: no xformOpOrder — try common ops directly
    if let Some(m) = read_transform(data, prim_path) {
        return Some(m);
    }

    let translate = read_vec3d(data, prim_path, "xformOp:translate");
    let scale = read_vec3d(data, prim_path, "xformOp:scale");
    let rotate_xyz = read_vec3d(data, prim_path, "xformOp:rotateXYZ");
    let rotate_x = read_single_rotation(data, prim_path, "xformOp:rotateX");
    let rotate_y = read_single_rotation(data, prim_path, "xformOp:rotateY");
    let rotate_z = read_single_rotation(data, prim_path, "xformOp:rotateZ");

    if translate.is_none() && scale.is_none() && rotate_xyz.is_none()
        && rotate_x.is_none() && rotate_y.is_none() && rotate_z.is_none() {
        return None;
    }

    let tx = translate.unwrap_or([0.0, 0.0, 0.0]);
    let sc = scale.unwrap_or([1.0, 1.0, 1.0]);
    let (rx_deg, ry_deg, rz_deg) = if let Some(r) = rotate_xyz {
        (r[0], r[1], r[2])
    } else {
        (rotate_x.unwrap_or(0.0), rotate_y.unwrap_or(0.0), rotate_z.unwrap_or(0.0))
    };

    Some(make_trs_matrix(tx, [rx_deg, ry_deg, rz_deg], sc))
}

/// Read a single xformOp property and return as a 4x4 column-vector matrix.
fn read_single_xform_op(data: &mut dyn AbstractData, prim_path: &sdf::Path, prop_name: &str) -> Option<[f64; 16]> {
    // Determine op type from the property name (e.g. "xformOp:translate:rotatePivot")
    let op_type = prop_name.strip_prefix("xformOp:").unwrap_or(prop_name);
    let base_type = op_type.split(':').next().unwrap_or(op_type);

    match base_type {
        "translate" => {
            let v = read_vec3d(data, prim_path, prop_name)?;
            Some(make_translate_matrix(v))
        }
        "scale" => {
            let v = read_vec3d(data, prim_path, prop_name)?;
            Some(make_scale_matrix(v))
        }
        "rotateXYZ" => {
            let v = read_vec3d(data, prim_path, prop_name)?;
            Some(make_trs_matrix([0.0; 3], v, [1.0, 1.0, 1.0]))
        }
        "rotateX" => {
            let deg = read_single_rotation(data, prim_path, prop_name)?;
            Some(make_trs_matrix([0.0; 3], [deg, 0.0, 0.0], [1.0, 1.0, 1.0]))
        }
        "rotateY" => {
            let deg = read_single_rotation(data, prim_path, prop_name)?;
            Some(make_trs_matrix([0.0; 3], [0.0, deg, 0.0], [1.0, 1.0, 1.0]))
        }
        "rotateZ" => {
            let deg = read_single_rotation(data, prim_path, prop_name)?;
            Some(make_trs_matrix([0.0; 3], [0.0, 0.0, deg], [1.0, 1.0, 1.0]))
        }
        "rotateZXY" => {
            // ZXY rotation order: apply Z, then X, then Y
            let v = read_vec3d(data, prim_path, prop_name)?;
            let rz = make_trs_matrix([0.0; 3], [0.0, 0.0, v[2]], [1.0; 3]);
            let rx = make_trs_matrix([0.0; 3], [v[0], 0.0, 0.0], [1.0; 3]);
            let ry = make_trs_matrix([0.0; 3], [0.0, v[1], 0.0], [1.0; 3]);
            Some(multiply_4x4(&multiply_4x4(&ry, &rx), &rz))
        }
        "transform" => {
            // Full 4x4 matrix — read and transpose from USD row-vector to column-vector
            read_transform_named(data, prim_path, prop_name)
        }
        _ => None, // Unknown op type — skip
    }
}

/// Read a named matrix4d property and transpose to column-vector convention.
fn read_transform_named(data: &mut dyn AbstractData, prim_path: &sdf::Path, prop_name: &str) -> Option<[f64; 16]> {
    let prop_path = prim_path.append_property(prop_name).ok()?;
    if let Ok(value) = data.get(&prop_path, "default") {
        if let Value::Matrix4d(m) = value.into_owned() {
            if m.len() == 16 {
                return Some([
                    m[0], m[4], m[8],  m[12],
                    m[1], m[5], m[9],  m[13],
                    m[2], m[6], m[10], m[14],
                    m[3], m[7], m[11], m[15],
                ]);
            }
        }
    }
    None
}

/// Build a translation matrix (column-vector convention).
fn make_translate_matrix(t: [f64; 3]) -> [f64; 16] {
    [1.0, 0.0, 0.0, t[0],
     0.0, 1.0, 0.0, t[1],
     0.0, 0.0, 1.0, t[2],
     0.0, 0.0, 0.0, 1.0]
}

/// Build a scale matrix (column-vector convention).
fn make_scale_matrix(s: [f64; 3]) -> [f64; 16] {
    [s[0], 0.0,  0.0,  0.0,
     0.0,  s[1], 0.0,  0.0,
     0.0,  0.0,  s[2], 0.0,
     0.0,  0.0,  0.0,  1.0]
}

/// Build a TRS matrix from translate, rotateXYZ (degrees), scale (column-vector convention).
fn make_trs_matrix(tx: [f64; 3], rot_deg: [f64; 3], sc: [f64; 3]) -> [f64; 16] {
    let rx = rot_deg[0].to_radians();
    let ry = rot_deg[1].to_radians();
    let rz = rot_deg[2].to_radians();

    let (sx, cx) = (rx.sin(), rx.cos());
    let (sy, cy) = (ry.sin(), ry.cos());
    let (sz, cz) = (rz.sin(), rz.cos());

    // R = Rz * Ry * Rx (column-vector convention)
    let r00 = cy * cz;
    let r01 = -(cy * sz);
    let r02 = sy;
    let r10 = cx * sz + sx * sy * cz;
    let r11 = cx * cz - sx * sy * sz;
    let r12 = -(sx * cy);
    let r20 = sx * sz - cx * sy * cz;
    let r21 = sx * cz + cx * sy * sz;
    let r22 = cx * cy;

    [r00 * sc[0], r01 * sc[1], r02 * sc[2], tx[0],
     r10 * sc[0], r11 * sc[1], r12 * sc[2], tx[1],
     r20 * sc[0], r21 * sc[1], r22 * sc[2], tx[2],
     0.0,         0.0,         0.0,          1.0]
}

/// Invert a 4x4 matrix. Returns None if singular.
fn invert_4x4(m: &[f64; 16]) -> Option<[f64; 16]> {
    // For translation-only matrices (common case for pivots), fast path
    if m[0] == 1.0 && m[1] == 0.0 && m[2] == 0.0
        && m[4] == 0.0 && m[5] == 1.0 && m[6] == 0.0
        && m[8] == 0.0 && m[9] == 0.0 && m[10] == 1.0
        && m[12] == 0.0 && m[13] == 0.0 && m[14] == 0.0 && m[15] == 1.0 {
        return Some([1.0, 0.0, 0.0, -m[3],
                     0.0, 1.0, 0.0, -m[7],
                     0.0, 0.0, 1.0, -m[11],
                     0.0, 0.0, 0.0, 1.0]);
    }
    // General 4x4 inverse using cofactor expansion
    let a = m;
    let mut inv = [0.0f64; 16];
    inv[0]  =  a[5]*a[10]*a[15] - a[5]*a[11]*a[14] - a[9]*a[6]*a[15] + a[9]*a[7]*a[14] + a[13]*a[6]*a[11] - a[13]*a[7]*a[10];
    inv[1]  = -a[1]*a[10]*a[15] + a[1]*a[11]*a[14] + a[9]*a[2]*a[15] - a[9]*a[3]*a[14] - a[13]*a[2]*a[11] + a[13]*a[3]*a[10];
    inv[2]  =  a[1]*a[6]*a[15]  - a[1]*a[7]*a[14]  - a[5]*a[2]*a[15] + a[5]*a[3]*a[14] + a[13]*a[2]*a[7]  - a[13]*a[3]*a[6];
    inv[3]  = -a[1]*a[6]*a[11]  + a[1]*a[7]*a[10]  + a[5]*a[2]*a[11] - a[5]*a[3]*a[10] - a[9]*a[2]*a[7]   + a[9]*a[3]*a[6];
    inv[4]  = -a[4]*a[10]*a[15] + a[4]*a[11]*a[14] + a[8]*a[6]*a[15] - a[8]*a[7]*a[14] - a[12]*a[6]*a[11] + a[12]*a[7]*a[10];
    inv[5]  =  a[0]*a[10]*a[15] - a[0]*a[11]*a[14] - a[8]*a[2]*a[15] + a[8]*a[3]*a[14] + a[12]*a[2]*a[11] - a[12]*a[3]*a[10];
    inv[6]  = -a[0]*a[6]*a[15]  + a[0]*a[7]*a[14]  + a[4]*a[2]*a[15] - a[4]*a[3]*a[14] - a[12]*a[2]*a[7]  + a[12]*a[3]*a[6];
    inv[7]  =  a[0]*a[6]*a[11]  - a[0]*a[7]*a[10]  - a[4]*a[2]*a[11] + a[4]*a[3]*a[10] + a[8]*a[2]*a[7]   - a[8]*a[3]*a[6];
    inv[8]  =  a[4]*a[9]*a[15]  - a[4]*a[11]*a[13] - a[8]*a[5]*a[15] + a[8]*a[7]*a[13] + a[12]*a[5]*a[11] - a[12]*a[7]*a[9];
    inv[9]  = -a[0]*a[9]*a[15]  + a[0]*a[11]*a[13] + a[8]*a[1]*a[15] - a[8]*a[3]*a[13] - a[12]*a[1]*a[11] + a[12]*a[3]*a[9];
    inv[10] =  a[0]*a[5]*a[15]  - a[0]*a[7]*a[13]  - a[4]*a[1]*a[15] + a[4]*a[3]*a[13] + a[12]*a[1]*a[7]  - a[12]*a[3]*a[5];
    inv[11] = -a[0]*a[5]*a[11]  + a[0]*a[7]*a[9]   + a[4]*a[1]*a[11] - a[4]*a[3]*a[9]  - a[8]*a[1]*a[7]   + a[8]*a[3]*a[5];
    inv[12] = -a[4]*a[9]*a[14]  + a[4]*a[10]*a[13] + a[8]*a[5]*a[14] - a[8]*a[6]*a[13] - a[12]*a[5]*a[10] + a[12]*a[6]*a[9];
    inv[13] =  a[0]*a[9]*a[14]  - a[0]*a[10]*a[13] - a[8]*a[1]*a[14] + a[8]*a[2]*a[13] + a[12]*a[1]*a[10] - a[12]*a[2]*a[9];
    inv[14] = -a[0]*a[5]*a[14]  + a[0]*a[6]*a[13]  + a[4]*a[1]*a[14] - a[4]*a[2]*a[13] - a[12]*a[1]*a[6]  + a[12]*a[2]*a[5];
    inv[15] =  a[0]*a[5]*a[10]  - a[0]*a[6]*a[9]   - a[4]*a[1]*a[10] + a[4]*a[2]*a[9]  + a[8]*a[1]*a[6]   - a[8]*a[2]*a[5];
    let det = a[0]*inv[0] + a[1]*inv[4] + a[2]*inv[8] + a[3]*inv[12];
    if det.abs() < 1e-12 { return None; }
    let inv_det = 1.0 / det;
    for v in inv.iter_mut() { *v *= inv_det; }
    Some(inv)
}

/// Read a single-axis rotation value (float/double in degrees).
fn read_single_rotation(data: &mut dyn AbstractData, prim_path: &sdf::Path, prop_name: &str) -> Option<f64> {
    let prop_path = prim_path.append_property(prop_name).ok()?;
    let value = data.get(&prop_path, "default").ok()?;
    match value.into_owned() {
        Value::Double(d) => Some(d),
        Value::Float(f) => Some(f as f64),
        Value::Half(h) => Some(h.to_f64()),
        _ => None,
    }
}

/// Read a vec3d/double3 property from xformOps.
fn read_vec3d(data: &mut dyn AbstractData, prim_path: &sdf::Path, prop_name: &str) -> Option<[f64; 3]> {
    let prop_path = prim_path.append_property(prop_name).ok()?;
    let value = data.get(&prop_path, "default").ok()?;
    match value.into_owned() {
        Value::Vec3d(v) if v.len() >= 3 => Some([v[0], v[1], v[2]]),
        Value::Vec3f(v) if v.len() >= 3 => Some([v[0] as f64, v[1] as f64, v[2] as f64]),
        Value::DoubleVec(v) if v.len() >= 3 => Some([v[0], v[1], v[2]]),
        _ => None,
    }
}

/// Read a double/float scalar property.
fn read_double_prop(data: &mut dyn AbstractData, prim_path: &sdf::Path, prop_name: &str) -> Option<f64> {
    let prop_path = prim_path.append_property(prop_name).ok()?;
    let value = data.get(&prop_path, "default").ok()?;
    match value.into_owned() {
        Value::Double(d) => Some(d),
        Value::Float(f) => Some(f as f64),
        _ => None,
    }
}

/// Transform flat [x,y,z,...] points by a 4x4 matrix using column-vector convention (v' = M*v).
/// The matrix is stored as m[row*4+col]. Translation is in m[3], m[7], m[11] or m[12..14].
fn transform_points(points: &[f32], m: &[f64; 16]) -> Vec<f32> {
    let mut out = Vec::with_capacity(points.len());
    for chunk in points.chunks_exact(3) {
        let (x, y, z) = (chunk[0] as f64, chunk[1] as f64, chunk[2] as f64);
        // v' = M * v (standard column-vector convention)
        let x2 = m[0] * x + m[1] * y + m[2]  * z + m[3];
        let y2 = m[4] * x + m[5] * y + m[6]  * z + m[7];
        let z2 = m[8] * x + m[9] * y + m[10] * z + m[11];
        out.push(x2 as f32);
        out.push(y2 as f32);
        out.push(z2 as f32);
    }
    out
}

/// Extract a parametric primitive (Cube, Sphere, Cylinder, Cone, Capsule) as mesh data.
fn extract_parametric(
    data: &mut dyn AbstractData,
    path: &sdf::Path,
    name: &str,
    prim_type: &str,
    transform: Option<&[f64; 16]>,
) -> Option<MeshData> {
    let (points, indices) = match prim_type {
        "Cube" => {
            let size = read_double_prop(data, path, "size").unwrap_or(2.0) as f32;
            generate_cube(size)
        }
        "Sphere" => {
            let radius = read_double_prop(data, path, "radius").unwrap_or(1.0) as f32;
            generate_sphere(radius, 16, 12)
        }
        "Cylinder" => {
            let radius = read_double_prop(data, path, "radius").unwrap_or(1.0) as f32;
            let height = read_double_prop(data, path, "height").unwrap_or(2.0) as f32;
            generate_cylinder(radius, height, 24)
        }
        "Cone" => {
            let radius = read_double_prop(data, path, "radius").unwrap_or(1.0) as f32;
            let height = read_double_prop(data, path, "height").unwrap_or(2.0) as f32;
            generate_cone(radius, height, 24)
        }
        "Capsule" => {
            let radius = read_double_prop(data, path, "radius").unwrap_or(0.5) as f32;
            let height = read_double_prop(data, path, "height").unwrap_or(2.0) as f32;
            generate_capsule(radius, height, 16, 8)
        }
        _ => return None,
    };

    // Read axis attribute (Cylinder, Cone, Capsule default to "Y" in our generators)
    let axis = read_token_prop(data, path, "axis").unwrap_or_default();

    // Rotate points if axis is not Y (our generators create along Y)
    let points = match axis.as_str() {
        "Z" => {
            // Swap Y and Z: (x, y, z) → (x, z, -y)
            let mut rotated = Vec::with_capacity(points.len());
            for chunk in points.chunks_exact(3) {
                rotated.push(chunk[0]);
                rotated.push(-chunk[2]);
                rotated.push(chunk[1]);
            }
            rotated
        }
        "X" => {
            // Swap Y and X: (x, y, z) → (y, x, z)
            let mut rotated = Vec::with_capacity(points.len());
            for chunk in points.chunks_exact(3) {
                rotated.push(chunk[1]);
                rotated.push(-chunk[0]);
                rotated.push(chunk[2]);
            }
            rotated
        }
        _ => points, // "Y" or unset — already correct
    };

    let display_color = read_display_color(data, path);
    let double_sided_path = path.append_property("doubleSided").ok();
    let double_sided = double_sided_path
        .and_then(|p| data.get(&p, "default").ok())
        .and_then(|v| match v.into_owned() {
            Value::Bool(b) => Some(b),
            _ => None,
        })
        .unwrap_or(false);

    let points = if let Some(t) = transform {
        transform_points(&points, t)
    } else {
        points
    };

    Some(MeshData {
        name: name.to_string(),
        points,
        indices,
        display_color,
        double_sided,
        transform: None,
        diffuse_texture: None,
        texcoords: None,
    })
}

/// Generate a cube with the given size (24 vertices for proper per-face normals).
fn generate_cube(size: f32) -> (Vec<f32>, Vec<i32>) {
    let h = size / 2.0;
    let points = vec![
        // front (z-)
        -h,-h,-h, h,-h,-h, h,h,-h, -h,h,-h,
        // back (z+)
        h,-h,h, -h,-h,h, -h,h,h, h,h,h,
        // top (y+)
        -h,h,-h, h,h,-h, h,h,h, -h,h,h,
        // bottom (y-)
        -h,-h,h, h,-h,h, h,-h,-h, -h,-h,-h,
        // right (x+)
        h,-h,-h, h,-h,h, h,h,h, h,h,-h,
        // left (x-)
        -h,-h,h, -h,-h,-h, -h,h,-h, -h,h,h,
    ];
    let mut indices = Vec::new();
    for f in 0..6 {
        let o = (f * 4) as i32;
        indices.extend_from_slice(&[o, o+1, o+2, o, o+2, o+3]);
    }
    (points, indices)
}

/// Generate a UV sphere.
fn generate_sphere(radius: f32, segs: usize, rings: usize) -> (Vec<f32>, Vec<i32>) {
    let mut pts = Vec::new();
    let mut idx = Vec::new();
    for r in 0..=rings {
        let phi = std::f32::consts::PI * r as f32 / rings as f32;
        for s in 0..=segs {
            let theta = 2.0 * std::f32::consts::PI * s as f32 / segs as f32;
            pts.push(radius * phi.sin() * theta.cos());
            pts.push(radius * phi.cos());
            pts.push(radius * phi.sin() * theta.sin());
        }
    }
    for r in 0..rings {
        for s in 0..segs {
            let a = (r * (segs + 1) + s) as i32;
            let b = a + (segs + 1) as i32;
            idx.extend_from_slice(&[a, b, a + 1, b, b + 1, a + 1]);
        }
    }
    (pts, idx)
}

/// Generate a cylinder with caps.
fn generate_cylinder(radius: f32, height: f32, segs: usize) -> (Vec<f32>, Vec<i32>) {
    let mut pts = Vec::new();
    let mut idx = Vec::new();
    let hh = height / 2.0;
    let pi2 = 2.0 * std::f32::consts::PI;

    // Side rings
    for ring in 0..2 {
        let y = if ring == 0 { -hh } else { hh };
        for s in 0..=segs {
            let theta = pi2 * s as f32 / segs as f32;
            pts.push(radius * theta.cos());
            pts.push(y);
            pts.push(radius * theta.sin());
        }
    }
    for s in 0..segs {
        let a = s as i32;
        let b = a + 1;
        let c = a + (segs + 1) as i32;
        let d = c + 1;
        idx.extend_from_slice(&[a, c, b, b, c, d]);
    }

    // Caps
    let bot_c = (pts.len() / 3) as i32;
    pts.extend_from_slice(&[0.0, -hh, 0.0]);
    let top_c = (pts.len() / 3) as i32;
    pts.extend_from_slice(&[0.0, hh, 0.0]);
    for s in 0..segs {
        let t1 = pi2 * s as f32 / segs as f32;
        let t2 = pi2 * (s + 1) as f32 / segs as f32;
        let bi1 = (pts.len() / 3) as i32;
        pts.extend_from_slice(&[radius * t1.cos(), -hh, radius * t1.sin()]);
        let bi2 = (pts.len() / 3) as i32;
        pts.extend_from_slice(&[radius * t2.cos(), -hh, radius * t2.sin()]);
        idx.extend_from_slice(&[bot_c, bi2, bi1]);
        let ti1 = (pts.len() / 3) as i32;
        pts.extend_from_slice(&[radius * t1.cos(), hh, radius * t1.sin()]);
        let ti2 = (pts.len() / 3) as i32;
        pts.extend_from_slice(&[radius * t2.cos(), hh, radius * t2.sin()]);
        idx.extend_from_slice(&[top_c, ti1, ti2]);
    }
    (pts, idx)
}

/// Generate a cone with base cap.
fn generate_cone(radius: f32, height: f32, segs: usize) -> (Vec<f32>, Vec<i32>) {
    let mut pts = Vec::new();
    let mut idx = Vec::new();
    let hh = height / 2.0;
    let pi2 = 2.0 * std::f32::consts::PI;

    // Apex
    pts.extend_from_slice(&[0.0, hh, 0.0]);
    // Base ring
    for s in 0..=segs {
        let theta = pi2 * s as f32 / segs as f32;
        pts.push(radius * theta.cos());
        pts.push(-hh);
        pts.push(radius * theta.sin());
    }
    for s in 0..segs {
        idx.extend_from_slice(&[0, (s + 1) as i32, (s + 2) as i32]);
    }

    // Base cap
    let base_c = (pts.len() / 3) as i32;
    pts.extend_from_slice(&[0.0, -hh, 0.0]);
    for s in 0..segs {
        let t1 = pi2 * s as f32 / segs as f32;
        let t2 = pi2 * (s + 1) as f32 / segs as f32;
        let bi1 = (pts.len() / 3) as i32;
        pts.extend_from_slice(&[radius * t1.cos(), -hh, radius * t1.sin()]);
        let bi2 = (pts.len() / 3) as i32;
        pts.extend_from_slice(&[radius * t2.cos(), -hh, radius * t2.sin()]);
        idx.extend_from_slice(&[base_c, bi2, bi1]);
    }
    (pts, idx)
}

/// Generate a capsule (cylinder + hemisphere caps).
fn generate_capsule(radius: f32, height: f32, segs: usize, rings: usize) -> (Vec<f32>, Vec<i32>) {
    let mut pts = Vec::new();
    let mut idx = Vec::new();
    let hh = height / 2.0;
    let pi = std::f32::consts::PI;
    let pi2 = 2.0 * pi;

    // Top hemisphere: from north pole (r=0) down to equator (r=rings)
    for r in 0..=rings {
        let phi = pi * 0.5 * (1.0 - r as f32 / rings as f32);
        let y = hh + radius * phi.sin();
        let ring_r = radius * phi.cos();
        for s in 0..=segs {
            let theta = pi2 * s as f32 / segs as f32;
            pts.push(ring_r * theta.cos());
            pts.push(y);
            pts.push(ring_r * theta.sin());
        }
    }
    // Bottom hemisphere: from equator (r=0) down to south pole (r=rings)
    let bot_off = pts.len() / 3;
    for r in 0..=rings {
        let phi = pi * 0.5 * r as f32 / rings as f32;
        let y = -hh - radius * phi.sin();
        let ring_r = radius * phi.cos();
        for s in 0..=segs {
            let theta = pi2 * s as f32 / segs as f32;
            pts.push(ring_r * theta.cos());
            pts.push(y);
            pts.push(ring_r * theta.sin());
        }
    }
    // Top hemisphere faces
    for r in 0..rings {
        for s in 0..segs {
            let a = (r * (segs + 1) + s) as i32;
            let b = a + (segs + 1) as i32;
            idx.extend_from_slice(&[a, a + 1, b, b, a + 1, b + 1]);
        }
    }
    // Connect top equator to bottom equator (cylinder section)
    let top_eq = (rings * (segs + 1)) as i32;
    let bot_eq = bot_off as i32;
    for s in 0..segs {
        let a = top_eq + s as i32;
        let b = bot_eq + s as i32;
        idx.extend_from_slice(&[a, a + 1, b, b, a + 1, b + 1]);
    }
    // Bottom hemisphere faces
    for r in 0..rings {
        for s in 0..segs {
            let a = (bot_off + r * (segs + 1) + s) as i32;
            let b = a + (segs + 1) as i32;
            idx.extend_from_slice(&[a, a + 1, b, b, a + 1, b + 1]);
        }
    }
    (pts, idx)
}

/// Triangulate polygon faces using fan triangulation.
fn triangulate(indices: &[i32], counts: &[i32]) -> Vec<i32> {
    let mut tris = Vec::new();
    let mut offset = 0usize;
    for &count in counts {
        let n = count as usize;
        if n >= 3 && offset + n <= indices.len() {
            let v0 = indices[offset];
            for i in 1..n - 1 {
                tris.push(v0);
                tris.push(indices[offset + i]);
                tris.push(indices[offset + i + 1]);
            }
        }
        offset += n;
    }
    tris
}

/// 4x4 matrix multiplication (row-major).
fn multiply_4x4(a: &[f64; 16], b: &[f64; 16]) -> [f64; 16] {
    let mut r = [0.0f64; 16];
    for row in 0..4 {
        for col in 0..4 {
            let mut s = 0.0;
            for k in 0..4 {
                s += a[row * 4 + k] * b[k * 4 + col];
            }
            r[row * 4 + col] = s;
        }
    }
    r
}

// Helper functions for reading common field types

fn get_string_field(data: &mut dyn AbstractData, path: &sdf::Path, field: &str) -> Option<String> {
    let value = data.get(path, field).ok()?;
    match value.into_owned() {
        Value::Token(s) | Value::String(s) => Some(s),
        _ => None,
    }
}

fn get_double_field(data: &mut dyn AbstractData, path: &sdf::Path, field: &str) -> Option<f64> {
    let value = data.get(path, field).ok()?;
    match value.into_owned() {
        Value::Double(d) => Some(d),
        Value::Float(f) => Some(f as f64),
        _ => None,
    }
}

fn get_token_vec_field(data: &mut dyn AbstractData, path: &sdf::Path, field: &str) -> Option<Vec<String>> {
    let value = data.get(path, field).ok()?;
    match value.into_owned() {
        Value::TokenVec(v) => Some(v),
        Value::StringVec(v) => Some(v),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_normals_usda() {
        let data = std::fs::read(
            "vendor/usd-wg-assets/test_assets/NormalsTextureBiasAndScale/NormalsTextureBiasAndScale.usda"
        ).unwrap();
        let result = parse_usd_meshes_inner(&data).unwrap();
        println!("up_axis={} meshes={} error={:?}", result.up_axis, result.meshes.len(), result.error);
        for m in &result.meshes {
            println!("  mesh: {} pts={} idx={} tex={:?}", m.name, m.points.len()/3, m.indices.len(), m.diffuse_texture);
        }
        assert!(result.error.is_none(), "Parse error: {:?}", result.error);
        assert_eq!(result.meshes.len(), 3, "Expected 3 meshes, got {}", result.meshes.len());
    }

    #[test]
    fn test_parse_roughness_usdz() {
        let data = std::fs::read(
            "C:/Users/mad/Documents/GitHub/assets/test_assets/RoughnessTest/RoughnessTest.usdz"
        ).unwrap();
        let cursor = std::io::Cursor::new(&data);
        let mut archive = zip::ZipArchive::new(cursor).unwrap();
        let mut usdc_data = None;
        for i in 0..archive.len() {
            let mut f = archive.by_index(i).unwrap();
            let name = f.name().to_string();
            println!("  USDZ entry: {}", name);
            if name.ends_with(".usdc") || name.ends_with(".usd") {
                let mut buf = Vec::new();
                std::io::Read::read_to_end(&mut f, &mut buf).unwrap();
                usdc_data = Some(buf);
            }
        }
        let usdc_data = usdc_data.expect("No USDC found in USDZ");
        let result = parse_usd_meshes_inner(&usdc_data).unwrap();
        println!("up_axis={} meshes={} error={:?}", result.up_axis, result.meshes.len(), result.error);
        for m in &result.meshes {
            println!("  mesh: {} pts={} idx={} tex={:?} uv={} color={:?} ds={} transform={}",
                m.name, m.points.len()/3, m.indices.len(),
                m.diffuse_texture, m.texcoords.as_ref().map(|t| t.len()/2).unwrap_or(0),
                m.display_color, m.double_sided, m.transform.is_some());
            // Print bbox
            let pts = &m.points;
            if pts.len() >= 3 {
                let (mut minx, mut miny, mut minz) = (f32::MAX, f32::MAX, f32::MAX);
                let (mut maxx, mut maxy, mut maxz) = (f32::MIN, f32::MIN, f32::MIN);
                for c in pts.chunks_exact(3) {
                    minx = minx.min(c[0]); maxx = maxx.max(c[0]);
                    miny = miny.min(c[1]); maxy = maxy.max(c[1]);
                    minz = minz.min(c[2]); maxz = maxz.max(c[2]);
                }
                println!("    bbox X[{:.2}..{:.2}] Y[{:.2}..{:.2}] Z[{:.2}..{:.2}]",
                    minx, maxx, miny, maxy, minz, maxz);
            }
        }
        assert!(result.error.is_none());
        assert_eq!(result.meshes.len(), 6);
    }

    #[test]
    fn test_parse_cesiumman_usdz_texture() {
        let data = std::fs::read(
            "vendor/usd-wg-assets/test_assets/USDZ/CesiumMan/CesiumMan.usdz"
        ).unwrap();
        // USDZ is a zip-like archive; parse_usd_meshes_inner expects raw USDC/USDA data.
        // We need to extract the USDC from the USDZ first.
        // USDZ is an uncompressed zip; the first file is usually the USDC.
        let result = parse_usd_meshes_inner(&data);
        // USDZ won't parse directly — the JS extracts the USDC first.
        // Let's manually find the USDC in the zip.
        let cursor = std::io::Cursor::new(&data);
        let mut archive = zip::ZipArchive::new(cursor).unwrap();
        let mut usdc_data = None;
        for i in 0..archive.len() {
            let mut f = archive.by_index(i).unwrap();
            let name = f.name().to_string();
            if name.ends_with(".usdc") || name.ends_with(".usd") {
                let mut buf = Vec::new();
                std::io::Read::read_to_end(&mut f, &mut buf).unwrap();
                usdc_data = Some(buf);
                println!("Found USDC in USDZ: {}", name);
                break;
            }
        }
        let usdc_data = usdc_data.expect("No USDC found in USDZ");
        let result = parse_usd_meshes_inner(&usdc_data).unwrap();
        println!("up_axis={} meshes={} error={:?}", result.up_axis, result.meshes.len(), result.error);
        for m in &result.meshes {
            println!("  mesh: {} pts={} idx={} tex={:?} uv={}",
                m.name, m.points.len()/3, m.indices.len(),
                m.diffuse_texture, m.texcoords.as_ref().map(|t| t.len()/2).unwrap_or(0));
        }
        // Check texture is found
        let has_texture = result.meshes.iter().any(|m| m.diffuse_texture.is_some());
        assert!(has_texture, "Expected at least one mesh with a diffuse texture");
    }

    #[test]
    fn test_schema_tests() {
        use std::path::Path;

        let base = Path::new("C:/Users/mad/Documents/GitHub/assets/test_assets/schemaTests");
        let mut files: Vec<std::path::PathBuf> = Vec::new();

        fn collect_usda(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.is_dir() {
                        collect_usda(&p, out);
                    } else if p.extension().and_then(|e| e.to_str()) == Some("usda") {
                        out.push(p);
                    }
                }
            }
        }

        collect_usda(base, &mut files);
        files.sort();

        println!("\n========== Schema Tests: {} .usda files ==========\n", files.len());

        let mut total_ok = 0usize;
        let mut total_err = 0usize;

        for path in &files {
            // Short path: strip base prefix
            let short = path.strip_prefix(base).unwrap_or(path);
            print!("--- {} ", short.display());

            let data = match std::fs::read(path) {
                Ok(d) => d,
                Err(e) => {
                    println!("  READ ERROR: {}", e);
                    total_err += 1;
                    continue;
                }
            };

            match parse_usd_meshes_inner(&data) {
                Ok(result) => {
                    println!("--- meshes={} error={:?}", result.meshes.len(), result.error);
                    if result.error.is_some() {
                        total_err += 1;
                    } else {
                        total_ok += 1;
                    }
                    for m in &result.meshes {
                        let pts_count = m.points.len() / 3;
                        let idx_count = m.indices.len();
                        // Compute bbox
                        let pts = &m.points;
                        if pts.len() >= 3 {
                            let (mut minx, mut miny, mut minz) = (f32::MAX, f32::MAX, f32::MAX);
                            let (mut maxx, mut maxy, mut maxz) = (f32::MIN, f32::MIN, f32::MIN);
                            for c in pts.chunks_exact(3) {
                                minx = minx.min(c[0]); maxx = maxx.max(c[0]);
                                miny = miny.min(c[1]); maxy = maxy.max(c[1]);
                                minz = minz.min(c[2]); maxz = maxz.max(c[2]);
                            }
                            println!("  mesh: \"{}\" pts={} idx={} bbox X[{:.4}..{:.4}] Y[{:.4}..{:.4}] Z[{:.4}..{:.4}]",
                                m.name, pts_count, idx_count,
                                minx, maxx, miny, maxy, minz, maxz);
                        } else {
                            println!("  mesh: \"{}\" pts={} idx={} (no points for bbox)", m.name, pts_count, idx_count);
                        }
                    }
                }
                Err(e) => {
                    println!("  PARSE FAILED: {}", e);
                    total_err += 1;
                }
            }
        }

        println!("\n========== Summary: {} OK, {} errors out of {} files ==========\n",
            total_ok, total_err, files.len());
    }
}
