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
        // Try as USDA text
        let text = std::str::from_utf8(data).context("Invalid UTF-8 in USDA file")?;
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

    // Triangulate if needed
    let indices = match face_vertex_counts {
        Some(counts) if counts.iter().any(|&c| c != 3) => {
            triangulate(&face_vertex_indices, &counts)
        }
        _ => face_vertex_indices,
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

    // Read UV coordinates (primvars:st)
    let texcoords = read_texcoords(data, path);

    // Check if this is a skinned mesh (has skel:skeleton binding)
    let is_skinned = path.append_property("skel:skeleton").ok()
        .map(|p| data.has_spec(&p))
        .unwrap_or(false)
        || path.append_property("skel:joints").ok()
            .map(|p| data.has_spec(&p))
            .unwrap_or(false);

    // Bake world transform into vertex positions.
    // Skip transform for skinned meshes — their parent transform is for the
    // skeleton system, not static display. Without skeleton/animation support,
    // the bind-pose vertices are already in a viewable orientation.
    let points = if !is_skinned {
        if let Some(t) = transform {
            transform_points(&points, t)
        } else {
            points
        }
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
        diffuse_texture,
        texcoords,
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
    // Try primvars:st first (most common), then primvars:st0, primvars:UVMap
    for prop_name in &["primvars:st", "primvars:st0", "primvars:UVMap"] {
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
    None
}

/// Follow material:binding → material → shader → diffuse texture asset path.
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

    for shader_name in &shader_children {
        let shader_path_str = format!("{}/{}", mat_path, shader_name);
        let shader_path = sdf::path(&shader_path_str).ok()?;

        // Check for inputs:diffuseColor connection to a texture
        // Try reading inputs:file on the shader itself (UsdUVTexture)
        if let Some(tex) = read_asset_path(data, &shader_path, "inputs:file") {
            return Some(tex);
        }

        // Check children of shader for UsdUVTexture nodes
        let tex_children = get_token_vec_field(data, &shader_path, "primChildren")
            .unwrap_or_default();
        for tex_name in &tex_children {
            let tex_path_str = format!("{}/{}", shader_path, tex_name);
            if let Ok(tex_path) = sdf::path(&tex_path_str) {
                if let Some(tex) = read_asset_path(data, &tex_path, "inputs:file") {
                    return Some(tex);
                }
            }
        }
    }

    // Also check nested structure: material may have deeper hierarchy
    // Try a common pattern: Material/Shader/diffuseColor_texture
    None
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
fn read_transform(data: &mut dyn AbstractData, prim_path: &sdf::Path) -> Option<[f64; 16]> {
    let prop_path = prim_path.append_property("xformOp:transform").ok()?;
    if let Ok(value) = data.get(&prop_path, "default") {
        if let Value::Matrix4d(m) = value.into_owned() {
            if m.len() == 16 {
                let mut arr = [0.0f64; 16];
                arr.copy_from_slice(&m);
                return Some(arr);
            }
        }
    }
    None
}

/// Read xformOps (translate, rotateXYZ, scale) and compose into a 4x4 matrix.
/// Returns row-major matrix with translation in last row (USD convention).
fn read_xform_ops(data: &mut dyn AbstractData, prim_path: &sdf::Path) -> Option<[f64; 16]> {
    // Try xformOp:transform first (full matrix)
    if let Some(m) = read_transform(data, prim_path) {
        return Some(m);
    }

    // Read individual ops
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

    // Determine rotation angles in degrees
    let (rx_deg, ry_deg, rz_deg) = if let Some(r) = rotate_xyz {
        (r[0], r[1], r[2])
    } else {
        (rotate_x.unwrap_or(0.0), rotate_y.unwrap_or(0.0), rotate_z.unwrap_or(0.0))
    };

    let rx = rx_deg.to_radians();
    let ry = ry_deg.to_radians();
    let rz = rz_deg.to_radians();

    // Build rotation matrix (column-vector convention: v' = M*v)
    let (sx, cx) = (rx.sin(), rx.cos());
    let (sy, cy) = (ry.sin(), ry.cos());
    let (sz, cz) = (rz.sin(), rz.cos());

    // Combined rotation R = Rz * Ry * Rx (column-vector convention)
    // This is the transpose of the row-vector Rx*Ry*Rz
    let r00 = cy * cz;
    let r01 = -(cy * sz);
    let r02 = sy;
    let r10 = cx * sz + sx * sy * cz;
    let r11 = cx * cz - sx * sy * sz;
    let r12 = -(sx * cy);
    let r20 = sx * sz - cx * sy * cz;
    let r21 = sx * cz + cx * sy * sz;
    let r22 = cx * cy;

    // Build matrix: M = T * R * S, stored as m[row*4+col]
    // Column-vector convention: translation in last column (indices 3, 7, 11)
    Some([
        r00 * sc[0], r01 * sc[1], r02 * sc[2], tx[0],
        r10 * sc[0], r11 * sc[1], r12 * sc[2], tx[1],
        r20 * sc[0], r21 * sc[1], r22 * sc[2], tx[2],
        0.0,         0.0,         0.0,          1.0,
    ])
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

    // Top hemisphere
    for r in 0..=rings {
        let phi = pi * 0.5 * r as f32 / rings as f32;
        for s in 0..=segs {
            let theta = pi2 * s as f32 / segs as f32;
            pts.push(radius * phi.cos() * theta.cos());
            pts.push(hh + radius * phi.sin());
            pts.push(radius * phi.cos() * theta.sin());
        }
    }
    // Bottom hemisphere
    let bot_off = pts.len() / 3;
    for r in 0..=rings {
        let phi = pi * 0.5 + pi * 0.5 * r as f32 / rings as f32;
        for s in 0..=segs {
            let theta = pi2 * s as f32 / segs as f32;
            pts.push(radius * phi.cos() * theta.cos());
            pts.push(-hh + radius * phi.sin());
            pts.push(radius * phi.cos() * theta.sin());
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
    // Bottom hemisphere faces
    for r in 0..rings {
        for s in 0..segs {
            let a = (bot_off + r * (segs + 1) + s) as i32;
            let b = a + (segs + 1) as i32;
            idx.extend_from_slice(&[a, a + 1, b, b, a + 1, b + 1]);
        }
    }
    // Connect top to bottom (cylinder section)
    let top_ring = (rings * (segs + 1)) as i32;
    let bot_ring = bot_off as i32;
    for s in 0..segs {
        let a = top_ring + s as i32;
        let b = bot_ring + s as i32;
        idx.extend_from_slice(&[a, a + 1, b, b, a + 1, b + 1]);
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
