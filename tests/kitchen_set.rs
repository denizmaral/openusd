//! Integration tests for Pixar Kitchen Set .geom.usd files.
//!
//! Tests that every .geom.usd file in the Kitchen Set can be parsed
//! and produces at least one mesh with valid geometry.

use std::path::{Path, PathBuf};

use openusd::sdf::AbstractData;
use openusd::wasm::parse_usd_meshes_inner;

const KITCHEN_SET_ASSETS: &str = "C:/Users/mad/Downloads/USD Tests/Kitchen_set/assets";

fn collect_geom_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_dir() {
                    walk(&p, out);
                } else if p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.ends_with(".geom.usd"))
                    .unwrap_or(false)
                {
                    out.push(p);
                }
            }
        }
    }
    walk(dir, &mut files);
    files.sort();
    files
}

#[test]
fn kitchen_set_all_geom_files_produce_meshes() {
    let base = Path::new(KITCHEN_SET_ASSETS);
    if !base.exists() {
        eprintln!("Skipping: Kitchen Set assets not found at {}", KITCHEN_SET_ASSETS);
        return;
    }

    let files = collect_geom_files(base);
    println!("\n========== Kitchen Set: {} .geom.usd files ==========\n", files.len());
    assert!(files.len() >= 70, "Expected at least 70 geom files, found {}", files.len());

    let mut total_ok = 0usize;
    let mut total_no_mesh = 0usize;
    let mut total_err = 0usize;
    let mut failed_files: Vec<String> = Vec::new();
    let mut no_mesh_files: Vec<String> = Vec::new();

    for path in &files {
        let short = path.strip_prefix(base).unwrap_or(path);
        let short_str = short.display().to_string();

        let data = match std::fs::read(path) {
            Ok(d) => d,
            Err(e) => {
                println!("{} READ ERROR: {}", short_str, e);
                total_err += 1;
                failed_files.push(format!("{}: read error: {}", short_str, e));
                continue;
            }
        };

        match parse_usd_meshes_inner(&data) {
            Ok(result) => {
                if let Some(ref err) = result.error {
                    println!("{} PARSE ERROR: {}", short_str, err);
                    total_err += 1;
                    failed_files.push(format!("{}: {}", short_str, err));
                } else if result.meshes.is_empty() {
                    println!("{} NO MESHES", short_str);
                    total_no_mesh += 1;
                    no_mesh_files.push(short_str.clone());
                } else {
                    let mut total_pts = 0;
                    let mut total_idx = 0;
                    for m in &result.meshes {
                        total_pts += m.points.len() / 3;
                        total_idx += m.indices.len();
                    }
                    println!("{} OK meshes={} verts={} tris={}",
                        short_str, result.meshes.len(), total_pts, total_idx / 3);
                    total_ok += 1;
                }
            }
            Err(e) => {
                println!("{} FAILED: {}", short_str, e);
                total_err += 1;
                failed_files.push(format!("{}: {}", short_str, e));
            }
        }
    }

    println!("\n========== Kitchen Set Summary ==========");
    println!("  OK (with meshes): {}", total_ok);
    println!("  No meshes found:  {}", total_no_mesh);
    println!("  Errors:           {}", total_err);
    println!("  Total files:      {}", files.len());

    if !no_mesh_files.is_empty() {
        println!("\nFiles with 0 meshes:");
        for f in &no_mesh_files {
            println!("  {}", f);
        }
    }
    if !failed_files.is_empty() {
        println!("\nFailed files:");
        for f in &failed_files {
            println!("  {}", f);
        }
    }

    assert_eq!(total_err, 0, "{} files had parse errors", total_err);
    assert_eq!(total_no_mesh, 0, "{} geom files had 0 meshes: {:?}", total_no_mesh, no_mesh_files);
}

#[test]
fn kitchen_set_spatula_bbox() {
    let path = Path::new(KITCHEN_SET_ASSETS).join("Spatula/Spatula.geom.usd");
    if !path.exists() {
        eprintln!("Skipping: Spatula.geom.usd not found");
        return;
    }

    let data = std::fs::read(&path).unwrap();
    let result = parse_usd_meshes_inner(&data).unwrap();

    assert!(result.error.is_none(), "Parse error: {:?}", result.error);
    assert!(!result.meshes.is_empty(), "Spatula should have meshes");

    println!("Spatula: up_axis={} meters_per_unit={}", result.up_axis, result.meters_per_unit);

    // Compute overall bounding box of all meshes
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    for m in &result.meshes {
        println!("  mesh: \"{}\" verts={} tris={}", m.name, m.points.len() / 3, m.indices.len() / 3);
        for chunk in m.points.chunks_exact(3) {
            for i in 0..3 {
                if chunk[i] < min[i] { min[i] = chunk[i]; }
                if chunk[i] > max[i] { max[i] = chunk[i]; }
            }
        }
    }

    let size = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
    println!("  bbox min: [{:.4}, {:.4}, {:.4}]", min[0], min[1], min[2]);
    println!("  bbox max: [{:.4}, {:.4}, {:.4}]", max[0], max[1], max[2]);
    println!("  bbox size: [{:.4}, {:.4}, {:.4}]", size[0], size[1], size[2]);

    // Spatula should be roughly elongated (handle is long).
    // The longest axis should be much larger than the other two.
    let max_dim = size[0].max(size[1]).max(size[2]);
    let min_dim = size[0].min(size[1]).min(size[2]);
    println!("  aspect ratio (max/min): {:.2}", max_dim / min_dim);

    // Spatula.geom.usd contains the blade/head (flat planes), not the handle.
    // Blade is roughly 2-3x wider than thick. Just sanity check we got valid geometry.
    assert!(max_dim > 1.0, "Spatula should have reasonable dimensions, max_dim={:.2}", max_dim);
}

#[test]
fn kitchen_set_spatula_debug_prims() {
    use openusd::sdf;
    use openusd::usdc;
    use std::io::Cursor;

    let path = Path::new(KITCHEN_SET_ASSETS).join("Spatula/Spatula.geom.usd");
    if !path.exists() {
        eprintln!("Skipping: Spatula.geom.usd not found");
        return;
    }

    let data = std::fs::read(&path).unwrap();
    let cursor = Cursor::new(&data);
    let mut crate_data = usdc::CrateData::open(cursor, true).unwrap();

    // Dump prim tree
    fn dump_prim(data: &mut dyn openusd::sdf::AbstractData, path: &sdf::Path, depth: usize) {
        let indent = " ".repeat(depth * 2);
        let type_name = data.get(path, "typeName").ok()
            .and_then(|v| match v.into_owned() {
                sdf::Value::Token(s) | sdf::Value::String(s) => Some(s),
                _ => None,
            })
            .unwrap_or_default();

        let has_points = path.append_property("points").ok()
            .map(|p| data.has_spec(&p))
            .unwrap_or(false);

        let xform_order = data.get(path, "xformOpOrder").ok()
            .and_then(|v| match v.into_owned() {
                sdf::Value::TokenVec(v) => Some(v),
                _ => None,
            });

        println!("{}{} [{}] mesh={} xformOps={:?}",
            indent, path, type_name, has_points, xform_order);

        // Check variant sets
        let vs_children = data.get(path, "variantSetChildren").ok()
            .and_then(|v| match v.into_owned() {
                sdf::Value::TokenVec(v) => Some(v),
                _ => None,
            });
        if let Some(ref vs) = vs_children {
            let vs_sel = data.get(path, "variantSelection").ok()
                .and_then(|v| match v.into_owned() {
                    sdf::Value::VariantSelectionMap(m) => Some(m),
                    _ => None,
                })
                .unwrap_or_default();
            println!("{}  variantSets={:?} selection={:?}", indent, vs, vs_sel);
        }

        let children = data.get(path, "primChildren").ok()
            .and_then(|v| match v.into_owned() {
                sdf::Value::TokenVec(v) => Some(v),
                _ => None,
            })
            .unwrap_or_default();

        for child_name in &children {
            let child_path_str = format!("{}/{}", path, child_name);
            if let Ok(child_path) = sdf::path(&child_path_str) {
                dump_prim(data, &child_path, depth + 1);
            }
        }
    }

    let root = sdf::Path::abs_root();
    let root_children = crate_data.get(&root, "primChildren").ok()
        .and_then(|v| match v.into_owned() {
            sdf::Value::TokenVec(v) => Some(v),
            _ => None,
        })
        .unwrap_or_default();

    println!("\n=== Spatula.geom.usd prim tree ===");
    for name in &root_children {
        let path = sdf::path(&format!("/{}", name)).unwrap();
        dump_prim(&mut crate_data, &path, 0);
    }

    // Read xformOpOrder and values from property path for key prims
    for prim_str in &["/Spatula/Geom", "/Spatula/Geom/pPlane319", "/Spatula/Geom/pPlane320"] {
        let prim_path = sdf::path(prim_str).unwrap();
        let prop_path = prim_path.append_property("xformOpOrder").unwrap();
        if crate_data.has_spec(&prop_path) {
            let val = crate_data.get(&prop_path, "default").unwrap();
            println!("  {} xformOpOrder = {:?}", prim_str, val.into_owned());
        }
        for op in &["xformOp:translate", "xformOp:translate:pivot", "xformOp:rotateXYZ"] {
            let op_path = prim_path.append_property(op).unwrap();
            if crate_data.has_spec(&op_path) {
                let val = crate_data.get(&op_path, "default").unwrap();
                println!("    {} = {:?}", op, val.into_owned());
            }
        }
    }
}

#[test]
fn kitchen_set_spatula_all_specs() {
    use openusd::sdf;
    use openusd::usdc;
    use std::io::Cursor;

    let path = Path::new(KITCHEN_SET_ASSETS).join("Spatula/Spatula.geom.usd");
    if !path.exists() {
        eprintln!("Skipping: Spatula.geom.usd not found");
        return;
    }

    let data = std::fs::read(&path).unwrap();
    let cursor = Cursor::new(&data);
    let crate_data = usdc::CrateData::open(cursor, true).unwrap();

    let mut paths = crate_data.all_paths();
    paths.sort_by(|a, b| a.to_string().cmp(&b.to_string()));

    println!("\n=== ALL specs in Spatula.geom.usd ({} total) ===", paths.len());
    for p in &paths {
        let spec_type = crate_data.spec_type(p).unwrap_or(sdf::SpecType::Unknown);
        let fields = crate_data.list(p).unwrap_or_default();
        println!("  {} [{:?}] fields={:?}", p, spec_type, fields);
    }
}

#[test]
fn kitchen_set_spatula_triangle_quality() {
    use openusd::sdf;
    use openusd::usdc;
    use std::io::Cursor;

    let path = Path::new(KITCHEN_SET_ASSETS).join("Spatula/Spatula.geom.usd");
    if !path.exists() {
        eprintln!("Skipping: Spatula.geom.usd not found");
        return;
    }

    // Read raw geometry data from USDC directly
    {
        let raw = std::fs::read(&path).unwrap();
        let cursor = Cursor::new(&raw);
        let mut crate_data = usdc::CrateData::open(cursor, true).unwrap();
        for prim_name in &["pPlane319", "pPlane320"] {
            let prim_path = sdf::path(&format!("/Spatula/Geom/{}", prim_name)).unwrap();
            let pts_path = prim_path.append_property("points").unwrap();
            let val = crate_data.get(&pts_path, "default").unwrap();
            if let sdf::Value::Vec3f(ref v) = *val {
                println!("RAW {} verts={}", prim_name, v.len() / 3);
            }
            // FaceVertexCounts
            let fvc_path = prim_path.append_property("faceVertexCounts").unwrap();
            let fvc = crate_data.get(&fvc_path, "default").unwrap();
            if let sdf::Value::IntVec(ref v) = *fvc {
                let mut counts: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
                for &c in v { *counts.entry(c).or_default() += 1; }
                println!("  faceVertexCounts: {} faces, distribution: {:?}", v.len(), counts);
            }
            // FaceVertexIndices count
            let fvi_path = prim_path.append_property("faceVertexIndices").unwrap();
            let fvi = crate_data.get(&fvi_path, "default").unwrap();
            if let sdf::Value::IntVec(ref v) = *fvi {
                println!("  faceVertexIndices: {} entries, range [{}, {}]",
                    v.len(), v.iter().min().unwrap(), v.iter().max().unwrap());
            }
            // UV
            let st_path = prim_path.append_property("primvars:st").unwrap();
            if let Ok(val) = crate_data.get(&st_path, "default") {
                if let sdf::Value::Vec2f(ref v) = *val { println!("  primvars:st: {} UVs", v.len() / 2); }
            }
            let sti_path = prim_path.append_property("primvars:st:indices").unwrap();
            if let Ok(val) = crate_data.get(&sti_path, "default") {
                if let sdf::Value::IntVec(ref v) = *val {
                    println!("  primvars:st:indices: {} entries, range [{}, {}]",
                        v.len(), v.iter().min().unwrap(), v.iter().max().unwrap());
                }
            }
            if let Ok(val) = crate_data.get(&st_path, "interpolation") {
                println!("  primvars:st interpolation: {:?}", val.into_owned());
            }
        }
    }

    let data = std::fs::read(&path).unwrap();
    let result = parse_usd_meshes_inner(&data).unwrap();
    assert!(result.error.is_none(), "Parse error: {:?}", result.error);

    for (mi, m) in result.meshes.iter().enumerate() {
        let pts = &m.points;
        let idx = &m.indices;
        let vert_count = pts.len() / 3;

        // Print first 5 vertices for debugging
        println!("mesh '{}' first 5 verts:", m.name);
        for vi in 0..5.min(vert_count) {
            println!("  v{}: ({:.4}, {:.4}, {:.4})", vi, pts[vi*3], pts[vi*3+1], pts[vi*3+2]);
        }
        println!("mesh '{}' first 3 tris:", m.name);
        for ti in 0..3.min(idx.len()/3) {
            println!("  tri{}: [{}, {}, {}]", ti, idx[ti*3], idx[ti*3+1], idx[ti*3+2]);
        }

        // Compute bbox diagonal for reference
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];
        for chunk in pts.chunks_exact(3) {
            for i in 0..3 {
                if chunk[i] < min[i] { min[i] = chunk[i]; }
                if chunk[i] > max[i] { max[i] = chunk[i]; }
            }
        }
        let diag = ((max[0]-min[0]).powi(2) + (max[1]-min[1]).powi(2) + (max[2]-min[2]).powi(2)).sqrt();
        println!("mesh '{}': verts={} tris={} bbox_diag={:.4}", m.name, vert_count, idx.len()/3, diag);

        // Check each triangle: no edge should be longer than bbox diagonal
        let mut bad_tris = 0;
        for tri in idx.chunks_exact(3) {
            let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            assert!(a < vert_count && b < vert_count && c < vert_count,
                "OOB index in mesh '{}': {} {} {} (vert_count={})", m.name, a, b, c, vert_count);

            let edge_len = |i: usize, j: usize| -> f32 {
                let dx = pts[i*3] - pts[j*3];
                let dy = pts[i*3+1] - pts[j*3+1];
                let dz = pts[i*3+2] - pts[j*3+2];
                (dx*dx + dy*dy + dz*dz).sqrt()
            };
            let e1 = edge_len(a, b);
            let e2 = edge_len(b, c);
            let e3 = edge_len(c, a);
            let max_edge = e1.max(e2).max(e3);
            if max_edge > diag * 0.5 {
                bad_tris += 1;
                if bad_tris <= 5 {
                    println!("  BAD tri [{},{},{}] max_edge={:.4} (>{:.4})",
                        a, b, c, max_edge, diag * 0.5);
                }
            }
        }
        println!("  bad tris (edge > 50% diag): {} / {}", bad_tris, idx.len()/3);
        assert_eq!(bad_tris, 0, "Mesh '{}' has {} degenerate triangles", m.name, bad_tris);
    }
}

#[test]
fn kitchen_set_pan_has_geometry() {
    let path = Path::new(KITCHEN_SET_ASSETS).join("Pan/Pan.geom.usd");
    if !path.exists() {
        eprintln!("Skipping: Pan.geom.usd not found");
        return;
    }

    let data = std::fs::read(&path).unwrap();
    let result = parse_usd_meshes_inner(&data).unwrap();

    println!("Pan.geom.usd: meshes={} error={:?}", result.meshes.len(), result.error);
    for m in &result.meshes {
        println!("  mesh: \"{}\" pts={} idx={} double_sided={}",
            m.name, m.points.len() / 3, m.indices.len(), m.double_sided);
    }

    assert!(result.error.is_none(), "Parse error: {:?}", result.error);
    assert!(!result.meshes.is_empty(), "Pan.geom.usd should have at least one mesh");

    // Validate geometry
    for m in &result.meshes {
        assert!(m.points.len() >= 9, "Mesh '{}' should have at least 3 vertices", m.name);
        assert!(m.indices.len() >= 3, "Mesh '{}' should have at least 1 triangle", m.name);
        assert_eq!(m.indices.len() % 3, 0, "Mesh '{}' indices should be multiple of 3", m.name);
        // All indices should be in range
        let vert_count = (m.points.len() / 3) as i32;
        for &idx in &m.indices {
            assert!(idx >= 0 && idx < vert_count,
                "Mesh '{}' has out-of-range index {} (vert_count={})", m.name, idx, vert_count);
        }
    }
}

#[test]
fn kitchen_set_mesh_data_integrity() {
    let base = Path::new(KITCHEN_SET_ASSETS);
    if !base.exists() {
        eprintln!("Skipping: Kitchen Set assets not found");
        return;
    }

    let files = collect_geom_files(base);
    let mut total_verts = 0usize;
    let mut total_tris = 0usize;
    let mut total_meshes = 0usize;

    for path in &files {
        let data = std::fs::read(path).unwrap();
        let result = parse_usd_meshes_inner(&data).unwrap();
        if result.error.is_some() { continue; }

        for m in &result.meshes {
            let vert_count = m.points.len() / 3;
            let tri_count = m.indices.len() / 3;

            // Points must be multiple of 3
            assert_eq!(m.points.len() % 3, 0,
                "Mesh '{}' points length {} not multiple of 3", m.name, m.points.len());

            // Indices must be multiple of 3 (triangulated)
            assert_eq!(m.indices.len() % 3, 0,
                "Mesh '{}' indices length {} not multiple of 3", m.name, m.indices.len());

            // No NaN/Inf in points
            for (i, &v) in m.points.iter().enumerate() {
                assert!(v.is_finite(),
                    "Mesh '{}' has non-finite point value at index {}: {}", m.name, i, v);
            }

            // Indices in range
            let vc = vert_count as i32;
            for &idx in &m.indices {
                assert!(idx >= 0 && idx < vc,
                    "Mesh '{}' index {} out of range [0..{})", m.name, idx, vc);
            }

            // If texcoords exist, they must match vertex count
            if let Some(ref uvs) = m.texcoords {
                assert_eq!(uvs.len() / 2, vert_count,
                    "Mesh '{}' UV count {} != vertex count {}",
                    m.name, uvs.len() / 2, vert_count);
            }

            total_verts += vert_count;
            total_tris += tri_count;
            total_meshes += 1;
        }
    }

    println!("\nKitchen Set data integrity: {} meshes, {} vertices, {} triangles",
        total_meshes, total_verts, total_tris);
    assert!(total_meshes > 0, "Expected at least some meshes");
}
