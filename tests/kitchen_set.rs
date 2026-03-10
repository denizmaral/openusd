//! Integration tests for Pixar Kitchen Set .geom.usd files.
//!
//! Tests that every .geom.usd file in the Kitchen Set can be parsed
//! and produces at least one mesh with valid geometry.

use std::path::{Path, PathBuf};

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
