//! USDC export of the viewer's scene.
//!
//! Meshes are grouped under `/Root/<group>`; materials are UsdPreviewSurface networks under
//! `/Materials`, with one UsdUVTexture per texture channel reading `st` through a primvar reader.
//! The layer is built as `sdf::Data` and encoded by upstream's crate writer.

use std::collections::HashMap;
use std::io::Cursor;

use anyhow::{anyhow, Result};
use openusd::sdf::{self, AbstractData, Path, SpecType, Specifier, Variability};
use openusd::vt::Value;
use openusd::{gf, tf, usdc};
use serde::Deserialize;

/// The scene the viewer exports, as JSON.
#[derive(Deserialize)]
pub struct ExportScene {
    pub meshes: Vec<ExportMesh>,
    #[serde(default)]
    pub materials: Vec<ExportMaterial>,
    /// Frames per second of the time samples.
    #[serde(default)]
    pub time_codes_per_second: Option<f64>,
    /// Start of the animation range.
    #[serde(default)]
    pub start_time_code: Option<f64>,
    /// End of the animation range.
    #[serde(default)]
    pub end_time_code: Option<f64>,
}

/// A triangle mesh.
#[derive(Deserialize)]
pub struct ExportMesh {
    pub name: String,
    #[serde(default = "default_group")]
    pub group: String,
    /// Flat `[x, y, z, ...]`.
    pub points: Vec<f32>,
    /// Three indices per triangle.
    pub indices: Vec<i32>,
    /// Flat `[u, v, ...]`, one pair per point.
    #[serde(default)]
    pub uvs: Option<Vec<f32>>,
    #[serde(default)]
    pub display_color: Option<[f32; 3]>,
    #[serde(default)]
    pub material_id: Option<usize>,
    /// Custom string attributes, such as IFC properties or TopoJSON fields.
    #[serde(default)]
    pub attributes: Option<HashMap<String, String>>,
    /// Visibility samples: `(time code, "invisible" | "inherited")`.
    #[serde(default)]
    pub visibility_times: Option<Vec<(f64, String)>>,
}

fn default_group() -> String {
    "Elements".to_string()
}

/// A UsdPreviewSurface material.
#[derive(Deserialize)]
pub struct ExportMaterial {
    pub id: usize,
    #[serde(default)]
    pub diffuse_color: Option<[f32; 3]>,
    #[serde(default)]
    pub metallic: Option<f32>,
    #[serde(default)]
    pub roughness: Option<f32>,
    /// Texture file per channel: `diffuse`, `normal`, `mr`, `emissive` or `ao`.
    #[serde(default)]
    pub textures: HashMap<String, String>,
}

/// Texture channels in the order their shader prims are written.
const TEXTURE_PRIMS: [(&str, &str); 5] = [
    ("diffuse", "DiffuseTexture"),
    ("normal", "NormalTexture"),
    ("mr", "MRTexture"),
    ("emissive", "EmissiveTexture"),
    ("ao", "AOTexture"),
];

/// The PreviewSurface inputs only a texture drives: channel, input name and type.
const TEXTURE_ONLY_INPUTS: [(&str, &str, &str); 3] = [
    ("emissive", "inputs:emissiveColor", "color3f"),
    ("normal", "inputs:normal", "normal3f"),
    ("ao", "inputs:occlusion", "float"),
];

/// Builds a USDC file from `scene`.
pub fn build_usdc_scene(scene: &ExportScene) -> Result<Vec<u8>> {
    let mut layer = LayerBuilder::default();
    layer.pseudo_root(scene);
    if !scene.materials.is_empty() {
        let names: Vec<String> = scene.materials.iter().map(|m| format!("Material_{}", m.id)).collect();
        layer.prim(&sdf::path("/Materials")?, None, &names, &[]);
        for material in &scene.materials {
            layer.material(material)?;
        }
    }
    layer.meshes(scene)?;

    let mut out = Cursor::new(Vec::new());
    usdc::CrateWriter::write(&layer.data, &mut out).map_err(|e| anyhow!("{e}"))?;
    let mut bytes = out.into_inner();
    stamp_minimum_version(&mut bytes);
    Ok(bytes)
}

/// Offset of the crate version (major, minor, patch) in the file header, after the `PXR-USDC` magic.
const VERSION_OFFSET: usize = 8;

/// Marks the file as crate version 0.8.0, the oldest that holds everything this exporter writes.
///
/// Upstream's writer stamps every file 0.12.0, while C++ `CrateFile` writes 0.8.0 and upgrades only for
/// a time code (0.9.0), a path expression (0.10.0), relocates (0.11.0) or a spline (0.12.0). This
/// exporter writes none of those, and the encodings of everything else are the same from 0.8.0 on, so
/// the lower stamp only lets older USD releases open the file. Drop this once the upstream writer
/// stamps the version its content needs.
fn stamp_minimum_version(bytes: &mut [u8]) {
    bytes[VERSION_OFFSET..VERSION_OFFSET + 3].copy_from_slice(&[0, 8, 0]);
}

#[derive(Default)]
struct LayerBuilder {
    data: sdf::Data,
}

impl LayerBuilder {
    fn pseudo_root(&mut self, scene: &ExportScene) {
        let root = Path::abs_root();
        self.data.create_spec(root.clone(), SpecType::PseudoRoot);
        self.set(&root, "defaultPrim", token("Root"));
        self.set(&root, "metersPerUnit", Value::Double(1.0));
        self.set(&root, "upAxis", token("Y"));
        if let Some(fps) = scene.time_codes_per_second {
            self.set(&root, "timeCodesPerSecond", Value::Double(fps));
            self.set(&root, "framesPerSecond", Value::Double(fps));
        }
        if let Some(start) = scene.start_time_code {
            self.set(&root, "startTimeCode", Value::Double(start));
        }
        if let Some(end) = scene.end_time_code {
            self.set(&root, "endTimeCode", Value::Double(end));
        }
        let mut children = Vec::new();
        if !scene.materials.is_empty() {
            children.push("Materials");
        }
        children.push("Root");
        self.set(&root, "primChildren", tokens(&children));
    }

    fn material(&mut self, material: &ExportMaterial) -> Result<()> {
        let path = sdf::path(format!("/Materials/Material_{}", material.id))?;
        let shader = path.append_path("PBRShader")?;
        let reader = path.append_path("Primvar_st")?;
        let textures: Vec<(&str, &str)> = TEXTURE_PRIMS
            .into_iter()
            .filter(|(channel, _)| material.textures.contains_key(*channel))
            .collect();

        let mut children = vec!["PBRShader".to_string()];
        children.extend(textures.iter().map(|(_, prim)| prim.to_string()));
        if !textures.is_empty() {
            children.push("Primvar_st".to_string());
        }
        self.prim(&path, Some("Material"), &children, &["outputs:surface"]);
        self.attribute(
            &path.append_property("outputs:surface")?,
            "token",
            [
                uniform(),
                (
                    "connectionPaths",
                    connection(shader.append_property("outputs:surface")?),
                ),
            ],
        );

        // The PreviewSurface, its inputs connected to the texture of their channel when there is one.
        let rgb_of = |channel: &str| -> Result<Option<Path>> {
            TEXTURE_PRIMS
                .iter()
                .find(|(c, _)| *c == channel && material.textures.contains_key(*c))
                .map(|(_, prim)| path.append_path(*prim)?.append_property("outputs:rgb"))
                .transpose()
                .map_err(Into::into)
        };
        let mut properties = vec![
            "info:id",
            "inputs:diffuseColor",
            "inputs:metallic",
            "inputs:roughness",
            "outputs:surface",
        ];
        properties.extend(
            TEXTURE_ONLY_INPUTS
                .iter()
                .filter(|(channel, ..)| material.textures.contains_key(*channel))
                .map(|(_, input, _)| *input),
        );
        self.prim(&shader, Some("Shader"), &[], &properties);
        self.attribute(
            &shader.append_property("info:id")?,
            "token",
            [uniform(), ("default", token("UsdPreviewSurface"))],
        );

        let diffuse = match (rgb_of("diffuse")?, material.diffuse_color) {
            (Some(rgb), _) => vec![("connectionPaths", connection(rgb))],
            (None, Some(color)) => vec![("default", Value::Vec3f(gf::Vec3f::from(color)))],
            (None, None) => vec![],
        };
        self.attribute(&shader.append_property("inputs:diffuseColor")?, "color3f", diffuse);
        let mr = rgb_of("mr")?;
        for (input, fallback) in [
            ("inputs:metallic", material.metallic.unwrap_or(0.1)),
            ("inputs:roughness", material.roughness.unwrap_or(0.8)),
        ] {
            let field = match &mr {
                Some(rgb) => ("connectionPaths", connection(rgb.clone())),
                None => ("default", Value::Float(fallback)),
            };
            self.attribute(&shader.append_property(input)?, "float", [field]);
        }
        for (channel, input, type_name) in TEXTURE_ONLY_INPUTS {
            if let Some(rgb) = rgb_of(channel)? {
                self.attribute(
                    &shader.append_property(input)?,
                    type_name,
                    [("connectionPaths", connection(rgb))],
                );
            }
        }
        self.attribute(&shader.append_property("outputs:surface")?, "token", []);

        for (channel, prim) in &textures {
            let texture = path.append_path(*prim)?;
            let properties = [
                "info:id",
                "inputs:file",
                "inputs:st",
                "inputs:wrapS",
                "inputs:wrapT",
                "outputs:rgb",
            ];
            self.prim(&texture, Some("Shader"), &[], &properties);
            self.attribute(
                &texture.append_property("info:id")?,
                "token",
                [uniform(), ("default", token("UsdUVTexture"))],
            );
            let file = sdf::AssetPath::new(material.textures[*channel].as_str());
            self.attribute(
                &texture.append_property("inputs:file")?,
                "asset",
                [("default", Value::AssetPath(file))],
            );
            for wrap in ["inputs:wrapS", "inputs:wrapT"] {
                self.attribute(&texture.append_property(wrap)?, "token", [("default", token("repeat"))]);
            }
            let st = connection(reader.append_property("outputs:result")?);
            self.attribute(
                &texture.append_property("inputs:st")?,
                "float2",
                [("connectionPaths", st)],
            );
            self.attribute(&texture.append_property("outputs:rgb")?, "float3", []);
        }

        if !textures.is_empty() {
            self.prim(
                &reader,
                Some("Shader"),
                &[],
                &["info:id", "inputs:varname", "outputs:result"],
            );
            let id = token("UsdPrimvarReader_float2");
            self.attribute(
                &reader.append_property("info:id")?,
                "token",
                [uniform(), ("default", id)],
            );
            let varname = Value::String("st".to_string());
            self.attribute(
                &reader.append_property("inputs:varname")?,
                "string",
                [("default", varname)],
            );
            self.attribute(&reader.append_property("outputs:result")?, "float2", []);
        }
        Ok(())
    }

    fn meshes(&mut self, scene: &ExportScene) -> Result<()> {
        // Groups in the order their first mesh appears, each listing its meshes in input order.
        let mut groups: Vec<(String, Vec<String>)> = Vec::new();
        for (i, mesh) in scene.meshes.iter().enumerate() {
            let group = usd_safe_name(&mesh.group);
            let name = format!("{}_{}", usd_safe_name(&mesh.name), i);
            match groups.iter_mut().find(|(g, _)| *g == group) {
                Some((_, names)) => names.push(name),
                None => groups.push((group, vec![name])),
            }
        }
        let group_names: Vec<String> = groups.iter().map(|(g, _)| g.clone()).collect();
        self.prim(&sdf::path("/Root")?, Some("Xform"), &group_names, &[]);
        for (group, names) in &groups {
            self.prim(&sdf::path(format!("/Root/{group}"))?, Some("Xform"), names, &[]);
        }

        for (i, mesh) in scene.meshes.iter().enumerate() {
            let path = sdf::path(format!(
                "/Root/{}/{}_{}",
                usd_safe_name(&mesh.group),
                usd_safe_name(&mesh.name),
                i
            ))?;
            let material = mesh
                .material_id
                .map(|id| sdf::path(format!("/Materials/Material_{id}")))
                .transpose()?
                .filter(|m| self.data.has_spec(m));

            let mut properties: Vec<String> = [
                "doubleSided",
                "faceVertexCounts",
                "faceVertexIndices",
                "points",
                "primvars:displayColor",
                "subdivisionScheme",
            ]
            .map(String::from)
            .into();
            if mesh.uvs.is_some() {
                properties.push("primvars:st".into());
            }
            if material.is_some() {
                properties.push("material:binding".into());
            }
            if let Some(attributes) = &mesh.attributes {
                properties.extend(attributes.keys().map(|key| usd_safe_name(key)));
            }
            if mesh.visibility_times.is_some() {
                properties.push("visibility".into());
            }
            properties.sort();
            properties.dedup();
            let properties: Vec<&str> = properties.iter().map(String::as_str).collect();
            self.prim(&path, Some("Mesh"), &[], &properties);

            let points = mesh
                .points
                .chunks_exact(3)
                .map(|p| gf::Vec3f::from([p[0], p[1], p[2]]))
                .collect();
            self.attribute(
                &path.append_property("points")?,
                "point3f[]",
                [("default", Value::Vec3fVec(points))],
            );
            let indices = Value::IntVec(mesh.indices.clone());
            self.attribute(
                &path.append_property("faceVertexIndices")?,
                "int[]",
                [("default", indices)],
            );
            let counts = Value::IntVec(vec![3; mesh.indices.len() / 3]);
            self.attribute(
                &path.append_property("faceVertexCounts")?,
                "int[]",
                [("default", counts)],
            );
            let scheme = [uniform(), ("default", token("none"))];
            self.attribute(&path.append_property("subdivisionScheme")?, "token", scheme);
            let double_sided = [uniform(), ("default", Value::Bool(true))];
            self.attribute(&path.append_property("doubleSided")?, "bool", double_sided);
            let color = gf::Vec3f::from(mesh.display_color.unwrap_or([0.5, 0.5, 0.5]));
            self.attribute(
                &path.append_property("primvars:displayColor")?,
                "color3f[]",
                [
                    ("default", Value::Vec3fVec(vec![color])),
                    ("interpolation", token("constant")),
                ],
            );
            if let Some(uvs) = &mesh.uvs {
                let uvs = uvs.chunks_exact(2).map(|uv| gf::Vec2f::from([uv[0], uv[1]])).collect();
                self.attribute(
                    &path.append_property("primvars:st")?,
                    "texCoord2f[]",
                    [("default", Value::Vec2fVec(uvs)), ("interpolation", token("vertex"))],
                );
            }
            if let Some(material) = material {
                let binding = path.append_property("material:binding")?;
                self.data.create_spec(binding.clone(), SpecType::Relationship);
                self.set(&binding, "targetPaths", connection(material));
            }
            if let Some(attributes) = &mesh.attributes {
                for (key, value) in attributes {
                    let fields = [("default", Value::String(value.clone())), ("custom", Value::Bool(true))];
                    self.attribute(&path.append_property(usd_safe_name(key))?, "string", fields);
                }
            }
            if let Some(times) = &mesh.visibility_times {
                let samples = times.iter().map(|(t, v)| (*t, token(v))).collect();
                self.attribute(
                    &path.append_property("visibility")?,
                    "token",
                    [("timeSamples", Value::TimeSamples(samples))],
                );
            }
        }
        Ok(())
    }

    /// A `def` prim, typeless when `type_name` is `None`.
    fn prim(&mut self, path: &Path, type_name: Option<&str>, children: &[String], properties: &[&str]) {
        self.data.create_spec(path.clone(), SpecType::Prim);
        self.set(path, "specifier", Value::Specifier(Specifier::Def));
        if let Some(type_name) = type_name {
            self.set(path, "typeName", token(type_name));
        }
        if !children.is_empty() {
            self.set(path, "primChildren", tokens(children));
        }
        if !properties.is_empty() {
            self.set(path, "propertyChildren", tokens(properties));
        }
    }

    fn attribute(&mut self, path: &Path, type_name: &str, fields: impl IntoIterator<Item = (&'static str, Value)>) {
        self.data.create_spec(path.clone(), SpecType::Attribute);
        self.set(path, "typeName", token(type_name));
        for (field, value) in fields {
            self.set(path, field, value);
        }
    }

    fn set(&mut self, path: &Path, field: &str, value: Value) {
        AbstractData::set_field(&mut self.data, path, field, value);
    }
}

fn token(s: &str) -> Value {
    Value::Token(tf::Token::from(s))
}

fn tokens<S: AsRef<str>>(items: &[S]) -> Value {
    Value::TokenVec(items.iter().map(|s| tf::Token::from(s.as_ref())).collect())
}

fn uniform() -> (&'static str, Value) {
    ("variability", Value::Variability(Variability::Uniform))
}

/// An explicit list holding the one path an attribute connects to or a relationship targets.
fn connection(target: Path) -> Value {
    Value::PathListOp(sdf::ListOp::explicit([target]))
}

/// Replaces every character but ASCII letters, digits and `_` with `_`, and prefixes a leading digit.
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
    use openusd::sdf::ListOp;
    use openusd::usdc::CrateData;

    use super::*;

    fn mesh(name: &str, group: &str, points: Vec<f32>, indices: Vec<i32>) -> ExportMesh {
        ExportMesh {
            name: name.to_string(),
            group: group.to_string(),
            points,
            indices,
            uvs: None,
            display_color: Some([1.0, 0.0, 0.0]),
            material_id: None,
            attributes: None,
            visibility_times: None,
        }
    }

    fn scene(meshes: Vec<ExportMesh>, materials: Vec<ExportMaterial>) -> ExportScene {
        ExportScene {
            meshes,
            materials,
            time_codes_per_second: None,
            start_time_code: None,
            end_time_code: None,
        }
    }

    const TRIANGLE: [f32; 9] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];

    /// Writes `scene` and reads it back with upstream's reader, which decodes as C++ `CrateFile` does.
    fn roundtrip(scene: &ExportScene) -> Result<CrateData> {
        let bytes = build_usdc_scene(scene)?;
        assert_eq!(&bytes[0..8], b"PXR-USDC");
        CrateData::open(bytes, true).map_err(|e| anyhow!("{e}"))
    }

    fn field(data: &CrateData, path: &str, field: &str) -> Value {
        let path = sdf::path(path).unwrap();
        data.try_field(&path, field)
            .unwrap()
            .unwrap_or_else(|| panic!("{path} has no {field}"))
            .into_owned()
    }

    fn ints(value: Value) -> Vec<i32> {
        match value {
            Value::IntVec(v) => v,
            other => panic!("expected int[], got {other:?}"),
        }
    }

    fn token_str(value: &Value) -> &str {
        match value {
            Value::Token(t) => t.as_str(),
            other => panic!("expected a token, got {other:?}"),
        }
    }

    #[test]
    fn minimal_layer_metadata() -> Result<()> {
        let data = roundtrip(&scene(
            vec![mesh("TestCube", "Elements", TRIANGLE.to_vec(), vec![0, 1, 2])],
            vec![],
        ))?;
        assert_eq!(token_str(&field(&data, "/", "defaultPrim")), "Root");
        assert_eq!(token_str(&field(&data, "/", "upAxis")), "Y");
        match field(&data, "/", "primChildren") {
            Value::TokenVec(children) => assert_eq!(children, [tf::Token::from("Root")]),
            other => panic!("unexpected primChildren {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn mesh_data_and_custom_attributes() -> Result<()> {
        let mut tri = mesh(
            "Tri",
            "Geometry",
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 1.0, 0.0],
            vec![0, 1, 2],
        );
        tri.uvs = Some(vec![0.0, 0.0, 1.0, 0.0, 0.5, 1.0]);
        tri.attributes = Some(HashMap::from([
            ("IfcGUID".to_string(), "2O2Fr$t4X7Z8f0Q3MC_25F".to_string()),
            ("level".to_string(), "Level 1".to_string()),
        ]));
        let data = roundtrip(&scene(vec![tri], vec![]))?;

        match field(&data, "/Root/Geometry/Tri_0.points", "default") {
            Value::Vec3fVec(points) => assert_eq!(points[2], gf::Vec3f::from([0.5, 1.0, 0.0])),
            other => panic!("unexpected points {other:?}"),
        }
        assert_eq!(
            ints(field(&data, "/Root/Geometry/Tri_0.faceVertexIndices", "default")),
            [0, 1, 2]
        );
        assert_eq!(
            ints(field(&data, "/Root/Geometry/Tri_0.faceVertexCounts", "default")),
            [3]
        );
        let guid = field(&data, "/Root/Geometry/Tri_0.IfcGUID", "default");
        assert_eq!(guid, Value::String("2O2Fr$t4X7Z8f0Q3MC_25F".to_string()));
        assert_eq!(field(&data, "/Root/Geometry/Tri_0.level", "custom"), Value::Bool(true));
        Ok(())
    }

    /// The fork's own writer compressed int arrays of 4 to 15 entries, which C++ reads uncompressed below
    /// `MinCompressedArraySize` (16): `usdcat` read `[0, 1, 2, 0, 2, 3]` back as `[11, 0, 102400, ...]`.
    #[test]
    fn quad_indices_survive() -> Result<()> {
        let quad = mesh(
            "Quad",
            "Elements",
            [TRIANGLE.as_slice(), &[1.0, 1.0, 0.0]].concat(),
            vec![0, 1, 2, 0, 2, 3],
        );
        let data = roundtrip(&scene(vec![quad], vec![]))?;
        assert_eq!(
            ints(field(&data, "/Root/Elements/Quad_0.faceVertexIndices", "default")),
            [0, 1, 2, 0, 2, 3]
        );
        assert_eq!(
            ints(field(&data, "/Root/Elements/Quad_0.faceVertexCounts", "default")),
            [3, 3]
        );
        Ok(())
    }

    #[test]
    fn large_arrays_survive() -> Result<()> {
        let points: Vec<f32> = (0..300).map(|i| i as f32 * 0.1).collect();
        let indices: Vec<i32> = (0..99).collect();
        let data = roundtrip(&scene(
            vec![mesh("Big", "Test", points.clone(), indices.clone())],
            vec![],
        ))?;
        match field(&data, "/Root/Test/Big_0.points", "default") {
            Value::Vec3fVec(read) => {
                let flat: Vec<f32> = read.into_iter().flat_map(<[f32; 3]>::from).collect();
                assert_eq!(flat, points);
            }
            other => panic!("unexpected points {other:?}"),
        }
        assert_eq!(
            ints(field(&data, "/Root/Test/Big_0.faceVertexIndices", "default")),
            indices
        );
        assert_eq!(
            ints(field(&data, "/Root/Test/Big_0.faceVertexCounts", "default")),
            vec![3; 33]
        );
        Ok(())
    }

    #[test]
    fn visibility_samples_and_animation_range() -> Result<()> {
        let mut animated = mesh("AnimMesh", "Elements", TRIANGLE.to_vec(), vec![0, 1, 2]);
        animated.visibility_times = Some(vec![
            (0.0, "invisible".to_string()),
            (24.0, "inherited".to_string()),
            (72.0, "invisible".to_string()),
        ]);
        let mut scene = scene(vec![animated], vec![]);
        scene.time_codes_per_second = Some(24.0);
        scene.start_time_code = Some(0.0);
        scene.end_time_code = Some(72.0);
        let data = roundtrip(&scene)?;

        assert_eq!(field(&data, "/", "timeCodesPerSecond"), Value::Double(24.0));
        assert_eq!(field(&data, "/", "startTimeCode"), Value::Double(0.0));
        assert_eq!(field(&data, "/", "endTimeCode"), Value::Double(72.0));
        let Value::TimeSamples(samples) = field(&data, "/Root/Elements/AnimMesh_0.visibility", "timeSamples") else {
            panic!("visibility has no time samples");
        };
        let read: Vec<(f64, &str)> = samples.iter().map(|(t, v)| (*t, token_str(v))).collect();
        assert_eq!(read, [(0.0, "invisible"), (24.0, "inherited"), (72.0, "invisible")]);
        Ok(())
    }

    /// A binding is a relationship's `targetPaths`; the fork's writer stored it as `default`, which
    /// other tools ignore.
    #[test]
    fn material_binding_targets_material() -> Result<()> {
        let mut colored = mesh("ColoredMesh", "Elements", TRIANGLE.to_vec(), vec![0, 1, 2]);
        colored.material_id = Some(0);
        let material = ExportMaterial {
            id: 0,
            diffuse_color: Some([1.0, 0.0, 0.0]),
            metallic: Some(0.0),
            roughness: Some(0.8),
            textures: HashMap::new(),
        };
        let data = roundtrip(&scene(vec![colored], vec![material]))?;
        let binding = field(&data, "/Root/Elements/ColoredMesh_0.material:binding", "targetPaths");
        assert_eq!(
            binding,
            Value::PathListOp(ListOp::explicit([sdf::path("/Materials/Material_0")?]))
        );
        let color = field(&data, "/Materials/Material_0/PBRShader.inputs:diffuseColor", "default");
        assert_eq!(color, Value::Vec3f(gf::Vec3f::from([1.0, 0.0, 0.0])));
        Ok(())
    }

    #[test]
    fn textures_reach_the_viewer() -> Result<()> {
        let mut textured = mesh("TexturedMesh", "Elements", TRIANGLE.to_vec(), vec![0, 1, 2]);
        textured.uvs = Some(vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
        textured.material_id = Some(0);
        let material = ExportMaterial {
            id: 0,
            diffuse_color: Some([1.0, 1.0, 1.0]),
            metallic: Some(0.0),
            roughness: Some(0.8),
            textures: HashMap::from([
                ("diffuse".to_string(), "tex_0_diffuse.png".to_string()),
                ("normal".to_string(), "tex_0_normal.png".to_string()),
            ]),
        };
        let bytes = build_usdc_scene(&scene(vec![textured], vec![material]))?;

        let result = crate::wasm::parse_usd_meshes_inner(&bytes)?;
        assert!(result.error.is_none(), "parse error: {:?}", result.error);
        let [m] = result.meshes.as_slice() else {
            panic!("expected one mesh")
        };
        assert_eq!(m.diffuse_texture.as_deref(), Some("tex_0_diffuse.png"));
        assert_eq!(m.normal_texture.as_deref(), Some("tex_0_normal.png"));
        assert_eq!(m.texcoords.as_ref().map(Vec::len), Some(6));
        Ok(())
    }

    /// Older USD releases refuse a crate version newer than theirs; C++ writes 0.8.0 for this content.
    #[test]
    fn exports_are_crate_version_0_8_0() -> Result<()> {
        let bytes = build_usdc_scene(&scene(
            vec![mesh("Tri", "Elements", TRIANGLE.to_vec(), vec![0, 1, 2])],
            vec![],
        ))?;
        assert_eq!(bytes[VERSION_OFFSET..VERSION_OFFSET + 3], [0, 8, 0]);
        Ok(())
    }

    /// A box has 12 triangles: 12 face counts, a length in the band C++ reads uncompressed.
    #[test]
    fn box_face_counts_survive() -> Result<()> {
        let points: Vec<f32> = (0..8)
            .flat_map(|i| [(i & 1) as f32, (i >> 1 & 1) as f32, (i >> 2) as f32])
            .collect();
        let indices = vec![
            0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5, 0, 4, 5, 0, 5, 1, 2, 3, 7, 2, 7, 6, 0, 2, 6, 0, 6, 4, 1, 5, 7, 1, 7, 3,
        ];
        let data = roundtrip(&scene(vec![mesh("Box", "Elements", points, indices.clone())], vec![]))?;
        assert_eq!(
            ints(field(&data, "/Root/Elements/Box_0.faceVertexIndices", "default")),
            indices
        );
        assert_eq!(
            ints(field(&data, "/Root/Elements/Box_0.faceVertexCounts", "default")),
            [3; 12]
        );
        Ok(())
    }
}
