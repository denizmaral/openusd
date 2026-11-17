//! Read access to one USD layer, in the value shapes the mesh extraction was written against.
//!
//! Upstream `openusd` holds typed values (`gf::Vec3f`, `tf::Token`, `sdf::AssetPath`). The extraction
//! code matches flat vectors and plain strings, so each field read here is converted to those shapes.
//! Arrays and scalars of a vector type both flatten into one `Vec`, as the earlier reader returned them.

use std::borrow::Cow;
use std::collections::HashMap;

use anyhow::{anyhow, Context, Result};
use half::f16;
use openusd::sdf::{AbstractData as _, Path};
use openusd::vt;

/// A list edit with its item lists exposed as fields.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ListOp<T> {
    pub explicit: bool,
    pub explicit_items: Vec<T>,
    pub added_items: Vec<T>,
    pub prepended_items: Vec<T>,
    pub appended_items: Vec<T>,
    pub deleted_items: Vec<T>,
    pub ordered_items: Vec<T>,
}

impl From<&openusd::sdf::PathListOp> for ListOp<Path> {
    fn from(op: &openusd::sdf::PathListOp) -> Self {
        Self {
            explicit: op.is_explicit(),
            explicit_items: op.explicit_items().to_vec(),
            added_items: op.added_items().to_vec(),
            prepended_items: op.prepended_items().to_vec(),
            appended_items: op.appended_items().to_vec(),
            deleted_items: op.deleted_items().to_vec(),
            ordered_items: op.ordered_items().to_vec(),
        }
    }
}

/// A field value. Kinds the extraction does not read become [`Value::Other`].
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Bool(bool),
    Int(i32),
    Int64(i64),
    Half(f16),
    Float(f32),
    Double(f64),
    String(String),
    StringVec(Vec<String>),
    Token(String),
    TokenVec(Vec<String>),
    AssetPath(String),
    IntVec(Vec<i32>),
    HalfVec(Vec<f16>),
    FloatVec(Vec<f32>),
    DoubleVec(Vec<f64>),
    Vec2h(Vec<f16>),
    Vec2f(Vec<f32>),
    Vec2d(Vec<f64>),
    Vec3h(Vec<f16>),
    Vec3f(Vec<f32>),
    Vec3d(Vec<f64>),
    Matrix4d(Vec<f64>),
    PathListOp(ListOp<Path>),
    VariantSelectionMap(HashMap<String, String>),
    TimeSamples(Vec<(f64, Value)>),
    Other,
}

/// Flattens a scalar or an array of a fixed-size vector type into one `Vec`.
macro_rules! flat {
    (one $v:expr, $t:ty) => {
        <$t>::from(*$v).to_vec()
    };
    (many $v:expr, $t:ty) => {
        $v.iter().flat_map(|v| <$t>::from(*v)).collect()
    };
}

impl From<&vt::Value> for Value {
    fn from(value: &vt::Value) -> Self {
        use vt::Value as V;
        match value {
            V::Bool(b) => Value::Bool(*b),
            V::Int(i) => Value::Int(*i),
            V::Int64(i) => Value::Int64(*i),
            V::Half(h) => Value::Half(*h),
            V::Float(f) => Value::Float(*f),
            V::Double(d) => Value::Double(*d),
            V::String(s) => Value::String(s.clone()),
            V::StringVec(s) => Value::StringVec(s.clone()),
            V::Token(t) => Value::Token(t.as_str().to_owned()),
            V::TokenVec(t) => Value::TokenVec(t.iter().map(|t| t.as_str().to_owned()).collect()),
            V::AssetPath(a) => Value::AssetPath(a.as_str().to_owned()),
            V::IntVec(v) => Value::IntVec(v.clone()),
            V::HalfVec(v) => Value::HalfVec(v.clone()),
            V::FloatVec(v) => Value::FloatVec(v.clone()),
            V::DoubleVec(v) => Value::DoubleVec(v.clone()),
            V::Vec2h(v) => Value::Vec2h(flat!(one v, [f16; 2])),
            V::Vec2hVec(v) => Value::Vec2h(flat!(many v, [f16; 2])),
            V::Vec2f(v) => Value::Vec2f(flat!(one v, [f32; 2])),
            V::Vec2fVec(v) => Value::Vec2f(flat!(many v, [f32; 2])),
            V::Vec2d(v) => Value::Vec2d(flat!(one v, [f64; 2])),
            V::Vec2dVec(v) => Value::Vec2d(flat!(many v, [f64; 2])),
            V::Vec3h(v) => Value::Vec3h(flat!(one v, [f16; 3])),
            V::Vec3hVec(v) => Value::Vec3h(flat!(many v, [f16; 3])),
            V::Vec3f(v) => Value::Vec3f(flat!(one v, [f32; 3])),
            V::Vec3fVec(v) => Value::Vec3f(flat!(many v, [f32; 3])),
            V::Vec3d(v) => Value::Vec3d(flat!(one v, [f64; 3])),
            V::Vec3dVec(v) => Value::Vec3d(flat!(many v, [f64; 3])),
            V::Matrix4d(m) => Value::Matrix4d(m.0.to_vec()),
            V::Matrix4dVec(m) => Value::Matrix4d(m.iter().flat_map(|m| m.0).collect()),
            V::PathListOp(op) => Value::PathListOp(op.into()),
            V::VariantSelectionMap(m) => Value::VariantSelectionMap(m.clone()),
            V::TimeSamples(samples) => Value::TimeSamples(samples.iter().map(|(t, v)| (*t, v.into())).collect()),
            _ => Value::Other,
        }
    }
}

/// Read access to the specs and fields of one layer.
pub trait AbstractData {
    /// Whether the layer has a spec at `path`.
    fn has_spec(&self, path: &Path) -> bool;

    /// The value of `field` at `path`; an error when it is not authored or cannot be decoded.
    fn get(&mut self, path: &Path, field: &str) -> Result<Cow<'_, Value>>;
}

/// One USDC or USDA layer read from bytes.
pub struct Layer(Box<dyn openusd::sdf::AbstractData>);

impl Layer {
    /// Reads a layer, telling USDC from USDA by the crate file's magic bytes.
    pub fn open(data: &[u8]) -> Result<Self> {
        if data.starts_with(openusd::usdc::MAGIC) {
            let crate_data = openusd::usdc::CrateData::open(data.to_vec(), true)
                .map_err(|e| anyhow!("{e}"))
                .context("Failed to parse USDC binary")?;
            return Ok(Self(Box::new(crate_data)));
        }
        let text = std::str::from_utf8(data).context("Invalid UTF-8 in USDA file")?;
        let text = text.strip_prefix('\u{FEFF}').unwrap_or(text);
        let parsed = openusd::usda::parse(text)
            .map_err(|e| anyhow!("{e}"))
            .context("Failed to parse USDA text")?;
        Ok(Self(Box::new(parsed)))
    }
}

impl AbstractData for Layer {
    fn has_spec(&self, path: &Path) -> bool {
        self.0.has_spec(path)
    }

    fn get(&mut self, path: &Path, field: &str) -> Result<Cow<'_, Value>> {
        // The crate format names a prim's property list `properties`; upstream reads it as `propertyChildren`.
        let field = if field == "properties" {
            "propertyChildren"
        } else {
            field
        };
        let value = self
            .0
            .try_field(path, field)
            .map_err(|e| anyhow!("{e}"))?
            .ok_or_else(|| anyhow!("{path} has no {field}"))?;
        Ok(Cow::Owned(Value::from(&*value)))
    }
}
