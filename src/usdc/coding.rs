//! Integer coding.
//!
//! See <https://github.com/PixarAnimationStudios/OpenUSD/blob/0b18ad3f840c24eb25e16b795a5b0821cf05126e/pxr/usd/usd/integerCoding.cpp#L40>

use std::{io, mem};

use anyhow::{bail, Result};
use num_traits::{AsPrimitive, PrimInt};

use super::reader::ReadExt;

const COMMON: u8 = 0;
const SMALL: u8 = 1;
const MEDIUM: u8 = 2;
const LARGE: u8 = 3;

pub fn encoded_buffer_size<T: PrimInt>(count: usize) -> usize {
    if count == 0 {
        0
    } else {
        let sz = mem::size_of::<T>();
        sz + (count * 2).div_ceil(8) + (sz * count)
    }
}

pub fn decode_ints<T: PrimInt + 'static>(data: &[u8], count: usize) -> Result<Vec<T>>
where
    i64: AsPrimitive<T>,
{
    let is_64_bit = mem::size_of::<T>() == 8;

    let mut codes_reader = io::Cursor::new(&data[0..]);

    let common_value = if is_64_bit {
        codes_reader.read_pod::<i64>()?
    } else {
        codes_reader.read_pod::<i32>()? as i64
    };

    let num_code_bytes = (count * 2).div_ceil(8);

    let mut ints_reader = {
        let offset = mem::size_of::<T>() + num_code_bytes;
        io::Cursor::new(&data[offset..])
    };

    let mut prev = 0_i64;
    let mut ints_left = count as isize;
    let mut output = Vec::new();

    while ints_left > 0 {
        let n = ints_left.min(4);
        ints_left -= 4;

        // Code byte stores integer types for the next 4 integers.
        let code_byte = codes_reader.read_pod::<u8>()?;

        for i in 0..n {
            let ty = (code_byte >> (2 * i)) & 3;
            let delta = match ty {
                COMMON => common_value,

                // 64 bits targets
                SMALL if is_64_bit => ints_reader.read_pod::<i16>()? as i64,
                MEDIUM if is_64_bit => ints_reader.read_pod::<i32>()? as i64,
                LARGE if is_64_bit => ints_reader.read_pod::<i64>()?,

                // 32 bits
                SMALL => ints_reader.read_pod::<i8>()? as i64,
                MEDIUM => ints_reader.read_pod::<i16>()? as i64,
                LARGE => ints_reader.read_pod::<i32>()? as i64,

                _ => bail!("Unexpected index: {ty}"),
            };

            prev += delta;

            output.push(prev.as_());
        }
    }

    Ok(output)
}

/// Encode a slice of integers using USD's delta + variable-width integer coding.
///
/// This is the inverse of `decode_ints`. The format:
/// 1. Common value (i32 or i64 depending on T size)
/// 2. Code bytes (2 bits per integer, 4 integers per byte)
/// 3. Data bytes (variable-width deltas)
pub fn encode_ints<T: PrimInt + 'static>(values: &[T]) -> Vec<u8>
where
    T: AsPrimitive<i64>,
{
    let count = values.len();
    if count == 0 {
        return Vec::new();
    }

    let is_64_bit = mem::size_of::<T>() == 8;

    // Compute deltas from running sum.
    let mut deltas = Vec::with_capacity(count);
    let mut prev: i64 = 0;
    for &v in values {
        let cur: i64 = v.as_();
        deltas.push(cur - prev);
        prev = cur;
    }

    // Find most common delta.
    let mut freq: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
    for &d in &deltas {
        *freq.entry(d).or_insert(0) += 1;
    }
    let common_value = *freq.iter().max_by_key(|(_, c)| *c).unwrap().0;

    // Classify each delta and build code bytes + data bytes.
    let num_code_bytes = (count * 2).div_ceil(8);
    let mut codes = vec![0u8; num_code_bytes];
    let mut data_bytes = Vec::new();

    for (i, &delta) in deltas.iter().enumerate() {
        let code_byte_idx = i / 4;
        let code_bit_pos = (i % 4) * 2;

        let code = if delta == common_value {
            COMMON
        } else if is_64_bit {
            if delta >= i16::MIN as i64 && delta <= i16::MAX as i64 {
                data_bytes.extend_from_slice(&(delta as i16).to_le_bytes());
                SMALL
            } else if delta >= i32::MIN as i64 && delta <= i32::MAX as i64 {
                data_bytes.extend_from_slice(&(delta as i32).to_le_bytes());
                MEDIUM
            } else {
                data_bytes.extend_from_slice(&delta.to_le_bytes());
                LARGE
            }
        } else {
            if delta >= i8::MIN as i64 && delta <= i8::MAX as i64 {
                data_bytes.extend_from_slice(&(delta as i8).to_le_bytes());
                SMALL
            } else if delta >= i16::MIN as i64 && delta <= i16::MAX as i64 {
                data_bytes.extend_from_slice(&(delta as i16).to_le_bytes());
                MEDIUM
            } else {
                data_bytes.extend_from_slice(&(delta as i32).to_le_bytes());
                LARGE
            }
        };

        codes[code_byte_idx] |= code << code_bit_pos;
    }

    // Assemble output: common_value + codes + data.
    let mut output = Vec::with_capacity(mem::size_of::<T>() + codes.len() + data_bytes.len());
    if is_64_bit {
        output.extend_from_slice(&common_value.to_le_bytes());
    } else {
        output.extend_from_slice(&(common_value as i32).to_le_bytes());
    }
    output.extend_from_slice(&codes);
    output.extend_from_slice(&data_bytes);

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode() {
        /*
        See https://github.com/PixarAnimationStudios/OpenUSD/blob/0b18ad3f840c24eb25e16b795a5b0821cf05126e/pxr/usd/usd/integerCoding.cpp#L85
        input  = [123, 124, 125, 100125, 100125, 100126, 100126]
        output = [int32(1) 01 00 00 11 01 00 01 XX int8(123) int32(100000) int8(0) int8(0)]
        */

        let mut output = Vec::new();
        output.extend_from_slice(&1_i32.to_le_bytes());
        debug_assert_eq!(output.len(), 4);

        // Little endian, swap bytes.
        let codes: u16 = 0b1100_0001_0001_0001;
        output.extend_from_slice(&codes.to_be_bytes());
        debug_assert_eq!(output.len(), 6);

        output.extend_from_slice(&123_i8.to_le_bytes());
        output.extend_from_slice(&100000_i32.to_le_bytes());
        output.extend_from_slice(&0_i16.to_le_bytes());

        let input = decode_ints::<u32>(&output, 7).expect("Failed to decode integers");

        assert_eq!(input.as_slice(), &[123_u32, 124, 125, 100125, 100125, 100126, 100126])
    }

    #[test]
    fn test_encode_roundtrip_u32() {
        let original: Vec<u32> = vec![123, 124, 125, 100125, 100125, 100126, 100126];
        let encoded = encode_ints(&original);
        let decoded = decode_ints::<u32>(&encoded, original.len()).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_encode_roundtrip_i32() {
        let original: Vec<i32> = vec![1, 2, 4, 5, -3, 4, 5, -2, 3, 0];
        let encoded = encode_ints(&original);
        let decoded = decode_ints::<i32>(&encoded, original.len()).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_encode_roundtrip_zeros() {
        let original: Vec<u32> = vec![0, 0, 0, 0, 0];
        let encoded = encode_ints(&original);
        let decoded = decode_ints::<u32>(&encoded, original.len()).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_encode_roundtrip_sequential() {
        let original: Vec<u32> = (0..20).collect();
        let encoded = encode_ints(&original);
        let decoded = decode_ints::<u32>(&encoded, original.len()).unwrap();
        assert_eq!(decoded, original);
    }
}
