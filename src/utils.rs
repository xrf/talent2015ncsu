extern crate num;
extern crate rand;
extern crate rustc_serialize;

use std::error::Error;
use std::ops::{BitOrAssign, Shl};
use rand::SeedableRng;

pub type MyRng = rand::isaac::Isaac64Rng;

// FIXME: does this work properly for signed integers??
pub fn repack_u8s<T>(s: &[u8]) -> Vec<T> where
    T: BitOrAssign + Shl<usize, Output = T> + num::NumCast {
    let mut v = Vec::new();
    let mut k = 0;
    for b in s {
        let b = num::cast::cast(*b).unwrap();
        if k == 0 {
            v.push(b);
        } else {
            *v.last_mut().unwrap() |= b << k * 8;
        }
        k = (k + 1) % ::std::mem::size_of::<T>();
    }
    v
}

pub fn rng_from_seed(seed: &[u8]) -> MyRng {
    MyRng::from_seed(&repack_u8s(seed))
}

/// Data type used for command-line arguments.
#[derive(Debug)]
pub struct RandomSeed {
    /// Value of the seed.
    pub value: Vec<u8>,
}

/// Decode a hex byte string in a little endian format.  If the last byte is
/// contains only a single digit, it is padded with zero.  For example,
/// `abcde` is decoded as `[0xba, 0xdc, 0x0e]` or `[186, 220, 14]`.
fn decode_hex_le(string: &str)
                 -> Result<Vec<u8>, rustc_serialize::hex::FromHexError> {
    let mut s = String::new();
    let mut prev_c = None;
    for c in string.chars() {
        match prev_c {
            Some(c2) => {
                s.push(c);
                s.push(c2);
                prev_c = None;
            },
            None => {
                prev_c = Some(c)
            },
        }
    }
    if let Some(c2) = prev_c {
        s.push('0');
        s.push(c2);
    }
    rustc_serialize::hex::FromHex::from_hex(s.as_str())
}

impl rustc_serialize::Decodable for RandomSeed {
    fn decode<D: rustc_serialize::Decoder>(d: &mut D)
                                           -> Result<RandomSeed, D::Error> {
        decode_hex_le(&try!(d.read_str()))
            .map(|x| RandomSeed { value: x.to_vec() })
            .map_err(|e| d.error(e.description()))
    }
}
