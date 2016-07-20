extern crate num;
extern crate rand;
extern crate rustc_serialize;

use std::error::Error;
use std::ops::{BitOrAssign, Shl};
use rand::SeedableRng;

pub type MyRng = rand::isaac::Isaac64Rng;

#[macro_export]
#[cfg(debug_assertions)]
macro_rules! ix {
    ( $v:expr , $i:expr ) => ($v[$i])
}

#[macro_export]
#[cfg(not(debug_assertions))]
macro_rules! ix {
    ( $v:expr , $i:expr ) => (*(unsafe { $v.get_unchecked($i) }))
}

#[macro_export]
#[cfg(debug_assertions)]
macro_rules! ix_mut {
    ( $v:expr , $i:expr ) => ($v[$i])
}

#[macro_export]
#[cfg(not(debug_assertions))]
macro_rules! ix_mut {
    ( $v:expr , $i:expr ) => (*(unsafe { $v.get_unchecked_mut($i) }))
}

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
pub fn decode_hex_le(string: &str)
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

pub fn decode_seed(seed: &str) -> Vec<u8> {
    decode_hex_le(seed).unwrap_or_else(|e| {
        panic!("invalid hexadecimal value: {} ({})", seed, e.description())
    })
}
