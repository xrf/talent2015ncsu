extern crate modulo;
extern crate rand;

use std::io::Write;
use std::f64;
use std::f64::consts::PI;
use modulo::Mod;
use rand::{Rng, SeedableRng};

/// `weigh` is a function used to evaluate the relative likelihood of the
/// given sample (e.g. a partition function in statistical mechanics).  It
/// must be scaled to return a value between 0 and 1.
///
/// `sampler` is a function that generates a random sample.
fn rejection_sample<R, T: ?Sized, W, S>(rng: &mut R,
                                        sample: &mut T,
                                        weigh: W,
                                        sampler: S,
                                        max_attempts: u64) -> u64 where
    R: Rng,
    W: Fn(&T) -> f64,
    S: Fn(&mut R, &mut T) -> () {
    for i in 0 .. max_attempts {
        sampler(rng, sample);
        if rng.next_f64() < weigh(sample) {
            println!("");
            std::io::stdout().flush().unwrap();
            return i;
        }
        print!(".");
        std::io::stdout().flush().unwrap();
    }
    max_attempts
}

/// `weigh` is a function used to evaluate the relative likelihood of the
/// given sample (e.g. a partition function in statistical mechanics).  It
/// must return a positive value.
///
/// `sample0` is the initial sample.
///
/// `perturb` generates a new sample from an existing sample.
///
/// `num_steps` is the number of Metropolis steps to run before returning the
/// sample.
fn metropolis_sample<'a, R, T: ?Sized, W, P>(
        rng: &mut R,
        mut sample: &'a mut T,
        weigh: W,
        perturb: P,
        num_steps: u64,
        mut sample_buf: &'a mut T) -> (&'a mut T, u64) where
    R: Rng,
    W: Fn(&T) -> f64,
    P: Fn(&mut R, &T, &mut T) {
    let mut weight = weigh(sample);
    let mut accepts = 0;
    for _ in 0 .. num_steps {
        perturb(rng, sample, sample_buf);
        let new_weight = weigh(sample_buf);
        if rng.next_f64() * weight < new_weight {
            std::mem::swap(&mut sample, &mut sample_buf);
            weight = new_weight;
            accepts += 1;
            print!("o");
        } else {
            print!(".");
        }
        std::io::stdout().flush().unwrap();
    }
    println!("");
    std::io::stdout().flush().unwrap();
    (sample, accepts)
}

fn potential_v1(r: f64) -> f64 {
    1. + (20. * PI * r).cos()
}

fn total_potential_v1(r: &[f64]) -> f64 {
    r.into_iter()
        .map(|ri| potential_v1(*ri))
        .fold(0.0, |s, x| { s + x })
}

fn potential_v2(v0: f64, r1: f64, r2: f64) -> f64 {
    v0 * if (r1 - r2).abs() < 0.05 { 1.0 } else { 0.0 }
}

fn total_potential_v2(v0: f64, r: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0 .. r.len() {
        for j in 0 .. i {
            s += potential_v2(v0, r[i], r[j]);
        }
    }
    s
}

fn partition_function(beta: f64, v0: f64, r: &[f64]) -> f64 {
    (-beta * (total_potential_v1(r) + total_potential_v2(v0, r))).exp()
}

pub type RngType = rand::isaac::Isaac64Rng;

#[no_mangle]
pub unsafe extern fn qmc_rng_new(seed: *const u64,
                                 seed_len: u64) -> *mut RngType {
    Box::into_raw(Box::new(RngType::from_seed(
        std::slice::from_raw_parts(seed, seed_len as usize))))
}

#[no_mangle]
pub unsafe extern fn qmc_rng_del(rng: *mut RngType) {
    assert!(!rng.is_null());
    Box::from_raw(rng);
}

#[no_mangle]
pub unsafe extern fn qmc_cmc_rejection(rng: *mut RngType,
                                       beta: f64,
                                       v0: f64,
                                       num_particles: u64,
                                       r: *mut f64) -> u64{
    assert!(!rng.is_null());
    assert!(!r.is_null());
    cmc_rejection(&mut *rng, beta, v0,
                  std::slice::from_raw_parts_mut(r, num_particles as usize))
}

fn cmc_rejection(rng: &mut RngType,
                 beta: f64,
                 v0: f64,
                 r: &mut [f64]) -> u64 {
    rejection_sample(rng, r, |r| {
        partition_function(beta, v0, r)
    }, |rng, r| {
        for r in r.iter_mut() {
            *r = rng.next_f64();
        }
    }, 10000)
}

#[no_mangle]
pub unsafe extern fn qmc_cmc_metropolis(rng: *mut RngType,
                                        beta: f64,
                                        v0: f64,
                                        num_particles: u64,
                                        num_steps: u64,
                                        r: *mut f64,
                                        r2: *mut f64) {
    assert!(!rng.is_null());
    assert!(!r.is_null());
    assert!(!r2.is_null());
    cmc_metropolis(&mut *rng, beta, v0, num_steps,
                   std::slice::from_raw_parts_mut(r, num_particles as usize),
                   std::slice::from_raw_parts_mut(r2, num_particles as usize))
}

pub extern fn cmc_metropolis(rng: &mut RngType,
                             beta: f64,
                             v0: f64,
                             num_steps: u64,
                             r: &mut [f64],
                             r2: &mut [f64]) {
    assert!(r.len() == r2.len());
    let step_size = 0.3;
    metropolis_sample(rng, r, |r| {
        partition_function(beta, v0, r)
    }, |rng, r, r2| {
        for i in 0 .. r.len() {
            r2[i] = (r[i] + step_size * (rng.next_f64() - 0.5)).modulo(1.0);
        }
    }, num_steps, r2); // FIXME: return value is needed
}
