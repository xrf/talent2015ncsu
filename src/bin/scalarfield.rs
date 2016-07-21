extern crate docopt;
extern crate ndarray;
extern crate num;
extern crate rand;
extern crate rustc_serialize;
#[macro_use]
extern crate qmc;

use std::io::Write;
use std::f64::consts::PI;
use ndarray::prelude::*;
use num::complex::Complex;
use rand::Rng;
use rand::distributions::{IndependentSample, Range};

use qmc::utils::*;

const USAGE: &'static str = "
Usage:
  scalarfield [options]

Options:

  -h --help
    Show this screen.

  -s --seed=<seed>
    Seed for the random number generator in LE hexadecimal.
    [default is empty string]

  --num-samples=<num-samples>
    [default: 1000000]

  --param-l=<param_l>
    [default: 10]

  --param-lt=<param_lt>
    [default: 10]

  --param-m=<param_m>
    [default: 10]

";

#[derive(Debug, RustcDecodable)]
struct Args {
    flag_seed: String,
    flag_num_samples: usize,
    flag_param_l: u64,
    flag_param_lt: u64,
    flag_param_m: f64,
}

fn main() {
    let args: Args = docopt::Docopt::new(USAGE)
        .and_then(|d| d.decode())
        .unwrap_or_else(|e| e.exit());
    let seed = decode_seed(&args.flag_seed);

    println!("# args: {:?}", args);

    let mut rng = rng_from_seed(&seed);

    let l = args.flag_param_l;
    let lt = args.flag_param_lt;
    let m = args.flag_param_m;
    let kunit = 2.0 * PI / (l as f64);
    let n_minus_m = [0, 0, 0, 0];

    let pcf = {
        let mut sum = Complex::new(0.0, 0.0);
        for kix in 0 .. l {
            for kiy in 0 .. l {
                for kiz in 0 .. l {
                    for kit in 0 .. l {
                        sum = sum +
                            Complex::new(
                                0.0,
                                kunit * (
                                    (
                                        kix * n_minus_m[0]
                                      + kiy * n_minus_m[1]
                                      + kiz * n_minus_m[2]
                                      + kit * n_minus_m[3]
                                    ) as f64
                                )
                            ).exp()
                            / (2.0 * (
                                4.0
                              - (kunit * (kix as f64)).cos()
                              - (kunit * (kiy as f64)).cos()
                              - (kunit * (kiz as f64)).cos()
                              - (kunit * (kit as f64)).cos()
                            ) + m.powi(2));
                    }
                }
            }
        }
        sum / ((l as f64).powi(3) * (lt as f64))
    };
    println!("{{");
    println!("'analytic': {},", pcf);
    println!("}}");
}
