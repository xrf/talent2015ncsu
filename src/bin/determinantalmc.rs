extern crate docopt;
extern crate la;
extern crate ndarray;
extern crate num;
extern crate rand;
extern crate rustc_serialize;
#[macro_use]
extern crate qmc;

use std::f64::consts::PI;
use std::io::{Write, stderr, stdout};
use ndarray::prelude::*;
use rand::Rng;
use rand::distributions::{IndependentSample, Range};

use qmc::utils::*;

const USAGE: &'static str = "
Usage:
  determinantalmc [options]

Options:

  -h --help
    Show this screen.

  -s --seed=<seed>
    Seed for the random number generator in LE hexadecimal.
    [default is empty string]

  --num-samples=<num-samples>
    [default: 1000000]

  --param-nx=<param-nx>
    [default: 10]

  --param-nt=<param-nt>
    [default: 10]

  --rate=<rate>
    Determines how much we should perturb the values of the auxiliary field.
    [default: 1.0]

";

#[derive(Debug, RustcDecodable)]
struct Args {
    flag_seed: String,
    flag_num_samples: usize,
    flag_param_nx: usize,
    flag_param_nt: usize,
    flag_rate: f64,
}

fn main() {
    let args: Args = docopt::Docopt::new(USAGE)
        .and_then(|d| d.decode())
        .unwrap_or_else(|e| e.exit());
    let seed = decode_seed(&args.flag_seed);

    println!("# args: {:?}", args);
    stdout().flush().unwrap();

    let mut rng = rng_from_seed(&seed);

    writeln!(stderr(), "# Calculating analytically ...").unwrap();

    let nx = args.flag_param_nx;
    let nt = args.flag_param_nt;
    let tau: f64 = 0.001;
    let a: f64 = 1.0;
    let m = 1.0;
    let g = 0.1;
    let sqrt_c = (tau * g).exp() - 1.0;

    let x_range = Range::new(0, nx);
    let t_range = Range::new(0, nt);
    let kinetic = {
        let mut t = Array::zeros((nx, nx));
        for i in 0 .. nx {
            t[(i, i)] = -2.0;
            t[(i, (i + 1) % nx)] = 1.0;
            t[(i, (i + nx - 1) % nx)] = 1.0;
        }
        // -tau T / 2 = (-tau / 2) (-1 / 2 m) (1 / a^2) finite_difference
        t *= 0.25 * tau / (m * a.powi(2));
        t
    };
    let evolve_kinetic = Array::eye(nx) - kinetic;
    let mut sigma = Array::from_elem((nx, nt), 1i8);
    for _ in 0 .. 10 {

        let x_change = x_range.ind_sample(&mut rng);
        let t_change = t_range.ind_sample(&mut rng);
        sigma[(x_change, t_change)] *= -1;

        let mut transfer = Array::eye(nx);
        for t in 0 .. nt {
            transfer = evolve_kinetic.dot(&transfer);
            for i in 0 .. nx {
                for x in 0 .. nx {
                    transfer[(x, i)] *= 1.0 + sqrt_c * sigma[(x, t)] as f64;
                }
            }
            transfer = evolve_kinetic.dot(&transfer);
        }
        let fermi = Array::eye(nx) + transfer;
        let detsq =
            ::la::Matrix::new(nx, nx, fermi.iter().cloned().collect())
            .det().powi(2);
        println!("{}", detsq);

    }

    let value = 0.0;

    println!("{{");
    println!("'value': {:?},", value);
    println!("}}");
}
