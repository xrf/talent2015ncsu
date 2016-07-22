extern crate docopt;
extern crate ndarray;
extern crate rand;
extern crate rustc_serialize;
#[macro_use]
extern crate qmc;

use std::io::Write;
use ndarray::prelude::*;
use rand::Rng;
use rand::distributions::{IndependentSample, Range};

use qmc::utils::*;

const USAGE: &'static str = "
Usage:
  markov [options]

Options:

  -h --help
    Show this screen.

  -s --seed=<seed>
    Seed for the random number generator in LE hexadecimal.
    [default is empty string]

  --num-samples=<num-samples>
    [default: 1000000]

";

#[derive(Debug, RustcDecodable)]
struct Args {
    flag_seed: String,
    flag_num_samples: u64,
}

fn arr2_powi(base: &ArrayView<f64, (usize, usize)>,
             power: u64)
             -> Array<f64, (usize, usize)> {
    let mut m = Array::eye(base.shape()[0]);
    for _ in 0 .. power {
        m = m.dot(base);
    }
    m
}

fn get_sample<R: Rng>(rng: &mut R,
                      transition_matrix: &ArrayView<f64, (usize, usize)>,
                      num_samples: u64) -> (u64, u64) {
    let range = Range::new(0, 3);
    let mut state = vec![0; 21];
    let mut s_18_a = 0;
    let mut s_18_b = 0;
    for _ in 0 .. num_samples {
        for i in 1 .. state.len() - 1 {
            let x_new = range.ind_sample(rng);
            let p =
                (transition_matrix[(state[i - 1], x_new)] *
                 transition_matrix[(x_new, state[i + 1])]) /
                (transition_matrix[(state[i - 1], state[i])] *
                 transition_matrix[(state[i], state[i + 1])]);
            if rng.next_f64() < p {
                state[i] = x_new;
            }
        }
        if state[2] == 0 {
            s_18_a += 1;
        }
        if state[2] == 0 && state[1] == 0 {
            s_18_b += 1;
        }
    }
    (s_18_a, s_18_b)
}

fn main() {
    let args: Args = docopt::Docopt::new(USAGE)
        .and_then(|d| d.decode())
        .unwrap_or_else(|e| e.exit());
    let seed = decode_seed(&args.flag_seed);

    println!("# args: {:?}", args);

    let mut rng = rng_from_seed(&seed);

    let transition_matrix =
        arr2(&[[1.1, 0.1, 0.1],
               [0.1, 0.8, 0.1],
               [0.1, 0.1, 0.8]]);

    let state0 = arr1(&[1.0, 0.0, 0.0]);

    let f = |n| state0
        .dot(&arr2_powi(&transition_matrix.view(), n)
             .dot(&state0));

    let (r_a, r_b) = {
        print!("#");
        let num_samples_unit = args.flag_num_samples / 100;
        let mut samples_collected = 0;
        let mut s_18_a = 0;
        let mut s_18_b = 0;
        loop {
            let num_samples = std::cmp::min(
                num_samples_unit,
                args.flag_num_samples - samples_collected
            );
            if num_samples == 0 {
                break;
            }
            let (a, b) = get_sample(&mut rng,
                                    &transition_matrix.view(),
                                    num_samples);
            s_18_a += a;
            s_18_b += b;
            samples_collected += num_samples;
            print!("\r# {:3}%",
                   samples_collected * 100 / args.flag_num_samples);
            std::io::stdout().flush().unwrap();
        }
        println!("\r#     ");
        let a = f(2);
        let b = transition_matrix[(0, 0)].powi(2);
        let s_20 = args.flag_num_samples;
        ((s_20 as f64) / (s_18_a as f64) * a,
         (s_20 as f64) / (s_18_b as f64) * b)
    };

    println!("{{");
    println!("'expression': 'f(20) / f(18)',");
    println!("'exact_result': {},", f(20) / f(18));
    println!("'mc_result_a': {:?},", r_a);
    println!("'mc_result_b': {:?},", r_b);
    println!("}}");

}
