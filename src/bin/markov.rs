extern crate docopt;
extern crate ndarray;
extern crate rand;
extern crate rustc_serialize;
#[macro_use]
extern crate qmc;

use ndarray::prelude::*;

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

  --population=<population>
    Walker population.
    [default: 10000]

";

#[derive(Debug, RustcDecodable)]
struct Args {
    flag_seed: String,
    flag_population: usize,
}

fn arr2_powi(base: &ArrayBase<Vec<f64>, (usize, usize)>,
             power: u64)
             -> ArrayBase<Vec<f64>, (usize, usize)> {
    let mut m = Array::eye(base.shape()[0]);
    for _ in 0 .. power {
        m = m.dot(base);
    }
    m
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

    let f = |n| state0.dot(&arr2_powi(&transition_matrix, n).dot(&state0));

    let h = {
        use rand::Rng;
        use rand::distributions::{IndependentSample, Range};

        let iters = 2000000;
        let mut state = vec![0; 21];
        let mut weight = transition_matrix[(0, 0)].powi(20);
        let mut s_20 = 0.0;
        let mut s_18 = 0.0;
        let range = Range::new(0, 3);
        for _ in 0 .. iters {
            for i in 1 .. state.len() - 1 {
                let x_new = range.ind_sample(&mut rng);
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
                s_18 += 1.0;
            }
            s_20 += 1.0;
        }
        let c = state0.dot(&arr2_powi(&transition_matrix, 2).dot(&state0));
        s_20 / s_18 * c
    };

    println!("{{");
    println!("'expression': 'f(20) / f(18)',");
    println!("'exact_result': {},", f(20) / f(18));
    println!("'mc_result': {:?},", h);
    println!("}}");

}
