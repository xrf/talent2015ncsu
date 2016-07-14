extern crate docopt;
extern crate rand;
extern crate rustc_serialize;
#[macro_use]
extern crate qmc;

use rand::Rng;
use rand::distributions::normal::StandardNormal;
use qmc::utils::*;

const USAGE: &'static str = "
Usage:
  qmc [--seed <seed>]
  qmc --help

Options:
  -h --help         Show this screen.
  -s --seed=<seed>  Seed for the random number generator in base 64
                    [default is empty string].
";

#[derive(Debug, RustcDecodable)]
struct Args {
    flag_seed: RandomSeed,
}

fn main() {
    let args: Args = docopt::Docopt::new(USAGE)
        .and_then(|d| d.decode())
        .unwrap_or_else(|e| e.exit());

    let mut rng = rng_from_seed(&args.flag_seed.value);

    println!("using seed: {:?}", &args.flag_seed.value);

unsafe {

    let num_trials = 1000; // numtrials
    let alpha: f64 = 1.4;
    let omega: f64 = 1.0;
    let dt: f64 = 0.001;
    let print_interval = 100; // nprintevery
    let branch_interval = 10; // nbranchevery
    let population_0 = 100000;

    let sigma = dt.sqrt();

    let mut e = 0.6; // E

    let mut wprod_tmp = Vec::with_capacity(population_0 * 2);
    let mut wprod = vec![1.0; population_0 * 2];

    let mut x_tmp = Vec::with_capacity(population_0 * 2);
    let mut x = vec![0.0; population_0 * 2];
    for i in 0 .. population_0 {
        let StandardNormal(r) = rng.gen();
        ix!(x, i) = r / alpha.sqrt();
    }

    let mut population = population_0; // numwalkm

    let mut w = Vec::with_capacity(population_0 * 2);

    let mut old_x = &mut x;
    let mut new_x = &mut x_tmp;
    let mut old_wprod = &mut wprod;
    let mut new_wprod = &mut wprod_tmp;

    for trial_index in 0 .. num_trials {

        w.resize(population, 0.0);
        for i in 0 .. population {
            let StandardNormal(r) = rng.gen();
            ix!(old_x, i) += r * sigma;

            let v = 0.5 * omega.powi(2) * ix!(old_x, i).powi(2);
            ix!(w, i) = (-(v - e) * dt).exp();
            ix!(old_wprod, i) *= ix!(w, i);
        }

        if trial_index % branch_interval == 0 {
            for i in 0 .. population {
                let num_clones = (ix!(w, i) + rng.next_f64()) as i64;
                for _ in 0 .. num_clones {
                    new_x.push(ix!(old_x, i));
                    new_wprod.push(ix!(old_wprod, i));
                }
            }
            old_x.clear();
            old_wprod.clear();
            std::mem::swap(&mut old_x, &mut new_x);
            std::mem::swap(&mut old_wprod, &mut new_wprod);
            population = old_x.len();
        }

        let mut e_numerator = 0.0;
        let mut denominator = 0.0;
        for i in 0 .. population {
            let u = (-0.5 * alpha * ix!(old_x, i).powi(2)).exp();
            e_numerator +=
                0.5 * ix!(old_wprod, i) * u * (
                    alpha + ix!(old_x, i).powi(2) * (omega.powi(2) - alpha.powi(2)));
            denominator += ix!(old_wprod, i) * u;
        }
        e += if denominator < 1.0 { 0.01 } else { -0.01 };
        let energy = e_numerator / denominator;
        if trial_index % print_interval == 0 {
            println!("trial_index = {}", trial_index);
            println!("time = {}", trial_index as f64 * dt);
            println!("average energy = {}", energy);
            println!("population = {}", population);
            println!("");
        }
    }

}

}
