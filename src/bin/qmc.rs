extern crate docopt;
extern crate rand;
extern crate rustc_serialize;
extern crate qmc;

use qmc::mt;
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

    let seed = [0x3f, 0xc9, 0x07]; // 510271;
    let num_trials = 100000; // numtrials
    let alpha: f64 = 1.4;
    let omega: f64 = 1.0;
    let dt: f64 = 0.001;
    let print_interval = 1000; // nprintevery
    let branch_interval = 10; // nbranchevery
    let population_0 = 100000;

    mt::seed(&seed);

    let sigma = dt.sqrt();

    let mut e = 0.6; // E

    let mut wprod_tmp = Vec::new();
    let mut wprod = vec![1.0; population_0];

    let mut x_tmp = Vec::new();
    let mut x = Vec::with_capacity(population_0);
    for _ in 0 .. x.capacity() {
        x.push(mt::gaussrnd() / alpha.sqrt());
    }

    let mut population = population_0; // numwalkm

    let mut w = Vec::new();

    let mut old_x = &mut x;
    let mut new_x = &mut x_tmp;
    let mut old_wprod = &mut wprod;
    let mut new_wprod = &mut wprod_tmp;

    for trial_index in 0 .. num_trials {

        w.resize(population, 0.0);
        for i in 0 .. population {
            let r = mt::gaussrnd();
            old_x[i] += r * sigma;

            let v = 0.5 * omega.powi(2) * old_x[i].powi(2);
            w[i] = (-(v - e) * dt).exp();
            old_wprod[i] *= w[i];
        }

        if trial_index % branch_interval == 0 {
            for i in 0 .. population {
                let num_clones = (w[i] + mt::grnd()) as i64;
                for _ in 0 .. num_clones {
                    new_x.push(old_x[i]);
                    new_wprod.push(old_wprod[i]);
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
            let u = (-0.5 * alpha * old_x[i].powi(2)).exp();
            e_numerator +=
                0.5 * old_wprod[i] * u * (
                    alpha + old_x[i].powi(2) * (omega.powi(2) - alpha.powi(2)));
            denominator += old_wprod[i] * u;
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
