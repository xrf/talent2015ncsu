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
  qmc [options]

Options:

  -h --help
    Show this screen.

  -s --seed=<seed>
    Seed for the random number generator in LE hexadecimal.
    [default is empty string]

  --num-steps=<num-steps>
    [default: 10000]

  --alpha=<alpha>
    [default: 1.4]

  --omega=<omega>
    [default: 1.0]

  --energy=<energy>
    [default: 0.6]

  --dt=<dt>
    [default: 0.001]

  --branch-interval=<branch-interval>
    [default: 10]

  --print-interval=<print-interval>
    [default: 1000]

  --num-steps=<num-steps>
    [default: 10000]

  --population=<population>
    [default: 10000]

";

#[derive(Debug, RustcDecodable)]
struct Args {
    flag_seed: RandomSeed,
    flag_num_steps: u64,
    flag_alpha: f64,
    flag_omega: f64,
    flag_dt: f64,
    flag_energy: f64,
    flag_print_interval: u64,
    flag_branch_interval: u64,
    flag_population: u64,
}

fn main() {
    let args: Args = docopt::Docopt::new(USAGE)
        .and_then(|d| d.decode())
        .unwrap_or_else(|e| e.exit());

    println!("# args: {:?}", args);

    let mut rng = rng_from_seed(&args.flag_seed.value);

    let num_steps = args.flag_num_steps;
    let alpha = args.flag_alpha;
    let omega = args.flag_omega;
    let dt = args.flag_dt;
    let print_interval = args.flag_print_interval;
    let branch_interval = args.flag_branch_interval;
    let sqrt_dt = dt.sqrt(); // controls the rate of diffusion
    let population_0 = args.flag_population as usize;
    let mut population = population_0;
    let mut e = args.flag_energy;

    let mut wprod_tmp = Vec::with_capacity(population * 2);
    let mut wprod = vec![1.0; population * 2];

    let mut x_tmp = Vec::with_capacity(population * 2);
    let mut x = vec![0.0; population * 2];
    for i in 0 .. population {
        let StandardNormal(r) = rng.gen();
        ix_mut!(x, i) = r / alpha.sqrt();
    }

    let mut w = Vec::with_capacity(population * 2);

    let mut old_x = &mut x;
    let mut new_x = &mut x_tmp;
    let mut old_wprod = &mut wprod;
    let mut new_wprod = &mut wprod_tmp;

    println!("[");
    for step_index in 0 .. num_steps {

        // update walkers
        w.resize(population, 0.0);
        for i in 0 .. population {
            // diffuse the position (this is the kinetic energy part)
            let StandardNormal(r) = rng.gen();
            ix_mut!(old_x, i) += r * sqrt_dt;

            // update the weight (this is the potential energy part)
            let v = 0.5 * omega.powi(2) * ix!(old_x, i).powi(2);
            ix_mut!(w, i) = (-(v - e) * dt).exp();
            ix_mut!(old_wprod, i) *= ix!(w, i);
        }

        // perform branching
        if step_index % branch_interval == 0 {
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

        // adjust energy shift
        e += if population <= population_0 { 0.01 } else { -0.01 };

        // compute statistics
        let mut e_numerator = 0.0;
        let mut denominator = 0.0;
        for i in 0 .. population {
            let u = (-0.5 * alpha * ix!(old_x, i).powi(2)).exp();
            e_numerator +=
                0.5 * ix!(old_wprod, i) * u * (
                    alpha + ix!(old_x, i).powi(2) * (omega.powi(2) - alpha.powi(2)));
            denominator += ix!(old_wprod, i) * u;
        }
        let energy = e_numerator / denominator;

        if step_index % print_interval == 0 {
            println!("{{");
            println!(" 'step_index': {},", step_index);
            println!(" 'time': {},", step_index as f64 * dt);
            println!(" 'average_energy': {},", energy);
            println!(" 'ref_energy': {},", e);
            println!(" 'population': {},", population);
            println!("}},");
        }
    }
    println!("]");

}
