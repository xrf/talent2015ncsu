extern crate docopt;
extern crate rand;
extern crate rustc_serialize;
#[macro_use]
extern crate qmc;

use qmc::utils::*;
use qmc::qmc::*;

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

  --trial-energy=<trial-energy>
    Initial trial energy.
    [default: 0.6]

  --overcorrection=<overcorrection>
    Affects how much the trial energy should be overcorrected with respect to
    the growth energy.
    [default: 0.25]

  --dt=<dt>
    [default: 0.001]

  --branch-interval=<branch-interval>
    [default: 10]

  --bin-interval=<branch-interval>
    Binning interval for VMC.
    [default: 10]

  --print-interval=<print-interval>
    Number of steps between prints.
    [default: 10]

  --vmc-steps=<vmc-num-steps>
    Number of VMC steps to take.  Can be zero.
    [default: 10000]

  --max-step-size=<max-step-size>
    Largest possible distance a particle can move in a single VMC step.
    [default: 0.2]

  --population=<population>
    Desired walker population.  Actual population may vary slightly.
    [default: 10000]

";

#[derive(Debug, RustcDecodable)]
struct Args {
    flag_seed: String,
    flag_num_steps: u64,
    flag_alpha: f64,
    flag_omega: f64,
    flag_trial_energy: f64,
    flag_overcorrection: f64,
    flag_dt: f64,
    flag_branch_interval: u64,
    flag_bin_interval: u64,
    flag_print_interval: u64,
    flag_vmc_steps: u64,
    flag_max_step_size: f64,
    flag_population: usize,
}

fn main() {
    let args: Args = docopt::Docopt::new(USAGE)
        .and_then(|d| d.decode())
        .unwrap_or_else(|e| e.exit());
    let seed = decode_seed(&args.flag_seed);

    println!("# args: {:?}", args);

    let mut rng = rng_from_seed(&seed);
    let sys = HO1D { omega: args.flag_omega };
    let trial_wavfun = HO1DTrial { system: &sys, alpha: args.flag_alpha };
    let distr = rand::distributions::range::Range::new(-0.5, 0.5);

    println!("{{");

    println!("'vmc':");
    let state = vmc(&mut rng,
                    args.flag_vmc_steps,
                    args.flag_bin_interval,
                    args.flag_population,
                    &trial_wavfun,
                    &distr,
                    args.flag_max_step_size).x_positions;

    println!("'dmc':");
    dmc(&mut rng,
        args.flag_num_steps,
        &state,
        args.flag_dt,
        args.flag_trial_energy,
        &trial_wavfun,
        args.flag_overcorrection,
        args.flag_print_interval,
        args.flag_branch_interval,
        NormalStrategy);

    println!("}}");
}
