extern crate docopt;
extern crate rand;
extern crate rustc_serialize;
#[macro_use]
extern crate qmc;

use rand::Rng;
use rand::distributions::{Sample, IndependentSample};
use rand::distributions::normal::StandardNormal;
use qmc::utils::*;

const USAGE: &'static str = "
Usage:
  qmc2 [options]

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
    flag_population: usize,
}

struct HO1D {
    omega: f64,
}

// TODO: maybe we should separate the wavefunction dependent part and
// the wavefunction+hamiltonian dependent parts?
struct HO1DTrial<'a> {
    system: &'a HO1D,
    alpha: f64,
}

impl HO1D {

    /// Calculate `V(x)`.
    fn potential_energy(&self, x: f64) -> f64 {
        0.5 * self.omega.powi(2) * x.powi(2)
    }

}

impl <'a> HO1DTrial<'a> {

    /// Calculate `psi(x)`.
    fn eval(&self, x: f64) -> f64 {
        (-0.5 * self.alpha * x.powi(2)).exp()
    }

    /// Calculate `(T psi)(x) / psi(x)`.
    fn local_kinetic_energy(&self, x: f64) -> f64 {
        -0.5 * ((self.alpha * x).powi(2) - self.alpha)
    }

    /// Calculate `(V psi)(x) / psi(x)`.
    fn local_potential_energy(&self, x: f64) -> f64 {
        self.system.potential_energy(x)
    }

    /// Calculate `(H psi)(x) / psi(x)`.
    fn local_energy(&self, x: f64) -> f64 {
        self.local_kinetic_energy(x) + self.local_potential_energy(x)
    }

    /// Calculate `(del psi)(x) / psi(x)`.
    fn local_gradient(&self, x: f64) -> f64 {
        -self.alpha * x
    }

}

impl<'a> Sample<f64> for HO1DTrial<'a> {

    fn sample<R: Rng>(&mut self, rng: &mut R) -> f64 {
        self.ind_sample(rng)
    }

}

impl<'a> IndependentSample<f64> for HO1DTrial<'a> {

    fn ind_sample<R: Rng>(&self, rng: &mut R) -> f64 {
        let StandardNormal(r) = rng.gen();
        r / self.alpha.sqrt()
    }

}

struct DMCState {
    /// Weight product of the walkers
    weight_products: Vec<f64>,
    /// Position of the walkers
    x_positions: Vec<f64>,
}

impl DMCState {

    fn new<R, D>(rng: &mut R,
                 distribution: &D,
                 population: usize,
                 initial_capacity: usize) -> DMCState
        where R: Rng,
              D: IndependentSample<f64> {
        let mut state = DMCState::with_capacity(initial_capacity);
        for _ in 0 .. population {
            state.x_positions.push(distribution.ind_sample(rng));
            state.weight_products.push(1.0);
        }
        state
    }

    fn with_capacity(capacity: usize) -> DMCState {
        DMCState {
            x_positions: Vec::with_capacity(capacity),
            weight_products: Vec::with_capacity(capacity),
        }
    }

    fn valid(&self) -> bool{
        self.weight_products.len() == self.x_positions.len()
    }

    fn len(&self) -> usize {
        debug_assert!(self.valid());
        self.weight_products.len()
    }

    fn clear(&mut self) {
        self.x_positions.clear();
        self.weight_products.clear();
    }

    fn branch_from<R: Rng>(&mut self,
                           rng: &mut R,
                           source: &DMCState,
                           weights: &[f64]) {
        for i in 0 .. source.len() {
            let num_clones = (ix!(weights, i) + rng.next_f64()) as i64;
            for _ in 0 .. num_clones {
                self.x_positions.push(ix!(source.x_positions, i));
                self.weight_products.push(ix!(source.weight_products, i));
            }
        }
    }

}

struct DMC {
    state: DMCState,
    new_state: DMCState,
    /// Current weights being calculated
    weights: Vec<f64>,
    time: f64,
    energy: f64,
    population_goal: usize,
}

impl DMC {

    fn new<R: Rng, D: IndependentSample<f64>>(rng: &mut R,
                                              distribution: &D,
                                              population: usize,
                                              energy: f64) -> DMC {
        let initial_capacity = (population as f64 * 1.4) as usize;
        DMC {
            state: DMCState::new(rng, distribution, population,
                                 initial_capacity),
            new_state: DMCState::with_capacity(initial_capacity),
            weights: Vec::with_capacity(initial_capacity),
            time: 0.0,
            energy: energy,
            population_goal: population,
        }
    }

    fn population(&self) -> usize {
        self.state.len()
    }

    fn diffuse<R: Rng>(&mut self,
                       rng: &mut R,
                       system: &HO1D,
                       num_steps: u64,
                       time_step: f64) {
        let sqrt_time_step = time_step.sqrt();
        let population = self.state.len();

        assert!(self.state.valid());

        for _ in 0 .. num_steps {

            // update walkers
            self.weights.resize(population, 0.0);
            for i in 0 .. population {
                let state = &mut self.state;

                // diffuse the position (this is the kinetic energy part)
                let StandardNormal(r) = rng.gen();
                let new_x = ix!(state.x_positions, i) + r * sqrt_time_step;
                ix_mut!(state.x_positions, i) = new_x;

                // update the weight (this is the potential energy part)
                let w = (-time_step * (system.potential_energy(new_x)
                                       - self.energy)).exp();
                ix_mut!(self.weights, i) = w;
                ix_mut!(state.weight_products, i) *= w;
            }

            self.control_population(0.01);
        }
        self.time += num_steps as f64 * time_step;
    }

    fn branch<R: Rng>(&mut self, rng: &mut R) {
        assert!(self.state.valid());
        assert!(self.state.len() == self.weights.len());
        assert!(self.new_state.len() == 0);

        self.control_population(-0.01);
        self.new_state.branch_from(rng, &self.state, &self.weights);
        self.state.clear();
        std::mem::swap(&mut self.state, &mut self.new_state);
        self.control_population(0.01);
    }

    fn control_population(&mut self, increment: f64) {
        self.energy +=
            if self.population() <= self.population_goal {
                increment
            } else {
                -increment
            };
    }

    fn stats(&self, trial_wavfun: &HO1DTrial) -> f64 {
        let mut e_numerator = 0.0;
        let mut denominator = 0.0;
        for i in 0 .. self.state.len() {
            let x = ix!(self.state.x_positions, i);
            let u = ix!(self.state.weight_products, i) * trial_wavfun.eval(x);
            e_numerator += u * trial_wavfun.local_energy(x);
            denominator += u;
        }
        e_numerator / denominator
    }

}

const INITIAL_DIFFUSE: bool = false;

fn dmc<R: Rng>(rng: &mut R,
               num_steps: u64,
               initial_population: usize,
               dt: f64,
               energy: f64,
               trial_wavfun: &HO1DTrial,
               print_interval: u64,
               branch_interval: u64) {
    let sys = trial_wavfun.system;
    let branches_per_print = print_interval / branch_interval;
    let num_prints = num_steps / (branches_per_print * branch_interval);

    let mut dmc = DMC::new(rng, trial_wavfun, initial_population, energy);
    println!("[");

    if INITIAL_DIFFUSE {
        dmc.diffuse(rng, sys, 1, dt);
        dmc.branch(rng);

        let average_energy = dmc.stats(trial_wavfun);
        println!("{{");
        println!(" 'step_index': {},", 0);
        println!(" 'time': {},", dmc.time - dt);
        println!(" 'average_energy': {},", average_energy);
        println!(" 'ref_energy': {},", dmc.energy);
        println!(" 'population': {},", dmc.population());
        println!("}},");
    }

    for i in 0 .. num_prints {

        for _ in 0 .. branches_per_print {
            dmc.diffuse(rng, sys, branch_interval, dt);
            dmc.branch(rng);
        }

        let average_energy = dmc.stats(trial_wavfun);
        println!("{{");
        println!(" 'step_index': {},",
                 (i + 1) * branches_per_print * branch_interval);
        println!(" 'time': {},", dmc.time - dt);
        println!(" 'average_energy': {},", average_energy);
        println!(" 'ref_energy': {},", dmc.energy);
        println!(" 'population': {},", dmc.population());
        println!("}},");

    }
    println!("]");
}

fn main() {
    let args: Args = docopt::Docopt::new(USAGE)
        .and_then(|d| d.decode())
        .unwrap_or_else(|e| e.exit());

    println!("# args: {:?}", args);

    let mut rng = rng_from_seed(&args.flag_seed.value);
    let sys = HO1D { omega: args.flag_omega };
    let trial_wavfun = HO1DTrial { system: &sys, alpha: args.flag_alpha };

//     {
//         let delta = 0.1;
//         // vmc
//         let mut walkers = VMCState {
//             x: vec![0.0; population],
//             w: vec![0.0; population],
//         };
//         let mut new_walkers = VMCState {
//             x: vec![0.0; population],
//             w: vec![0.0; population],
//         };
//         for i in 0 .. population {
//             let StandardNormal(r) = rng.gen();
//             let x = r / trial_wavefun.alpha.sqrt();
//             ix!(walkers.x, i) = x;
//             ix!(walkers.w, i) = trial_wavfun.eval(alpha, x).powi(2);
// //            println!("{}", walkers.x[i]);
//         }
//         for n in 0 .. num_steps {
//             for i in 0 .. population {
//                 let new_x = ix!(walkers.x, i) + delta * (rng.next_f64() - 0.5);
//                 let new_w = trial_wavfun.eval(alpha, new_x).powi(2);
//                 if rng.next_f64() * ix!(walkers.w, i) < new_w {
//                     ix!(walkers.x, i) = new_x;
//                     ix!(walkers.w, i) = new_w;
//                 }
//             }

//             let mut energy_sum = 0.0;
//             for i in 0 .. population {
//                 -0.5 * ((alpha * x).powi(2) - alpha)
//                 0.5 * sys.omega.powi(2)
//             }

//         }
//         std::process::exit(0);
//     }

    dmc(&mut rng,
        args.flag_num_steps,
        args.flag_population,
        args.flag_dt,
        args.flag_energy,
        &trial_wavfun,
        args.flag_print_interval,
        args.flag_branch_interval);

}
