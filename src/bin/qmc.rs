extern crate docopt;
extern crate rand;
extern crate rustc_serialize;
#[macro_use]
extern crate qmc;

use rand::Rng;
use rand::distributions::{Sample, IndependentSample};
use rand::distributions::normal::StandardNormal;
use qmc::utils::*;

pub trait WaveFunction {

    /// Calculate `psi(x)`.
    fn eval(&self, x: f64) -> f64;

    /// Calculate `(T psi)(x) / psi(x)`.
    fn local_kinetic_energy(&self, x: f64) -> f64;

    /// Calculate `(V psi)(x) / psi(x)`.
    fn local_potential_energy(&self, x: f64) -> f64;

    /// Calculate `(H psi)(x) / psi(x)`.
    fn local_energy(&self, x: f64) -> f64;

    /// Calculate `(del psi)(x) / psi(x)`.
    fn local_gradient(&self, x: f64) -> f64;

}

pub struct HO1D {
    omega: f64,
}

// TODO: maybe we should separate the wavefunction dependent part and
// the wavefunction+hamiltonian dependent parts?
pub struct HO1DTrial<'a> {
    system: &'a HO1D,
    alpha: f64,
}

impl HO1D {

    /// Calculate `V(x)`.
    fn potential_energy(&self, x: f64) -> f64 {
        0.5 * self.omega.powi(2) * x.powi(2)
    }

}

impl <'a> WaveFunction for HO1DTrial<'a> {

    fn eval(&self, x: f64) -> f64 {
        (-0.5 * self.alpha * x.powi(2)).exp()
    }

    fn local_kinetic_energy(&self, x: f64) -> f64 {
        -0.5 * ((self.alpha * x).powi(2) - self.alpha)
    }

    fn local_potential_energy(&self, x: f64) -> f64 {
        self.system.potential_energy(x)
    }

    fn local_energy(&self, x: f64) -> f64 {
        self.local_kinetic_energy(x) + self.local_potential_energy(x)
    }

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
                           source: &DMCState) {
        for i in 0 .. source.len() {
            let num_clones =
                (ix!(source.weight_products, i) + rng.next_f64()) as i64;
            for _ in 0 .. num_clones {
                self.x_positions.push(ix!(source.x_positions, i));
                self.weight_products.push(1.0);
            }
        }
    }

}

pub trait Strategy {
    fn diffusion_offset<F>(&self, trial: &F, x: f64, dt: f64) -> f64
        where F: WaveFunction;
    fn potential_energy<F>(&self, trial: &F, x: f64) -> f64
        where F: WaveFunction;
    fn weight_coeff<F>(&self, trial: &F, x: f64) -> f64
        where F: WaveFunction;
}

pub struct NormalStrategy;

impl Strategy for NormalStrategy {
    fn diffusion_offset<F>(&self, _: &F, _: f64, _: f64) -> f64
        where F: WaveFunction {
        0.0
    }
    fn potential_energy<F>(&self, trial: &F, x: f64) -> f64
        where F: WaveFunction {
        trial.local_potential_energy(x)
    }
    fn weight_coeff<F>(&self, trial: &F, x: f64) -> f64
        where F: WaveFunction {
        trial.eval(x)
    }
}

pub struct ImportanceSamplingStrategy;

impl Strategy for ImportanceSamplingStrategy {
    fn diffusion_offset<F>(&self, trial: &F, x: f64, dt: f64) -> f64
        where F: WaveFunction {
        trial.local_gradient(x) * dt
    }
    fn potential_energy<F>(&self, trial: &F, x: f64) -> f64
        where F: WaveFunction {
        trial.local_energy(x)
    }
    fn weight_coeff<F>(&self, _: &F, _: f64) -> f64
        where F: WaveFunction {
        1.0
    }
}

struct DMC<Strat> {
    state: DMCState,
    new_state: DMCState,
    energy: f64,
    population_goal: usize,
    strategy: Strat,
}

impl <S: Strategy> DMC<S> {

    fn new<R, D>(rng: &mut R,
                 distribution: &D,
                 population: usize,
                 energy: f64,
                 strategy: S) -> Self
        where R: Rng,
              D: IndependentSample<f64> {
        let initial_capacity = (population as f64 * 1.4) as usize;
        DMC {
            state: DMCState::new(rng, distribution, population,
                                 initial_capacity),
            new_state: DMCState::with_capacity(initial_capacity),
            energy: energy,
            population_goal: population,
            strategy: strategy,
        }
    }

    fn population(&self) -> usize {
        self.state.len()
    }

    /// `time_step` is expected to be in units of `<length_unit>^2 m / hbar`.
    fn step<R, F>(&mut self,
                  rng: &mut R,
                  trial_wavfun: &F,
                  time_step: f64)
        where R: Rng,
              F: WaveFunction {

        let population = self.state.len();

        // update walkers
        for i in 0 .. population {
            let state = &mut self.state;

            // diffuse the position (this is the kinetic energy part)
            let StandardNormal(r) = rng.gen();
            let x = ix!(state.x_positions, i);
            let new_x =
                x
                + self.strategy.diffusion_offset(
                    trial_wavfun, x, time_step)
                + r * time_step.sqrt();
            ix_mut!(state.x_positions, i) = new_x;

            // update the weight (this is the potential energy part)
            let v = self.strategy.potential_energy(trial_wavfun, new_x);
            let w = (-time_step * (v - self.energy)).exp();
            ix_mut!(state.weight_products, i) *= w;
        }

        self.control_population(0.01);
    }

    fn branch<R: Rng>(&mut self, rng: &mut R) {
        assert!(self.state.valid());
        assert!(self.new_state.len() == 0);

        self.control_population(-0.01);
        self.new_state.branch_from(rng, &self.state);
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

    fn stats<F, G>(&self, trial_wavfun: &F, mut f: G) -> f64
        where F: WaveFunction,
              G: FnMut(f64, f64) {
        let mut denominator = 0.0;
        for i in 0 .. self.state.len() {
            let x = ix!(self.state.x_positions, i);
            let u =
                ix!(self.state.weight_products, i)
                * self.strategy.weight_coeff(trial_wavfun, x);
            f(u, x);
            denominator += u;
        }
        denominator
    }

    fn all_stats<F: WaveFunction>(&self, trial_wavfun: &F) -> Stats {
        let mut stats = Stats::new();
        let denom = self.stats(trial_wavfun, |u, x| {
            let e = trial_wavfun.local_energy(x);
            stats.energy += u * e;
            stats.sq_energy += u * e.powi(2);
        });
        &stats / denom
    }

}

struct Stats {
    energy: f64,
    sq_energy: f64,
}

impl Stats {

    fn new() -> Stats {
        Stats {
            energy: 0.0,
            sq_energy: 0.0,
        }
    }

    fn print(&self) {
        println!(" 'avg_energy': {},", self.energy);
        println!(" 'avg_sq_energy': {},", self.sq_energy);
    }

}

impl <'a> std::ops::Div<f64> for &'a Stats  {
    type Output = Stats;
    fn div(self, rhs: f64) -> Self::Output {
        Stats {
            energy: self.energy / rhs,
            sq_energy: self.sq_energy / rhs,
        }
    }
}

const INITIAL_DIFFUSE: bool = false;

fn dmc<R, S, F>(rng: &mut R,
                num_steps: u64,
                initial_population: usize,
                time_step: f64,
                energy: f64,
                trial_wavfun: &F,
                print_interval: u64,
                branch_interval: u64,
                strategy: S)
    where R: Rng,
          S: Strategy,
          F: IndependentSample<f64> + WaveFunction {
    let branches_per_print = print_interval / branch_interval;
    let num_prints = num_steps / (branches_per_print * branch_interval);

    let mut dmc = DMC::new(rng, trial_wavfun,
                           initial_population, energy, strategy);
    println!("[");

    if INITIAL_DIFFUSE {
        dmc.step(rng, trial_wavfun, time_step);
        dmc.branch(rng);
    }

    let stats = dmc.all_stats(trial_wavfun);
    println!("{{");
    println!(" 'step_index': {},", 0);
    println!(" 'time': {},", 0.0);
    stats.print();
    println!(" 'ref_energy': {},", dmc.energy);
    println!(" 'population': {},", dmc.population());
    println!("}},");

    for i in 0 .. num_prints {

        for _ in 0 .. branches_per_print {
            for _ in 0 .. branch_interval {
                dmc.step(rng, trial_wavfun, time_step);
            }
            dmc.branch(rng);
        }

        let step_index = (i + 1) * branches_per_print * branch_interval;
        let stats = dmc.all_stats(trial_wavfun);
        println!("{{");
        println!(" 'step_index': {},", step_index);
        println!(" 'time': {},", step_index as f64 * time_step);
        stats.print();
        println!(" 'ref_energy': {},", dmc.energy);
        println!(" 'population': {},", dmc.population());
        println!("}},");

    }
    println!("]");
}

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
        args.flag_branch_interval,
        NormalStrategy);

}
