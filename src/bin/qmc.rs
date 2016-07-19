extern crate docopt;
extern crate rand;
extern crate rustc_serialize;
#[macro_use]
extern crate qmc;

use rand::Rng;
use rand::distributions::{Sample, IndependentSample};
use rand::distributions::normal::StandardNormal;
use qmc::utils::*;

pub fn stdev(mean: f64, sum_squares: f64, total_weight: f64) -> f64 {
    (sum_squares / total_weight - mean.powi(2)).sqrt()
}

#[derive(RustcEncodable)]
pub struct Stats {
    weight: f64,
    energy: f64,
    sq_energy: f64,
    x: f64,
    sq_x: f64,
}

impl Default for Stats {
    fn default() -> Self {
        Stats {
            weight: 0.0,
            energy: 0.0,
            sq_energy: 0.0,
            x: 0.0,
            sq_x: 0.0,
        }
    }
}

impl Stats {
    fn update<W: WaveFunction>(&mut self, trial_wavfun: &W, w: f64, x: f64) {
        let e = trial_wavfun.local_energy(x);
        self.weight += w;
        self.energy += w * e;
        self.sq_energy += w * e.powi(2);
        self.x += w * x;
        self.sq_x += w * x.powi(2);
    }

    fn calc_average(&mut self) {
        self.energy /= self.weight;
        self.sq_energy /= self.weight;
        self.x /= self.weight;
        self.sq_x /= self.weight;
    }
}

pub trait WaveFunction {

    /// Calculate `psi(x)`.
    fn eval(&self, x: f64) -> f64;

    /// Calculate `|psi(x)|^2`.
    fn normsq_eval(&self, x: f64) -> f64;

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

    fn normsq_eval(&self, x: f64) -> f64 {
        self.eval(x).powi(2)
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

struct QMCState {
    /// Weight product of the walkers
    weights: Vec<f64>,
    /// Position of the walkers
    x_positions: Vec<f64>,
}

impl QMCState {

    pub fn new<R, D>(rng: &mut R,
                     distribution: &D,
                     population: usize) -> QMCState
        where R: Rng,
              D: IndependentSample<f64> {
        QMCState::new_with_capacity(rng, distribution, population, population)
    }

    pub fn new_with_capacity<R, D>(rng: &mut R,
                                   distribution: &D,
                                   population: usize,
                                   initial_capacity: usize) -> QMCState
        where R: Rng,
              D: IndependentSample<f64> {
        let mut state = QMCState::with_capacity(initial_capacity);
        for _ in 0 .. population {
            state.x_positions.push(distribution.ind_sample(rng));
            state.weights.push(1.0);
        }
        state
    }

    pub fn with_capacity(capacity: usize) -> QMCState {
        QMCState {
            x_positions: Vec::with_capacity(capacity),
            weights: Vec::with_capacity(capacity),
        }
    }

    pub fn valid(&self) -> bool {
        self.weights.len() == self.x_positions.len()
    }

    pub fn len(&self) -> usize {
        debug_assert!(self.valid());
        self.weights.len()
    }

    pub fn clear(&mut self) {
        self.x_positions.clear();
        self.weights.clear();
    }

    pub fn branch_from<R: Rng>(&mut self,
                               rng: &mut R,
                               source: &QMCState) {
        for i in 0 .. source.len() {
            let num_clones =
                (ix!(source.weights, i) + rng.next_f64()) as i64;
            for _ in 0 .. num_clones {
                self.x_positions.push(ix!(source.x_positions, i));
                self.weights.push(1.0);
            }
        }
    }

    pub fn metropolis<R, P>(&mut self,
                            rng: &mut R,
                            probability: P,
                            max_step_size: f64)
        where R: Rng,
              P: Fn(f64) -> f64 {
        for i in 0 .. self.len() {
            let x = ix!(self.x_positions, i);
            let p = ix!(self.weights, i);
            let x_new = x + max_step_size * (rng.next_f64() - 0.5);
            let p_new = probability(x_new);
            if rng.next_f64() * p < p_new {
                ix_mut!(self.x_positions, i) = x_new;
                ix_mut!(self.weights, i) = p_new;
            }
        }
    }

    pub fn stats<F>(&self, mut update_stats: F)
        where F: FnMut(f64, f64) {
        for i in 0 .. self.len() {
            let x = ix!(self.x_positions, i);
            let u = ix!(self.weights, i);
            update_stats(u, x);
        }
    }

}

pub trait Strategy {
    fn diffusion_offset<F>(&self, trial: &F, x: f64, dt: f64) -> f64
        where F: WaveFunction;

    /// Despite the name, this part isn't necessarily the potential energy.
    /// This term controls the weight calculation.
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

fn vmc<R, W, D>(rng: &mut R,
                num_steps: u64,
                steps_per_bin: u64,
                population: usize,
                trial_wavfun: &W,
                initial_distribution: &D,
                max_step_size: f64) -> QMCState
    where R: Rng,
          W: WaveFunction,
          D: IndependentSample<f64> {
    let num_bins = num_steps / steps_per_bin;
    let mut vmc = QMCState::new(rng, initial_distribution, population);
    println!("[");
    for _ in 0 .. num_bins {
        let mut stats: Stats = Default::default();
        for _ in 0 .. steps_per_bin {
            vmc.metropolis(rng,
                           |x| trial_wavfun.normsq_eval(x),
                           max_step_size);
            vmc.stats(|u, x| stats.update(trial_wavfun, u, x));
        }
        stats.calc_average();
        println!("{},", rustc_serialize::json::encode(&stats).unwrap());
    }
    println!("],");
    vmc
}

struct DMC<Strat> {
    state: QMCState,
    new_state: QMCState,
    trial_energy: f64,
    overcorrection: f64,
    // Note that we store the growth_energy directly instead of total weight
    // or average weight because the population / trial_energy might've
    // changed since then by branching
    growth_energy: f64,
    weight_product_sum: f64,
    population_goal: usize,
    strategy: Strat,
}

impl <S: Strategy> DMC<S> {

    fn new(state: &[f64],
           trial_energy: f64,
           overcorrection: f64,
           strategy: S) -> Self {
        let population = state.len();
        let initial_capacity = (population as f64 * 1.4) as usize;
        let mut qmc_state = QMCState::with_capacity(initial_capacity);
        for x in state.iter() {
            qmc_state.x_positions.push(*x);
            qmc_state.weights.push(1.0);
        }
        DMC {
            state: qmc_state,
            new_state: QMCState::with_capacity(initial_capacity),
            trial_energy: trial_energy,
            overcorrection: overcorrection,
            growth_energy: trial_energy,
            population_goal: population,
            weight_product_sum: population as f64,
            strategy: strategy,
        }
    }

    fn population(&self) -> usize {
        self.state.len()
    }

    /// `time_step` is expected to be in units of `<length_unit>^2 m / hbar`.
    fn step<R, W>(&mut self,
                  rng: &mut R,
                  trial_wavfun: &W,
                  time_step: f64)
        where R: Rng,
              W: WaveFunction {

        let population = self.population();

        // update walkers
        let mut weight_sum = 0.0;
        let mut weight_product_sum = 0.0;
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
            let w = (time_step * (self.trial_energy - v)).exp();
            ix_mut!(state.weights, i) *= w;
            weight_sum += w;
            weight_product_sum += w;
        }

        let avg_weight = weight_sum / self.population() as f64;
        self.growth_energy = self.trial_energy - avg_weight.ln() / time_step;
        self.weight_product_sum = weight_product_sum;
        self.control_population();
    }

    fn branch<R: Rng>(&mut self, rng: &mut R) {
        assert!(self.state.valid());
        assert!(self.new_state.len() == 0);

        self.new_state.branch_from(rng, &self.state);
        self.state.clear();
        std::mem::swap(&mut self.state, &mut self.new_state);
    }

    fn control_population(&mut self) {
        let sign =
            if self.weight_product_sum < self.population_goal as f64 {
                1.0
            } else {
                -1.0
            };
        let growth_correction =
            (1.0 + self.overcorrection)
            * (self.growth_energy - self.trial_energy);
        if sign * growth_correction > 0.0 {
            self.trial_energy += growth_correction;
        }
    }

    fn stats<W, F>(&self, trial_wavfun: &W, mut update_stats: F)
        where W: WaveFunction,
              F: FnMut(f64, f64) {
        self.state.stats(|u, x| {
            update_stats(u * self.strategy.weight_coeff(trial_wavfun, x), x)
        });
    }

    fn print_all_stats<W>(&self,
                          step_index: u64,
                          time_step: f64,
                          trial_wavfun: &W)
        where W: WaveFunction {
        let mut stats: Stats = Default::default();
        self.stats(trial_wavfun, |u, x| stats.update(trial_wavfun, u, x));
        stats.calc_average();
        println!("{{");
        println!(" 'step_index': {},", step_index);
        println!(" 'time': {},", step_index as f64 * time_step);
        println!(" 'population': {},", self.population());
        println!(" 'trial_energy': {},", self.trial_energy);
        println!(" 'growth_energy': {},", self.growth_energy);
        println!(" 'stats': {},",
                 rustc_serialize::json::encode(&stats).unwrap());
        println!("}},");
    }

}

fn dmc<R, S, W>(rng: &mut R,
                num_steps: u64,
                state: &[f64],
                time_step: f64,
                trial_energy: f64,
                trial_wavfun: &W,
                overcorrection: f64,
                print_interval: u64,
                branch_interval: u64,
                strategy: S)
    where R: Rng,
          S: Strategy,
          W: IndependentSample<f64> + WaveFunction {
    let branches_per_print = print_interval / branch_interval;
    let num_prints = num_steps / (branches_per_print * branch_interval);

    let mut dmc = DMC::new(state, trial_energy, overcorrection, strategy);
    println!("[");

    dmc.print_all_stats(0, time_step, trial_wavfun);
    for i in 0 .. num_prints {
        for _ in 0 .. branches_per_print {
            for _ in 0 .. branch_interval {
                dmc.step(rng, trial_wavfun, time_step);
            }
            dmc.branch(rng);
        }
        let step_index = (i + 1) * branches_per_print * branch_interval;
        dmc.print_all_stats(step_index, time_step, trial_wavfun);
    }
    println!("],");
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

  --print-interval=<print-interval>
    [default: 1000]

  --num-steps=<num-steps>
    [default: 10000]

  --population=<population>
    Desired walker population.  Actual population may vary slightly.
    [default: 10000]

";

#[derive(Debug, RustcDecodable)]
struct Args {
    flag_seed: RandomSeed,
    flag_num_steps: u64,
    flag_alpha: f64,
    flag_omega: f64,
    flag_dt: f64,
    flag_trial_energy: f64,
    flag_overcorrection: f64,
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

    use rand::distributions::range::Range;

    const NO_VMC: bool = false;

    println!("{{");

    let state =
        if NO_VMC {
            let mut state = Vec::with_capacity(args.flag_population);
            let range = Range::new(-1.0, 1.0);
            for _ in 0 .. args.flag_population {
                state.push(range.ind_sample(&mut rng));
            }
            state
        } else {
            println!("'vmc':");
            vmc(&mut rng,
                args.flag_num_steps,
                100,
                100,
                &trial_wavfun,
                &Range::new(-1.0, 1.0),
                0.2).x_positions
        };

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
