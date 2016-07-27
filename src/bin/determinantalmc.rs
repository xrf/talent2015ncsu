extern crate docopt;
extern crate la;
extern crate ndarray;
extern crate num;
extern crate rand;
extern crate rustc_serialize;
#[macro_use]
extern crate qmc;

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
    [default: 1000]

  --sampling-interval=<sampling-interval>
    [default: 10]

  --param-nx=<param-nx>
    [default: 10]

  --param-nt=<param-nt>
    [default: 10]

  --param-dt=<param-dt>
    [default: 0.001]

  --param-mdx2=<param-mdx2>
    m dx^2
    [default: 1.0]

  --param-g=<param-g>
    [default: 0.1]

";

#[derive(Debug, RustcDecodable)]
struct Args {
    flag_seed: String,
    flag_num_samples: usize,
    flag_sampling_interval: usize,
    flag_param_nx: usize,
    flag_param_nt: usize,
    flag_param_dt: f64,
    flag_param_mdx2: f64,
    flag_param_g: f64,
}

fn inverse(m: &ArrayView<f64, (usize, usize)>) -> Array<f64, (usize, usize)> {
    let n = m.shape()[0];
    assert_eq!(n, m.shape()[1]);
    let m = ::la::Matrix::new(n, n, m.iter().cloned().collect());
    Array::from_iter(m.inverse().unwrap().get_data().iter().cloned())
        .into_shape((n, n)).unwrap()
}

fn log_det_sq(m: &ArrayView<f64, (usize, usize)>) -> f64 {
    let n = m.shape()[0];
    assert_eq!(n, m.shape()[1]);
    let m = ::la::Matrix::new(n, n, m.iter().cloned().collect());
    let lu = ::la::LUDecomposition::new(&m);
    let mut z = 0.0;
    for i in 0 .. n {
        z +=
            lu.get_l().get(i, i).abs().ln() +
            lu.get_u().get(i, i).abs().ln();
    }
    2.0 * z
}

fn kinetic_matrix(nx: usize) -> Array<f64, (usize, usize)> {
    let mut t = Array::zeros((nx, nx));
    for i in 0 .. nx {
        t[(i, (i + nx - 1) % nx)] = 1.0;
        t[(i, i)] = -2.0;
        t[(i, (i + 1) % nx)] = 1.0;
    }
    t
}

fn apply_potential(sqrt_c: f64,
                   d_sqrt_c: f64,
                   transfer: &mut ArrayViewMut<f64, (usize, usize)>,
                   d_transfer: &mut ArrayViewMut<f64, (usize, usize)>,
                   sigma: &ArrayView<i8, usize>) {
    let nx = transfer.shape()[0];
    for x in 0 .. nx {
        for i in 0 .. nx {
            let sigma_x = sigma[x] as f64;
            let b = 1.0 + sqrt_c * sigma_x;
            let transfer_x_i = transfer[(x, i)];
            transfer[(x, i)] = b * transfer_x_i;
            d_transfer[(x, i)] =
                d_sqrt_c * sigma_x * transfer_x_i +
                b * d_transfer[(x, i)];
        }
    }
}

struct PartitionFunction {
    sqrt_c: f64,
    d_sqrt_c: f64,
    evolve_kinetic: Array<f64, (usize, usize)>,
    d_evolve_kinetic: Array<f64, (usize, usize)>,
}

impl PartitionFunction {
    fn new(nx: usize, dt: f64, mdx2: f64, g: f64) -> PartitionFunction {
        let exp_dt_g = (dt * g).exp();
        let t = kinetic_matrix(nx);
        PartitionFunction {
            sqrt_c: exp_dt_g - 1.0,
            d_sqrt_c: exp_dt_g * g,
            evolve_kinetic: Array::eye(nx) + (-0.25 * dt / mdx2) * &t,
            d_evolve_kinetic: (-0.25 / mdx2) * &t,
        }
    }

    fn eval(&self, sigma: &ArrayView<i8, (usize, usize)>) -> (f64, f64) {
        let nt = sigma.shape()[0];
        let nx = sigma.shape()[1];
        let mut transfer = Array::eye(nx);
        let mut d_transfer = Array::zeros((nx, nx));
        for t in 0 .. nt {
            d_transfer =
                self.d_evolve_kinetic.dot(&transfer) +
                self.evolve_kinetic.dot(&d_transfer);
            transfer = self.evolve_kinetic.dot(&transfer);
            apply_potential(self.sqrt_c,
                            self.d_sqrt_c,
                            &mut transfer.view_mut(),
                            &mut d_transfer.view_mut(),
                            &sigma.subview(Axis(0), t));
            d_transfer =
                self.d_evolve_kinetic.dot(&transfer) +
                self.evolve_kinetic.dot(&d_transfer);
            transfer = self.evolve_kinetic.dot(&transfer);
        }
        let fermi = Array::eye(nx) + transfer;
        let d_fermi = d_transfer;
        let mdm = 2.0 * &inverse(&fermi.view()).dot(
            &((-1.0 / (nt as f64)) * d_fermi));
        let mut tr_mdm = 0.0;
        for i in 0 .. nx {
            tr_mdm = mdm[(i, i)];
        }
        (log_det_sq(&fermi.view()), tr_mdm)
    }
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

    let num_samples = args.flag_num_samples;
    let sampling_interval = args.flag_sampling_interval;
    let nx = args.flag_param_nx;
    let nt = args.flag_param_nt;
    let dt = args.flag_param_dt;
    let mdx2 = args.flag_param_mdx2;    // m dx^2
    let g = args.flag_param_g;

    let x_range = Range::new(0, nx);
    let t_range = Range::new(0, nt);
    let partition_function = PartitionFunction::new(nx, dt, mdx2, g);

    let mut sigma = Array::from_elem((nt, nx), 1i8);
    let (mut z, mut e) = partition_function.eval(&sigma.view());
    let mut e_sum = 0.0;
    for _ in 0 .. num_samples {

        for _ in 0 .. sampling_interval {
            let x_change = x_range.ind_sample(&mut rng);
            let t_change = t_range.ind_sample(&mut rng);
            sigma[(t_change, x_change)] *= -1;

            let (new_z, new_e) = partition_function.eval(&sigma.view());
            println!("# {} -> {}", z, new_z);
            println!("# {} -> {}", e, new_e);

            if rng.next_f64() < (new_z - z).exp() {
                z = new_z;
                e = new_e;
            } else {
                println!("# rejected");
                sigma[(t_change, x_change)] *= -1;
            }
        }

        // compute statistics (currently just energy)
        e_sum += e;

    }

    println!("{{");
    println!("'num_samples': {:?},", num_samples);
    println!("'energy_avg': {:?},", e_sum / (num_samples as f64));
    println!("}}");
}
