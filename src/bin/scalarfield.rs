extern crate docopt;
extern crate ndarray;
extern crate num;
extern crate rand;
extern crate rustc_serialize;
#[macro_use]
extern crate qmc;

use std::io::{Write, stderr, stdout};
use std::f64::consts::PI;
use ndarray::prelude::*;
use rand::Rng;
use rand::distributions::{IndependentSample, Range};

use qmc::utils::*;

const USAGE: &'static str = "
Usage:
  scalarfield [options]

Options:

  -h --help
    Show this screen.

  -s --seed=<seed>
    Seed for the random number generator in LE hexadecimal.
    [default is empty string]

  --num-samples=<num-samples>
    [default: 1000000]

  --param-l=<param-l>
    [default: 10]

  --param-lt=<param-lt>
    [default: 10]

  --param-m-sq=<param-m-sq>
    [default: 1.0]

  --rate=<rate>
    Determines how much we should perturb the values of the scalar field.
    [default: 1.0]

";

#[derive(Debug, RustcDecodable)]
struct Args {
    flag_seed: String,
    flag_num_samples: usize,
    flag_param_l: usize,
    flag_param_lt: usize,
    flag_param_m_sq: f64,
    flag_rate: f64,
}

#[allow(dead_code)]
fn action(m_sq: f64,
          l: usize,
          lt: usize,
          phi: &ArrayView<f64, (usize, usize, usize, usize)>)
          -> f64{
    let mut r1 = 0.0;
    let mut r2 = 0.0;
    for nx in 0 .. l {
        for ny in 0 .. l {
            for nz in 0 .. l {
                for nt in 0 .. lt {
                    let r3 = phi[(nx, ny, nz, nt)];
                    r1 += r3 * (
                        phi[((nx + 1) % l, ny, nz, nt)]
                      + phi[(nx, (ny + 1) % l, nz, nt)]
                      + phi[(nx, ny, (nz + 1) % l, nt)]
                      + phi[(nx, ny, nz, (nt + 1) % lt)]
                    );
                    r2 += r3.powi(2);
                }
            }
        }
    }
    -r1 + 0.5 * (8.0 + m_sq) * r2
}

fn nearby_action(m_sq: f64,
                 l: usize,
                 lt: usize,
                 phi: &ArrayView<f64, (usize, usize, usize, usize)>,
                 n: (usize, usize, usize, usize),
                 phi_n: f64)
                 -> f64{
    let c = 0.5 * (8.0 + m_sq);
    let (nx, ny, nz, nt) = n;
    phi_n * (-(
        phi[((nx + 1) % l, ny, nz, nt)]
      + phi[(nx, (ny + 1) % l, nz, nt)]
      + phi[(nx, ny, (nz + 1) % l, nt)]
      + phi[(nx, ny, nz, (nt + 1) % lt)]
      + phi[((nx + l - 1) % l, ny, nz, nt)]
      + phi[(nx, (ny + l - 1) % l, nz, nt)]
      + phi[(nx, ny, (nz + l - 1) % l, nt)]
      + phi[(nx, ny, nz, (nt + l - 1) % lt)]
    ) + c * phi_n)
}

fn mcmc_step<R: Rng>(rng: &mut R,
                     phi: &mut ArrayViewMut<f64, (usize, usize, usize, usize)>,
                     m_sq: f64,
                     l: usize,
                     lt: usize,
                     rate: f64) {
    let l_range = Range::new(0, l);
    let lt_range = Range::new(0, lt);
    let phi_range = Range::new(-rate, rate);
    let nx = l_range.ind_sample(rng);
    let ny = l_range.ind_sample(rng);
    let nz = l_range.ind_sample(rng);
    let nt = lt_range.ind_sample(rng);
    let dphi = phi_range.ind_sample(rng);
    let old_phi = phi[(nx, ny, nz, nt)];
    let new_phi = old_phi + dphi;
    let p_ratio = (
       -nearby_action(m_sq, l, lt, &phi.view(), (nx, ny, nz, nt), new_phi)
      + nearby_action(m_sq, l, lt, &phi.view(), (nx, ny, nz, nt), old_phi)
    ).exp();
    if p_ratio >= 1.0 || rng.next_f64() < p_ratio {
        phi[(nx, ny, nz, nt)] = new_phi;
    }
}

fn exact_pcf(m_sq: f64,
             l: usize,
             lt: usize,
             n_minus_m: (usize, usize, usize, usize))
             -> f64 {
    let kunit = 2.0 * PI / (l as f64);
    let ktunit = 2.0 * PI / (lt as f64);
    let mut sum = 0.0;
    for kix in 0 .. l {
        for kiy in 0 .. l {
            for kiz in 0 .. l {
                for kit in 0 .. l {
                    sum = sum +
                        (kunit * (
                            kix * n_minus_m.0
                          + kiy * n_minus_m.1
                          + kiz * n_minus_m.2
                        ) as f64 + ktunit * ((kit * n_minus_m.3) as f64)
                        ).cos()
                        / (2.0 * (
                            4.0
                          - (kunit * (kix as f64)).cos()
                          - (kunit * (kiy as f64)).cos()
                          - (kunit * (kiz as f64)).cos()
                          - (ktunit * (kit as f64)).cos()
                        ) + m_sq);
                }
            }
        }
    }
    sum / ((l as f64).powi(3) * (lt as f64))
}

fn mcmc_pcf<R: Rng>(rng: &mut R,
                    m_sq: f64,
                    l: usize,
                    lt: usize,
                    n_minus_m_args: &[(usize, usize, usize, usize)],
                    num_samples: usize,
                    rate: f64)
                    -> Vec<f64> {
    let mut phi = Array::zeros((l, l, l, lt));
    let num_points = num_samples * l * l;
    let mut pcfs = vec![0.0; n_minus_m_args.len()];
    write!(stderr(), "#").unwrap();
    for k in 0 .. num_samples {
        mcmc_step(rng, &mut phi.view_mut(), m_sq, l, lt, rate);
        // note that we do not sample over the entire 4D volume
        // because it can increase the computational cost significantly
        // which means we would be forced to do fewer MCMC iterations
        // and thus reducing the quality of our samples
        for nx in 0 .. 1 {
            for ny in 0 .. l {
                for nz in 0 .. l {
                    for nt in 0 .. 1 {
                        for i in 0 .. n_minus_m_args.len() {
                            let n_minus_m = n_minus_m_args[i];
                            pcfs[i] +=
                                phi[((nx + n_minus_m.0) % l,
                                     (ny + n_minus_m.1) % l,
                                     (nz + n_minus_m.2) % l,
                                     (nt + n_minus_m.3) % lt)]
                              * phi[(nx, ny, nz, nt)];
                        }
                    }
                }
            }
        }
        if k % (num_samples / 100) == 0 {
            write!(stderr(), "\r# {:3}%", k * 100 / num_samples).unwrap();
            stderr().flush().unwrap();
        }
    }
    writeln!(stderr(), "\r#     ").unwrap();
    for pcf in pcfs.iter_mut() {
        *pcf /= num_points as f64
    }
    pcfs
}

fn main() {
    let args: Args = docopt::Docopt::new(USAGE)
        .and_then(|d| d.decode())
        .unwrap_or_else(|e| e.exit());
    let seed = decode_seed(&args.flag_seed);

    println!("# args: {:?}", args);
    stdout().flush().unwrap();

    let mut rng = rng_from_seed(&seed);

    let l = args.flag_param_l;
    let lt = args.flag_param_lt;
    let m_sq = args.flag_param_m_sq;

    let n_minus_m_args: Vec<_> = (0 .. l).map(|nx| (nx, 0, 0, 0)).collect();

    writeln!(stderr(), "# Calculating analytically ...").unwrap();
    stderr().flush().unwrap();
    let mut pcfs_exact = Vec::new();
    for n_minus_m in n_minus_m_args.iter() {
        pcfs_exact.push(exact_pcf(m_sq, l, lt, *n_minus_m));
    }

    writeln!(stderr(), "# Calculating using MCMC ...").unwrap();
    stderr().flush().unwrap();
    let pcfs_mcmc = mcmc_pcf(&mut rng, m_sq, l, lt, &n_minus_m_args,
                             args.flag_num_samples, args.flag_rate);

    println!("{{");
    println!("'analytic': {:?},", pcfs_exact);
    println!("'mcmc': {:?},", pcfs_mcmc);
    println!("}}");
}
