#[macro_use]
extern crate clap;
extern crate rand;

use clap::{Arg, App};
use rand::distributions::{IndependentSample, Range};

fn main() {
    let matches = App::new("pi")
        .arg(Arg::with_name("NUM_SAMPLES"))
        .get_matches();

    let num_samples =
        value_t!(matches, "NUM_SAMPLES", u64)
        .unwrap_or_else(|e| e.exit());

    let between = Range::new(-1.0f64, 1.0);
    let mut rng = rand::thread_rng();

    println!("calculating pi with {} samples ...", num_samples);
    let mut count = 0;
    for _ in 0 .. num_samples {
        let x = between.ind_sample(&mut rng);
        let y = between.ind_sample(&mut rng);
        if x * x + y * y <= 1.0 {
            count += 1;
        }
    }
    println!("pi = {}", 4. * (count as f64) / (num_samples as f64));
}
