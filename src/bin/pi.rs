extern crate rand;

use rand::distributions::{IndependentSample, Range};

fn main() {
    let between = Range::new(-1.0f64, 1.0);
    let mut rng = rand::thread_rng();

    println!("calculating pi ...");
    let n = 10000000;
    let mut count = 0;
    for _ in 0 .. n {
        let x = between.ind_sample(&mut rng);
        let y = between.ind_sample(&mut rng);
        if x * x + y * y <= 1.0 {
            count += 1;
        }
    }
    println!("pi = {}", 4. * (count as f64) / (n as f64));
}
