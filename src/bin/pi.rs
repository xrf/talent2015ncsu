extern crate docopt;
extern crate rand;
extern crate rustc_serialize;

use rand::Rng;

const USAGE: &'static str = "
Usage:
  pi <num-samples>
  pi --help

Options:
  -h --help         Show this screen.
  <num-samples>     Number of samples.
";

#[derive(Debug, RustcDecodable)]
struct Args {
    arg_num_samples: u64,
}

fn main() {
    let args: Args = docopt::Docopt::new(USAGE)
        .and_then(|d| d.decode())
        .unwrap_or_else(|e| e.exit());

    let num_samples = args.arg_num_samples;
    let mut rng = rand::thread_rng();

    println!("calculating pi with {} samples ...", num_samples);
    let mut count = 0;
    for _ in 0 .. num_samples {
        let x = rng.next_f64();
        let y = rng.next_f64();
        if x * x + y * y <= 1.0 {
            count += 1;
        }
    }
    println!("pi = {}", 4. * (count as f64) / (num_samples as f64));
}
