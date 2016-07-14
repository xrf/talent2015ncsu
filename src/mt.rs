extern crate libc;

#[link(name = "mt", kind = "static")]
extern {
    fn sgrnd_(seed: *const libc::c_int);
    fn grnd_() -> libc::c_double;
    fn gaussrnd_(rr: *mut libc::c_double);
}

pub fn sgrnd(seed: i32) {
    unsafe { sgrnd_(&seed); }
}

pub fn seed(seed: &[u8]) {
    sgrnd(*::utils::repack_u8s(&seed).get(0).unwrap_or(&0));
}

pub fn grnd() -> f64 {
    unsafe { grnd_() }
}

pub fn gaussrnd() -> f64 {
    unsafe {
        let mut rr = ::std::mem::uninitialized();
        gaussrnd_(&mut rr);
        rr
    }
}
