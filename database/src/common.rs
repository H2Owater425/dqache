use std::{
	error::Error,
	result::Result as _Result,
	sync::LazyLock,
	time::{SystemTime, UNIX_EPOCH}
};
use crate::argument::Argument;

#[macro_export]
macro_rules! exit_with {
	($code:literal, $($arg:tt)*) => {{
		eprint!($($arg)*);
		exit($code);
	}};
}

pub static ARGUMENT: LazyLock<Argument> = LazyLock::new(|| Argument::new());

pub fn unix_epoch() -> Result<u64> {
	Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs())
}

pub fn log1p(x: u64) -> f32 {
	(x as f64).ln_1p() as f32
}

pub type Job = Box<dyn FnOnce() + Send + 'static>;

pub type Result<T, E = Box<dyn Error>> = _Result<T, E>;