use std::{
	error::Error,
	io::{stderr, stdout},
	net::TcpStream,
	process::exit,
	result::Result as _Result,
	sync::LazyLock,
	time::{SystemTime, UNIX_EPOCH}
};
use crate::{
	argument::Argument,
	logger::Logger
};

pub type Job = Box<dyn FnOnce() + Send + 'static>;

pub type Result<T, E = Box<dyn Error>> = _Result<T, E>;

pub const ARGUMENT: LazyLock<Argument> = LazyLock::new(|| {
	match Argument::new() {
		Ok(argument) => argument,
		Err(error) => {
			eprint!("{}\n", error);

			exit(1);
		}
	}
});

pub const LOGGER: LazyLock<Logger> = LazyLock::new(|| Logger::new(stdout(), stderr(), 5));

#[macro_export]
macro_rules! info {
	($($arg:tt)*) => {
		crate::common::LOGGER.info(&format!($($arg)*));
	}
}

#[macro_export]
macro_rules! fatal {
	($($arg:tt)*) => {
		crate::common::LOGGER.fatal(&format!($($arg)*));
	}
}

#[macro_export]
macro_rules! error {
	($($arg:tt)*) => {
		crate::common::LOGGER.error(&format!($($arg)*));
	}
}

#[macro_export]
macro_rules! warn {
	($($arg:tt)*) => {
		crate::common::LOGGER.warn(&format!($($arg)*));
	}
}

#[macro_export]
macro_rules! debug {
	($($arg:tt)*) => {
		crate::common::LOGGER.debug(&format!($($arg)*));
	}
}

//#[macro_export]
//macro_rules! trace {
//	($($arg:tt)*) => {
//		crate::common::LOGGER.trace(&format!($($arg)*));
//	}
//}

pub fn unix_epoch() -> Result<u64> {
	Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs())
}

pub fn log1p(x: u64) -> f32 {
	(x as f64).ln_1p() as f32
}

pub fn get_address(stream: &TcpStream) -> String {
	if let Ok(address) = stream.peer_addr() {
		address.to_string()
	} else {
		"unknown".to_owned()
	}
}