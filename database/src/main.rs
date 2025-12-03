use std::process::exit;
use crate::protocol::serve;

mod argument;
mod cache;
mod common;
mod model;
mod protocol;
mod storage;
mod thread_pool;
mod logger;

fn main() {
	if let Err(error) = serve() {
		fatal!("{}\n", error);
		exit(1);
	}
}