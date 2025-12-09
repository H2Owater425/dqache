use std::{
	env::{
		Args,
		args,
		consts::{ARCH, OS},
		current_exe
	},
	fs::metadata,
	iter::Skip,
	net::Ipv4Addr,
	process::exit,
	thread::available_parallelism
};
use crate::{
	common::Result,
	model::Model,
	protocol::Version
};

pub struct Argument {
	pub model: Model,
	pub capacity: usize,
	pub directory: String,
	pub host: Ipv4Addr,
	pub port: u16,
	pub thread_count: usize,
	pub is_verbose: bool,
	pub version: Version,
	pub platform: String
}

const DEFAULT_DIRECTORY: &str = if cfg!(target_os = "windows") {
	".\\data"
} else {
	"./data"
};

impl Argument {
	pub fn new() -> Result<Self> {
		let mut argument: Argument = Argument {
			model: Model::DeepQNetwork,
			capacity: 128,
			directory: DEFAULT_DIRECTORY.to_string(),
			host: Ipv4Addr::new(127, 0, 0, 1),
			port: 5190,
			thread_count: available_parallelism()?.get() * 2,
			is_verbose: false,
			version: Version::try_from(env!("CARGO_PKG_VERSION"))?,
			platform: format!("{}-{}-{}{}", ARCH, OS, if cfg!(target_vendor = "apple") {
				"apple"
			} else if cfg!(target_vendor = "pc") {
				"pc"
			} else if cfg!(target_vendor = "fortanix") {
				"fortanix"
			} else {
				"unknown"
			}, if cfg!(target_env = "gnu") {
				"-gnu"
			} else if cfg!(target_env = "msvc") {
				"-msvc"
			} else if cfg!(target_env = "musl") {
				"-musl"
			} else if cfg!(target_env = "sgx") {
				"-sgx"
			} else {
				""
			})
		};

		let file_name: String = if let Some(raw_file_name) = current_exe()?.file_name() {
			raw_file_name.display()
				.to_string()
		} else {
			return Err(Box::from("executable path must be valid"));
		};

		let mut arguments: Skip<Args> = args().skip(1);

		while let Some(value) = arguments.next() {
			match value.as_str() {
				"--model" | "-m" => if let Some(raw_model) = arguments.next() {
					match raw_model.to_ascii_lowercase()
						.as_str() {
						"dqn" | "deepqnetwork" => argument.model = Model::DeepQNetwork,
						"lru" | "leastrecentlyused" => argument.model = Model::LeastRecentlyUsed,
						"lfu" | "leastfrequentlyused" => argument.model = Model::LeastFrequentlyUsed,
						_ => return Err(Box::from("model must be one of dqn, lru, lfu"))
					}
				} else {
					return Err(Box::from("model must be provided"));
				}
				"--capacity" | "-c" => if let Some(raw_capacity) = arguments.next() {
					argument.capacity = raw_capacity.parse::<usize>()?;

					if argument.capacity == 0 {
						return Err(Box::from("capacity must be greater than 0"))
					}
				} else {
					return Err(Box::from("capacity must be provided"));
				},
				"--directory" | "-d" => if let Some(directory) = arguments.next() {
					argument.directory = directory;

					if !metadata(&argument.directory)?.is_dir() {
						return Err(Box::from("directory must be folder"));
					}
				} else {
					return Err(Box::from("directory must be provided"));
				}
				"--host" | "-H" => if let Some(raw_host) = arguments.next() {
					argument.host = raw_host.parse::<Ipv4Addr>()?;
				} else {
					return Err(Box::from("host must be provided"));
				},
				"--port" | "-p" => if let Some(raw_port) = arguments.next() {
					argument.port = raw_port.parse::<u16>()?;

					if argument.port == 0 {
						return Err(Box::from("port must be greater than 0"));
					}
				},
				"--threadcount" | "-t" => if let Some(raw_thread_count) = arguments.next() {
					argument.thread_count = raw_thread_count.parse::<usize>()?;

					if argument.thread_count == 0 {
						return Err(Box::from("thread count must be greater than 0"));
					}
				},
				"--verbose" | "-v" => argument.is_verbose = true,
				"--version" | "-V" => {
					print!("{} {}\n", file_name, argument.version);

					exit(0);
				},
				"--help" | "-h" => {
					print!("Usage: {} [OPTIONS]

	Options:
		-m, --model <MODEL>          Set cache model [dqn, lru, lfu] (default: dqn)
		-c, --capacity <CAPACITY>    Set cache capacity (default: 128)
		-d, --directory <DIRECTORY>  Set data directory (default: {})
		-H, --host <HOST>            Set server host (default: 127.0.0.1)
		-p, --port <PORT>            Set server port (default: 5190)
		-t, --threadcount <COUNT>    Set thread count (default: number of logical cores * 2)
		-v, --verbose                Enable verbose output
		-V, --version                Print version information
		-h, --help                   Print this help message
", file_name, DEFAULT_DIRECTORY);

					exit(0);
				},
				"--" => if let Some(_) = arguments.next() {
					return Err(Box::from("positional arguments must not be provided"));
				},
				_ => return Err(Box::from(format!("Usage: {} [-m <MODEL>] [-c <CAPACITY>] [-d <DIRECTORY>] [-H <HOST>] [-p <PORT>] [-t <COUNT>] [-v] [-V] [-h]", file_name)))
			}
		}

		Ok(argument)
	}
}