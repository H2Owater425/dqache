use std::{
	env::{
		args,
		consts::{ARCH, OS},
		Args
	},
	fs::metadata,
	process::exit
};
use crate::{
	cache::Model,
	exit_with,
	protocol::Version
};

pub struct Argument {
	pub model: Model,
	pub capacity: usize,
	pub directory: String,
	pub port: u16,
	pub is_verbose: bool,
	pub version: Version,
	pub platform: String
}

impl Argument {
	pub fn new() -> Self {
		let mut argument: Argument = Argument {
			model: Model::DeepQNetwork,
			capacity: 128,
			directory: "./data".to_string(),
			port: 5190,
			is_verbose: false,
			version: Version::try_from(env!("CARGO_PKG_VERSION"))
				.unwrap_or_else(|_| exit_with!(1, "package version must be sementic\n")),
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

		let mut arguments: Args = args();

		arguments.next();

		while let Some(value) = arguments.next() {
			match value.as_str() {
				"--model" | "-M" => if let Some(model) = arguments.next() {
					argument.model = match model.to_ascii_lowercase().as_str() {
						"dqn" | "deepqnetwork" => Model::DeepQNetwork,
						"lru" | "leastrecentlyused" => Model::LeastRecentlyUsed,
						"lfu" | "leastfrequentlyused" => Model::LeastFrequentlyUsed,
						_ => exit_with!(1, "model type must be one of dqn, lru, lfu\n")
					}
				} else {
					exit_with!(1, "model type must be provided\n");
				},
				"--capacity" | "-C" => if let Some(value) = arguments.next() {
					if let Ok(capacity) = value.parse::<usize>() {
						argument.capacity = capacity;
					} else {
						exit_with!(1, "capacity must be integer greater than 0\n");
					}
				} else {
					exit_with!(1, "capacity must be provided\n");
				},
				"--directory" | "-D" => if let Some(directory) = arguments.next() {
					if let Ok(metadata) = metadata(&directory) {
						if metadata.is_dir() {
							argument.directory = directory;
						} else {
							exit_with!(1, "directory must not be file\n");
						}
					} else {
						exit_with!(1, "directory must be accessible\n");
					}
				} else {
					exit_with!(1, "directory must be provided\n");
				},
				"--port" | "-P" => if let Some(value) = arguments.next() {
					if let Ok(port) = value.parse::<u16>() {
						argument.port = port;
					} else {
						exit_with!(1, "port must be between 1 to 65535\n");
					}
				} else {
					exit_with!(1, "port must be provided\n");
				},
				"--verbose" | "-v" => argument.is_verbose = true,
				"--version" | "-V" => exit_with!(1, "qache {}\n", argument.version),
				"--help" | "-h" => exit_with!(0, "Usage: qache [OPTIONS]

Options:
	-M, --model <MODEL>          Set cache model [DQN, LRU, LFU] (default: DQN)
	-C, --capacity <CAPACITY>    Set cache capacity (default: 128)
	-D, --directory <DIRECTORY>  Set data directory (default: ./data)
	-P, --port <PORT>            Set server port (default: 5190)
	-v, --verbose                Enable verbose output
	-V, --version                Print version information
	-h, --help                   Print this help message
"),
				"--" => if let Some(_) = arguments.next() {
					exit_with!(1, "positional arguments must not be provided\n");
				},
				_ => exit_with!(1, "Usage: qache [-M <MODEL>] [-C <CAPACITY>] [-D <DIRECTORY>] [-P <PORT>] [-v] [-V] [-h]\n")
			}
		}

		argument
	}
}