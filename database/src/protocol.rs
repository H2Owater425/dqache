use std::{
	cmp::Ordering,
	error::Error,
	fmt::{Display, Formatter, Result as _Result},
	io::{IoSlice, Read, Write, Error as _Error, ErrorKind},
	net::{TcpListener, TcpStream},
	sync::{
		Arc,
		Mutex,
		MutexGuard,
		PoisonError,
		RwLock,
		RwLockReadGuard,
		RwLockWriteGuard
	},
	thread::available_parallelism,
	time::Duration
};
use crate::{
	common::Result,
	cache::{Cache, Entry},
	common::{ARGUMENT, get_address},
	storage::Storage,
	thread_pool::ThreadPool,
	error,
	info,
	warn
};

/*
	big endian

	-- handshake --
	READY <major:u8> <minor:u8> <patch:u8>
	HELLO <major:u8> <minor:u8> <patch:u8>

	-- request --
	NOP
	SET   <length:u8> <key:String> <length:u32> <value:String>
	DEL   <length:u8> <key:String>
	GET   <length:u8> <key:String>

	-- responses --
	OKAY
	VALUE <length:u32> <value:String>
	ERROR <length:u32> <message:String>

	-- termination --
	QUIT
*/

pub const OPERATION_READY: &[u8; 1] = &[0b10000000];
pub const OPERATION_HELLO: &[u8; 1] = &[0b00000000];
pub const OPERATION_NOP: &[u8; 1] = &[0b00000010];
pub const OPERATION_SET: &[u8; 1] = &[0b00000011];
pub const OPERATION_DEL: &[u8; 1] = &[0b00000100];
pub const OPERATION_GET: &[u8; 1] = &[0b00000101];
pub const OPERATION_OK: &[u8; 1] = &[0b10000010];
pub const OPERATION_VALUE: &[u8; 1] = &[0b10000011];
pub const OPERATION_ERROR: &[u8; 1] = &[0b10000100];
pub const OPERATION_QUIT: &[u8; 1] = &[0b11111111];

pub fn read_string<const N: usize>(stream: &mut TcpStream, byte_or_double_word: &mut [u8; N]) -> Result<String> {
	stream.read_exact(byte_or_double_word)?;

	let mut buffer: Vec<u8> = vec![0; if N == 1 {
		byte_or_double_word[0] as usize
	} else if N == 4 {
		(byte_or_double_word[0] as usize) << 24 | (byte_or_double_word[1] as usize) << 16 | (byte_or_double_word[2] as usize) << 8 | byte_or_double_word[3] as usize
	} else {
		return Err(Box::from("buffer size must be 1 or 4"));
	}];

	if buffer.len() == 0 {
		return Err(Box::from("length must be greater than 0"));
	}

	stream.read_exact(&mut buffer)?;

	Ok(String::from_utf8(buffer)?)
}

pub fn send_error(stream: &mut TcpStream, double_word: &mut [u8; 4], message: String) -> Result<()> {
	let message_length: usize = message.len();

	error!("{} to {}\n", message, stream.peer_addr()?);

	double_word[0] = (message_length >> 24) as u8;
	double_word[1] = (message_length >> 16) as u8;
	double_word[2] = (message_length >> 8) as u8;
	double_word[3] = message_length as u8;

	stream.write_vectored(&[
		IoSlice::new(OPERATION_ERROR),
		IoSlice::new(double_word),
		IoSlice::new(message.as_bytes())
	])?;

	Ok(())
}

pub struct Version {
	major: u8,
	minor: u8,
	patch: u8
}

impl Version {
	pub fn new(major: u8, minor: u8, patch: u8) -> Self {
		Version {
			major: major,
			minor: minor,
			patch: patch
		}
	}

	pub fn as_bytes(self: &Self) -> [u8; 3] {
		[self.major, self.minor, self.patch]
	}
}

impl TryFrom<&str> for Version {
	type Error = Box<dyn Error>;

	fn try_from(value: &str) -> Result<Self> {
		let mut version: Version = Version::new(0, 0, 0);
		let mut start: usize = 0;

		if let Some(end) = value.find('.') {
			version.major = value[start..end].parse::<u8>()?;
			start = end + 1;
		} else {
			version.major = value[start..].parse::<u8>()?;

			return Ok(version);
		}

		if let Some(end) = value[start..].find('.') {
			version.minor = value[start..start + end].parse::<u8>()?;
			start = start + end + 1;
		} else {
			version.minor = value[start..].parse::<u8>()?;

			return Ok(version);
		}

		version.patch = value[start..].parse::<u8>()?;

		Ok(version)
	}
}

impl TryFrom<&[u8]> for Version {
	type Error = Box<dyn Error>;

	fn try_from(value: &[u8]) -> Result<Self> {
		if value.len() != 3 {
			return Err(Box::from("value length must be 3"));
		}

		Ok(Version::new(value[0], value[1], value[2]))
	}
}

impl PartialEq for Version {
	fn eq(self: &Self, other: &Self) -> bool {
		self.major == other.major && self.minor == other.minor && self.patch == other.patch
	}
}

impl PartialOrd for Version {
	fn partial_cmp(self: &Self, other: &Self) -> Option<Ordering> {
		match self.major.cmp(&other.major) {
			Ordering::Equal => (),
			ordering => return Some(ordering)
		}

		match self.minor.cmp(&other.minor) {
			Ordering::Equal => (),
			ordering => return Some(ordering)
		}

		Some(self.patch.cmp(&other.patch))
	}
}

impl Display for Version {
	fn fmt(self: &Self, formatter: &mut Formatter<'_>) -> _Result {
		write!(formatter, "{}.{}.{}", self.major, self.minor, self.patch)
	}
}

pub fn serve() -> Result<()> {
	info!("starting dQache {} on {}\n", ARGUMENT.version, ARGUMENT.platform);

	let cache: Arc<Mutex<Cache>> = Arc::new(Mutex::new(Cache::new(ARGUMENT.model, ARGUMENT.capacity)?));
	let storage: Arc<RwLock<Storage>> = Arc::new(RwLock::new(Storage::new(&ARGUMENT.directory)?));
	let thread_pool: ThreadPool = ThreadPool::new(available_parallelism()?.get() * 2)?;
	let listener: TcpListener = TcpListener::bind((ARGUMENT.host, ARGUMENT.port))?;

	info!("lisening on {}:{} with {} threads\n", ARGUMENT.host, ARGUMENT.port, thread_pool.size());

	for stream in listener.incoming() {
		let mut stream: TcpStream = stream?;
		let cache: Arc<Mutex<Cache>> = cache.clone();
		let storage: Arc<RwLock<Storage>> = storage.clone();

		stream.set_read_timeout(Some(Duration::from_secs(60)))?;
		stream.set_nodelay(true)?;

		thread_pool.execute(move || {
			let mut double_word: [u8; 4] = [0; 4];

			if let Err(error) = (|| -> Result<()> {
				stream.write_vectored(&[
					IoSlice::new(OPERATION_READY),
					IoSlice::new(&ARGUMENT.version.as_bytes())
				])?;

				// Use value_length as handshake
				stream.read_exact(&mut double_word)?;

				if double_word[0] != OPERATION_HELLO[0] {
					return Err(Box::from("handshake must start with HELLO operation"));
				}

				if let Ok(version) = Version::try_from(&double_word[1..4]) {
					if version > ARGUMENT.version {
						return Err(Box::from(format!("client version must be less than or equal to {}", ARGUMENT.version)));
					}

					info!("client connected with {} from {}\n", version, stream.peer_addr()?);
				} else {
					return Err(Box::from("client version must be invalid\n"));
				}

				stream.write(OPERATION_OK)?;

				Ok(())
			})() {
				let _ = send_error(&mut stream, &mut double_word, error.to_string());

				return;
			}

			let mut byte: [u8; 1] = [0];

			loop {
				if let Err(error) = (|| -> Result<()> {
					stream.read_exact(&mut byte)?;

					match &byte {
						OPERATION_SET => {
							let key: String = read_string::<1>(&mut stream, &mut byte)?;
							let value: String = read_string::<4>(&mut stream, &mut double_word)?;

							cache.lock()
								.map_err(|error: PoisonError<MutexGuard<'_, Cache>>| error.to_string())?
								.set(&key, Entry::new(&value)?)?;
							storage.write()
								.map_err(|error: PoisonError<RwLockWriteGuard<'_, Storage>>| error.to_string())?
								.write(&key, value)?;

							stream.write(OPERATION_OK)?;
						},
						OPERATION_DEL => {
							let key: String = read_string::<1>(&mut stream, &mut byte)?;

							cache.lock()
								.map_err(|error: PoisonError<MutexGuard<'_, Cache>>| error.to_string())?
								.remove(&key);

							if !storage.write()
								.map_err(|error: PoisonError<RwLockWriteGuard<'_, Storage>>| error.to_string())?
								.delete(&key)? {
								return Err(Box::from("key must exist"));
							}

							stream.write(OPERATION_OK)?;
						},
						OPERATION_GET => {
							let key: String = read_string::<1>(&mut stream, &mut byte)?;
							let (is_cached, value): (bool, String) = if let Some(entry) = cache.lock()
								.map_err(|error: PoisonError<MutexGuard<'_, Cache>>| error.to_string())?
								.get(&key)? {
								(true, entry.value.clone())
							} else {
								if let Some(value) = storage.read()
									.map_err(|error: PoisonError<RwLockReadGuard<'_, Storage>>| error.to_string())?
									.read(&key)? {
										(false, value)
									} else {
										return Err(Box::from("key must exist"));
									}
							};

							if !is_cached {
								cache.lock()
									.map_err(|error: PoisonError<MutexGuard<'_, Cache>>| error.to_string())?
									.set(&key, Entry::new(&value)?)?;
							}

							let value_length: usize = value.len();

							double_word[0] = (value_length >> 24) as u8;
							double_word[1] = (value_length >> 16) as u8;
							double_word[2] = (value_length >> 8) as u8;
							double_word[3] = value_length as u8;

							stream.write_vectored(&[
								IoSlice::new(OPERATION_VALUE),
								IoSlice::new(&double_word),
								IoSlice::new(value.as_bytes())
							])?;
						},
						OPERATION_NOP => {
							stream.write(OPERATION_OK)?;
						},
						OPERATION_QUIT => {
							return Err(Box::from(""));
						},
						_ => {
							return Err(Box::from("operation must be valid"));
						}
					}

					Ok(())
				})() {
					if let Some(error) = error.downcast_ref::<_Error>() {
						let _ = send_error(&mut stream, &mut double_word, match error.kind() {
							ErrorKind::UnexpectedEof => {
								warn!("client terminated from {}\n", get_address(&stream));

								break;
							},
							ErrorKind::StorageFull => "storage must have free space".to_owned(),
							ErrorKind::WouldBlock | ErrorKind::TimedOut => "packet must be sent in time".to_owned(),
							ErrorKind::OutOfMemory => "memory must have free space".to_owned(),
							_ => error.to_string()
						});
						let _ = stream.write(OPERATION_QUIT);

						break;
					}

					let message: String = error.to_string();

					if message.len() == 0 {
						info!("client disconnected from {}\n", get_address(&stream));

						break;
					}

					if send_error(&mut stream, &mut double_word, message).is_err() {
						break;
					}
				}
			}
		})?;
	}

	Ok(())
}