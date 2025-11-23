mod argument;
mod cache;
mod common;
mod model;
mod protocol;
mod storage;
mod thread_pool;

use std::{
	io::{Error, ErrorKind, Read, Write},
	net::{Ipv4Addr, TcpListener, TcpStream},
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
	cache::{Cache, Entry},
	common::{Result, ARGUMENT},
	protocol::{
		read_string,
		send_error,
		Version,
		OPERATION_DEL,
		OPERATION_GET,
		OPERATION_HELLO,
		OPERATION_NOP,
		OPERATION_OK,
		OPERATION_QUIT,
		OPERATION_READY,
		OPERATION_SET,
		OPERATION_VALUE
	},
	storage::Storage,
	thread_pool::ThreadPool
};

fn main() -> Result<()> {
	print!("starting qache {} on {}\n", ARGUMENT.version, ARGUMENT.platform);

	let cache: Arc<Mutex<Cache>> = Arc::new(Mutex::new(Cache::new(ARGUMENT.model, ARGUMENT.capacity)?));
	let storage: Arc<RwLock<Storage>> = Arc::new(RwLock::new(Storage::new(&ARGUMENT.directory)?));
	let thread_pool: ThreadPool = ThreadPool::new(available_parallelism()?.get())?;
	let listener: TcpListener = TcpListener::bind((Ipv4Addr::new(0, 0, 0, 0), ARGUMENT.port))?;

	print!("lisening on 0.0.0.0:{}\n", ARGUMENT.port);

	for stream in listener.incoming() {
		let mut stream: TcpStream = stream?;
		let cache: Arc<Mutex<Cache>> = cache.clone();
		let storage: Arc<RwLock<Storage>> = storage.clone();

		stream.set_read_timeout(Some(Duration::from_mins(1)))?;

		thread_pool.execute(move || {
			if let Err(error) = (|| -> Result<()> {
				stream.write(OPERATION_READY)?;

				let mut handshake: [u8; 4] = [0, 0, 0, 0];

				stream.read_exact(&mut handshake)?;

				if handshake[0] != OPERATION_HELLO[0] {
					return Err(Box::from("handshake must start with HELLO operation"));
				}

				if let Ok(version) = Version::try_from(&handshake[1..4]) {
					if version > ARGUMENT.version {
						return Err(Box::from(format!("client version must be less than or equal to {}", ARGUMENT.version)));
					}

					print!("client connected with {} from {}\n", version, stream.peer_addr()?);
				} else {
					return Err(Box::from("client version must be invalid\n"));
				}

				stream.write(OPERATION_OK)?;

				Ok(())
			})() {
				let _ = send_error(&mut stream, error.to_string());

				return;
			}

			let mut operation: [u8; 1] = [0];
			let mut key_length: [u8; 1] = [0];
			let mut value_length: [u8; 4] = [0, 0, 0, 0];

			loop {
				if let Err(error) = (|| -> Result<()> {
					stream.read_exact(&mut operation)?;

					match &operation {
						OPERATION_SET => {
							let key: String = read_string::<1>(&mut stream, &mut key_length)?;
							let value: String = read_string::<4>(&mut stream, &mut value_length)?;

							cache.lock()
								.map_err(|error: PoisonError<MutexGuard<'_, Cache>>| error.to_string())?
								.set(&key, Entry::new(&value)?)?;
							storage.write()
								.map_err(|error: PoisonError<RwLockWriteGuard<'_, Storage>>| error.to_string())?
								.write(&key, value)?;

							stream.write(OPERATION_OK)?;
						},
						OPERATION_DEL => {
							let key: String = read_string::<1>(&mut stream, &mut key_length)?;

							cache.lock()
								.map_err(|error: PoisonError<MutexGuard<'_, Cache>>| error.to_string())?
								.remove(&key);
							storage.write()
								.map_err(|error: PoisonError<RwLockWriteGuard<'_, Storage>>| error.to_string())?
								.delete(&key)?;

							stream.write(OPERATION_OK)?;
						},
						OPERATION_GET => {
							let key: String = read_string::<1>(&mut stream, &mut key_length)?;
							let (is_cached, value): (bool, String) = if let Some(entry) = cache.lock()
								.map_err(|error: PoisonError<MutexGuard<'_, Cache>>| error.to_string())?
								.get(&key)? {
								(true, entry.value.clone())
							} else {
								(false, storage.read()
									.map_err(|error: PoisonError<RwLockReadGuard<'_, Storage>>| error.to_string())?
									.read(&key)?)
							};

							if !is_cached && value.len() != 0 {
								cache.lock()
									.map_err(|error: PoisonError<MutexGuard<'_, Cache>>| error.to_string())?
									.set(&key, Entry::new(&value)?)?;
							}

							let value_length: usize = value.len();

							stream.write(&[OPERATION_VALUE[0], (value_length >> 24) as u8, (value_length >> 16) as u8, (value_length >> 8) as u8, value_length as u8])?;
							stream.write_all(value.as_bytes())?;
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
					if let Some(error) = error.downcast_ref::<Error>() {
						let _ = send_error(&mut stream, match error.kind() {
							ErrorKind::UnexpectedEof => {
								print!("client terminated from {}\n", if let Ok(address) = stream.peer_addr() {
									address.to_string()
								} else {
									"unknown".to_owned()
								});

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
						print!("client disconnected from {}\n", if let Ok(address) = stream.peer_addr() {
							address.to_string()
						} else {
							"unknown".to_owned()
						});

						break;
					}

					if send_error(&mut stream, message).is_err() {
						break;
					}
				}
			}
		})?;
	}

	Ok(())
}