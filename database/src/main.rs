mod argument;
mod cache;
mod common;
mod model;
mod protocol;
mod storage;
mod thread_pool;
mod logger;

use std::{
	io::{Error, ErrorKind, IoSlice, Read, Write},
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
	cache::{Cache, Entry},
	common::{ARGUMENT, Result, get_address},
	protocol::{
		OPERATION_DEL,
		OPERATION_GET,
		OPERATION_HELLO,
		OPERATION_NOP,
		OPERATION_OK,
		OPERATION_QUIT,
		OPERATION_READY,
		OPERATION_SET,
		OPERATION_VALUE,
		Version,
		read_string,
		send_error
	},
	storage::Storage,
	thread_pool::ThreadPool
};

fn main() {
	if let Err(error) = (|| -> Result<()> {
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
						if let Some(error) = error.downcast_ref::<Error>() {
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
	})() {
		fatal!("{}\n", error);
	}
}