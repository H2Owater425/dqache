use std::{
	cell::{RefCell, RefMut},
	io::{IoSlice, Write},
	sync::{Arc, LazyLock, Mutex, MutexGuard}
};
use crate::common::unix_epoch;

const FATAL_SLICE: LazyLock<IoSlice<'static>> = LazyLock::new(|| IoSlice::new(b"\x1b[31m FATAL \x1b[0m"));
const ERROR_SLICE: LazyLock<IoSlice<'static>> = LazyLock::new(|| IoSlice::new(b"\x1b[31m ERROR \x1b[0m"));
const WARN_SLICE: LazyLock<IoSlice<'static>> = LazyLock::new(|| IoSlice::new(b"\x1b[33m WARN \x1b[0m"));
const INFO_SLICE: LazyLock<IoSlice<'static>> = LazyLock::new(|| IoSlice::new(b"\x1b[32m INFO \x1b[0m"));
const DEBUG_SLICE: LazyLock<IoSlice<'static>> = LazyLock::new(|| IoSlice::new(b"\x1b[34m DEBUG \x1b[0m"));
//const TRACE_SLICE: LazyLock<IoSlice<'static>> = LazyLock::new(|| IoSlice::new(b"\x1b[36m TRACE \x1b[0m"));

struct Timestamp {
	last_second: u64,
	buffer: [u8; 19]
}

thread_local! {
	static TIMESTAMP: RefCell<Timestamp> = RefCell::new(Timestamp {
		last_second: 0,
		buffer: b"0000/00/00 00:00:00".to_owned()
	});
}

pub struct Logger {
	output: Arc<Mutex<dyn Write + Send>>,
	error: Arc<Mutex<dyn Write + Send>>,
	level: u8
}

impl Logger {
	pub fn new<T: Write + Send + 'static, U: Write + Send + 'static>(output: T, error: U, level: u8) -> Logger {
		Logger {
			output: Arc::new(Mutex::new(output)),
			error: Arc::new(Mutex::new(error)),
			level: level
		}
	}

	#[cfg(target_os = "windows")]
	fn write_vectored(self: &Self, level: u8, buffers: &[IoSlice<'_>]) {
		let mut writer: MutexGuard<'_, dyn Write + Send> = if level < 3 {
			self.error.lock()
				.unwrap()
		} else {
			self.output.lock()
				.unwrap()
		};

		for buffer in buffers {
			writer.write_all(&buffer)
				.unwrap();
		}

		writer.flush()
			.unwrap();
	}

	#[cfg(not(target_os = "windows"))]
	fn write_vectored(self: &Self, level: u8, buffers: &[IoSlice<'_>]) {
		let mut writer: MutexGuard<'_, dyn Write + Send> = if level < 3 {
			self.error.lock()
				.unwrap()
		} else {
			self.output.lock()
				.unwrap()
		};

		writer.write_vectored(buffers)
			.unwrap();
		writer.flush()
			.unwrap();
	}

	fn log(self: &Self, level: u8, message: &str) {
		if level > self.level {
			return;
		}

		TIMESTAMP.with(|timestamp: &RefCell<Timestamp>| {
			let mut timestamp: RefMut<Timestamp> = timestamp.borrow_mut();
			let current_second: u64 = unix_epoch().unwrap();

			if current_second != timestamp.last_second {
				let day_count: u64 = current_second / 86400 + 719468;
				let remainder: u64 = current_second % 86400;
				let hour: u64 = remainder / 3600;
				let minute: u64 = (remainder % 3600) / 60;
				let second: u64 = remainder % 60;

				timestamp.buffer[11] = (hour / 10) as u8 + 48;
				timestamp.buffer[12] = (hour % 10) as u8 + 48;
				timestamp.buffer[14] = (minute / 10) as u8 + 48;
				timestamp.buffer[15] = (minute % 10) as u8 + 48;
				timestamp.buffer[17] = (second / 10) as u8 + 48;
				timestamp.buffer[18] = (second % 10) as u8 + 48;

				// Hinnant's Algorithm
				let era: u64 = day_count / 146097;
				let day_of_era: u64 = day_count - era * 146097;
				let year_of_era: u64 = (day_of_era - day_of_era / 1460 + day_of_era / 36524 - day_of_era / 146096) / 365;

				let day_of_year: u64 = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
				let month_period: u64 = (5 * day_of_year + 2) / 153;

				let mut month: u64 = month_period;

				if month_period < 10 {
					month += 3;
				} else {
					month -= 9;
				}

				let day: u64 = day_of_year - (153 * month_period + 2) / 5 + 1;
				let mut year = year_of_era + era * 400;

				if month <= 2 {
					year += 1;
				}

				timestamp.buffer[0] = (year / 1000) as u8 + 48;

				year %= 1000;

				timestamp.buffer[1] = (year / 100) as u8 + 48;

				year %= 100;

				timestamp.buffer[2] = (year / 10) as u8 + 48;
				timestamp.buffer[3] = (year % 10) as u8 + 48;
				timestamp.buffer[5] = (month / 10) as u8 + 48;
				timestamp.buffer[6] = (month % 10) as u8 + 48;
				timestamp.buffer[8] = (day / 10) as u8 + 48;
				timestamp.buffer[9] = (day % 10) as u8 + 48;

				timestamp.last_second = current_second;
			}

			self.write_vectored(level, &[
				IoSlice::new(&timestamp.buffer),
				match level {
					1 => *FATAL_SLICE,
					2 => *ERROR_SLICE,
					3 => *WARN_SLICE,
					4 => *INFO_SLICE,
					5 => *DEBUG_SLICE,
					//6 => *TRACE_SLICE,
					_ => *INFO_SLICE
				},
				IoSlice::new(message.as_bytes())
			]);
		});
	}

	pub fn fatal(self: &Self, message: &str) {
		self.log(1, message);
	}

	pub fn error(self: &Self, message: &str) {
		self.log(2, message);
	}

	pub fn warn(self: &Self, message: &str) {
		self.log(3, message);
	}

	pub fn info(self: &Self, message: &str) {
		self.log(4, message);
	}

	pub fn debug(self: &Self, message: &str) {
		self.log(5, message);
	}

	//pub fn trace(self: &Self, message: &str) {
	//	self.log(6, message);
	//}
}