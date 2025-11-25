use std::{
	cmp::Ordering,
	error::Error,
	fmt::{Display, Formatter, Result as _Result},
	io::{IoSlice, Read, Write},
	net::TcpStream
};
use crate::{
	common::Result,
	error
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

pub const OPERATION_READY: &[u8] = &[0b10000000];
pub const OPERATION_HELLO: &[u8] = &[0b00000000];
pub const OPERATION_NOP: &[u8] = &[0b00000010];
pub const OPERATION_SET: &[u8] = &[0b00000011];
pub const OPERATION_DEL: &[u8] = &[0b00000100];
pub const OPERATION_GET: &[u8] = &[0b00000101];
pub const OPERATION_OK: &[u8] = &[0b10000010];
pub const OPERATION_VALUE: &[u8] = &[0b10000011];
pub const OPERATION_ERROR: &[u8] = &[0b10000100];
pub const OPERATION_QUIT: &[u8] = &[0b11111111];

pub fn read_string<const N: usize>(stream: &mut TcpStream, n_word/* I wasn't meaning that */: &mut [u8; N]) -> Result<String> {
	let mut buffer: Vec<u8>;

	if N == 2 {
		buffer = vec![0; n_word[1] as usize];
	} else if N == 4 {
		stream.read_exact(n_word)?;
		buffer = vec![0; (n_word[0] as usize) << 24 | (n_word[1] as usize) << 16 | (n_word[2] as usize) << 8 | n_word[3] as usize];
	} else {
		return Err(Box::from("length array size must be 1 or 4"));
	}

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

	pub fn major(self: &Self) -> u8 {
		self.major
	}

	pub fn minor(self: &Self) -> u8 {
		self.minor
	}

	pub fn patch(self: &Self) -> u8 {
		self.patch
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