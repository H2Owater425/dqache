use std::{
	fs::{create_dir_all, exists, read, remove_file, write},
	path::PathBuf
};
use crate::{
	common::{ARGUMENT, Result},
	debug
};

pub struct Storage {
	root: PathBuf
}

impl Storage {
	pub fn new(root: &str) -> Result<Storage> {
		let root: PathBuf = PathBuf::from(root);

		create_dir_all(&root)?;

		Ok(Storage {
			root: root
		})
	}

	pub fn read(self: &Self, key: &str) -> Result<Option<String>> {
		let file: PathBuf = self.root.join(key);

		if ARGUMENT.is_verbose {
			debug!("read {:?} from {:?}\n", key, file.display());
		}

		Ok(if exists(&file)? {
			Some(String::from_utf8(read(&file)?)?)
		} else {
			None
		})
	}

	pub fn write(self: &Self, key: &str, value: String) -> Result<()> {
		let file: PathBuf = self.root.join(key);

		if ARGUMENT.is_verbose {
			debug!("wrote {:?} to {:?}\n", key, file.display());
		}

		Ok(write(&file, value)?)
	}

	pub fn delete(self: &Self, key: &str) -> Result<bool> {
		let file: PathBuf = self.root.join(key);

		if exists(&file)? {
			if ARGUMENT.is_verbose {
				debug!("deleted {:?} from {:?}\n", key, file.display());
			}

			remove_file(&file)?;

			Ok(true)
		} else {
			Ok(false)
		}
	}
}