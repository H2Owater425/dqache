use std::{
	fs::{create_dir_all, exists, read, remove_file, write},
	path::PathBuf
};
use crate::common::{Result, ARGUMENT};

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

	pub fn read(self: &Self, key: &str) -> Result<String> {
		let file: PathBuf = self.root.join(key);

		if ARGUMENT.is_verbose {
			print!("read {} from {}\n", key, file.display());
		}

		Ok(if exists(&file)? {
			String::from_utf8(read(&file)?)?
		} else {
			String::new()
		})
	}

	pub fn write(self: &Self, key: &str, value: String) -> Result<()> {
		let file: PathBuf = self.root.join(key);

		if ARGUMENT.is_verbose {
			print!("wrote {} from {}\n", key, file.display());
		}

		Ok(write(&file, value)?)
	}

	pub fn delete(self: &Self, key: &str) -> Result<()> {
		let file: PathBuf = self.root.join(key);

		if ARGUMENT.is_verbose {
			print!("deleted {} from {}\n", key, file.display());
		}

		if exists(&file)? {
			remove_file(&file)?
		}

		Ok(())
	}
}