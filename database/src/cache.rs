use std::{collections::HashMap, fmt::{Debug, Formatter, Result as _Result}};

use crate::{common::{ARGUMENT, Result, unix_epoch}, model::{DeepQNetwork, LeastFrequentlyUsed, LeastRecentlyUsed}};

pub struct Entry {
	pub value: String,
	pub accessed_at: u64,
	pub access_count: u64
}

impl Debug for Entry {
	fn fmt(self: &Self, formatter: &mut Formatter<'_>) -> _Result {
			formatter.debug_struct("")
				.field("size", &self.value.len())
				.field("accessed_at", &self.accessed_at)
				.field("access_count", &self.access_count)
				.finish()
	}
}

impl Entry {
	pub fn new(value: &str) -> Result<Entry> {
		Ok(Entry {
			value: value.to_owned(),
			accessed_at: unix_epoch()?,
			access_count: 1
		})
	}
}

pub trait Evictor {
	fn select_victim(self: &mut Self, entries: &HashMap<String, Entry>) -> Result<String>;
}

#[derive(Debug, Clone, Copy)]
pub enum Model {
	DeepQNetwork,
	LeastRecentlyUsed,
	LeastFrequentlyUsed
}

pub struct Cache {
	entries: HashMap<String, Entry>,
	model: Box<dyn Evictor + Send>,
	capacity: usize
}

impl Cache {
	pub fn new(model: Model, capacity: usize) -> Result<Cache> {
		if ARGUMENT.is_verbose {
			println!("cache initialized with {:?} model and capacity of {}", model, capacity);
		}

		Ok(Cache {
			entries: HashMap::with_capacity(capacity),
			model: match model {
				Model::DeepQNetwork => Box::new(DeepQNetwork::new()?),
				Model::LeastFrequentlyUsed => Box::new(LeastFrequentlyUsed {}),
				Model::LeastRecentlyUsed => Box::new(LeastRecentlyUsed {})
			},
			capacity: capacity
		})
	}

	pub fn set(self: &mut Self, key: &str, entry: Entry) -> Result<()> {
		if let Some(old_entry) = self.entries.get_mut(key) {
			old_entry.value = entry.value;
			old_entry.accessed_at = entry.accessed_at;
			old_entry.access_count += entry.access_count;

			if ARGUMENT.is_verbose {
				print!("set {}{:?}", key, old_entry);
			}
		} else {
			if self.entries.len() == self.capacity {
				let victim_key: String = self.model.select_victim(&self.entries)?;

				if let Some(old_entry)  = self.entries.remove(&victim_key) {
					if ARGUMENT.is_verbose {
						print!("evicted {}{:?} to set {}{:?}", victim_key, old_entry, key, entry);
					}
				}
			} else if ARGUMENT.is_verbose {
				print!("set {}{:?}", key, entry);
			}

			self.entries.insert(key.to_owned(), entry);
		}

		if ARGUMENT.is_verbose {
			print!(" from {:?}\n", self.entries);
		}

		Ok(())
	}

	pub fn get(self: &mut Self, key: &str) -> Result<Option<&mut Entry>> {
		Ok(if let Some(entry) = self.entries.get_mut(key) {
			entry.access_count += 1;
			entry.accessed_at = unix_epoch()?;

			if ARGUMENT.is_verbose {
				print!("get {}{:?}\n", key, entry);
			}

			Some(entry)
		} else {
			None
		})
	}

	pub fn remove(self: &mut Self, key: &str) -> bool {
		if let Some(entry) = self.entries.remove(key) {
			if ARGUMENT.is_verbose {
				print!("removed {}{:?} from {:?}\n", key, entry, self.entries);
			}

			true
		} else {
			false
		}
	}
}