use std::{
	collections::HashMap,
	fmt::{Debug, Formatter, Result as _Result}
};
use crate::{
	common::{ARGUMENT, Result, unix_epoch},
	debug,
	info,
	model::{DeepQNetwork, LeastFrequentlyUsed, LeastRecentlyUsed, Model}
};

pub struct Entry {
	pub value: String,
	pub accessed_at: u64,
	pub access_count: u64
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

impl Debug for Entry {
	fn fmt(self: &Self, formatter: &mut Formatter<'_>) -> _Result {
		formatter.debug_struct("")
			.field("size", &self.value.len())
			.field("accessed_at", &self.accessed_at)
			.field("access_count", &self.access_count)
			.finish()
	}
}

pub struct Cache {
	entries: HashMap<String, Entry>,
	model: Box<dyn Evictor + Send>,
	capacity: usize
}

impl Cache {
	pub fn new(model: Model, capacity: usize) -> Result<Cache> {
		info!("initializing cache with capacity of {}\n", capacity);

		Ok(Cache {
			entries: HashMap::with_capacity(capacity),
			model: match model {
				Model::DeepQNetwork => Box::new(DeepQNetwork::new()?),
				Model::LeastFrequentlyUsed => Box::new(LeastFrequentlyUsed::new()),
				Model::LeastRecentlyUsed => Box::new(LeastRecentlyUsed::new())
			},
			capacity: capacity
		})
	}

	pub fn set(self: &mut Self, key: &str, entry: Entry) -> Result<()> {
		let entries: String = if ARGUMENT.is_verbose {
			format!("{:#?}", self.entries)
		} else {
			String::new()
		};

		if let Some(old_entry) = self.entries.get_mut(key) {
			old_entry.value = entry.value;
			old_entry.accessed_at = entry.accessed_at;
			old_entry.access_count += entry.access_count;

			if ARGUMENT.is_verbose {
				debug!("set {:?}:{:#?} to {}\n", key, old_entry, entries);
			}
		} else {
			if self.entries.len() == self.capacity {
				let victim_key: String = self.model.select_victim(&self.entries)?;

				if let Some(old_entry) = self.entries.remove(&victim_key) {
					if ARGUMENT.is_verbose {
						debug!("evicted {:?}:{:#?} and set {:?}:{:#?} to {}\n", victim_key, old_entry, key, entry, entries);
					}
				}
			} else if ARGUMENT.is_verbose {
				debug!("set {:?}:{:#?} to {}\n", key, entry, entries);
			}

			self.entries.insert(key.to_owned(), entry);
		}

		Ok(())
	}

	pub fn get(self: &mut Self, key: &str) -> Result<Option<&Entry>> {
		let entries: String = if ARGUMENT.is_verbose {
			format!("{:#?}", self.entries)
		} else {
			String::new()
		};

		Ok(if let Some(entry) = self.entries.get_mut(key) {
			entry.access_count += 1;
			entry.accessed_at = unix_epoch()?;

			if ARGUMENT.is_verbose {
				debug!("get {:?} from {}\n", key, entries);
			}

			Some(entry)
		} else {
			None
		})
	}

	pub fn remove(self: &mut Self, key: &str) -> bool {
		if let Some(entry) = self.entries.remove(key) {
			if ARGUMENT.is_verbose {
				debug!("removed {:?}:{:#?} and became {:#?}\n", key, entry, self.entries);
			}

			true
		} else {
			false
		}
	}
}