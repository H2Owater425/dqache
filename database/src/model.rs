use ort::{
	execution_providers::{
		CPUExecutionProvider,
		CUDAExecutionProvider,
		CoreMLExecutionProvider,
		DirectMLExecutionProvider,
		ExecutionProvider,
		TensorRTExecutionProvider,
		XNNPACKExecutionProvider
	},
	session::{
		InMemorySession,
		Session,
		SessionOutputs,
		builder::{GraphOptimizationLevel, SessionBuilder}
	},
	value::Value
};
use std::{collections::HashMap, iter::zip};
use crate::{
	cache::{Entry, Evictor},
	common::{ARGUMENT, Result, log1p, unix_epoch},
	debug,
	info
};

#[derive(Debug, Clone, Copy)]
pub enum Model {
	DeepQNetwork,
	LeastRecentlyUsed,
	LeastFrequentlyUsed
}

pub struct DeepQNetwork<'a> {
	model: InMemorySession<'a>
}

impl<'a> DeepQNetwork<'a> {
	pub fn new() -> Result<Self>  {
		let mut session: SessionBuilder = Session::builder()?.with_optimization_level(GraphOptimizationLevel::Level3)?;

		info!("initializing model using DeepQNetwork on {}\n", if TensorRTExecutionProvider::default().register(&mut session).is_ok() {
			"TensorRT"
		} else if CUDAExecutionProvider::default().register(&mut session).is_ok() {
			"CUDA"
		} else if DirectMLExecutionProvider::default().register(&mut session).is_ok() {
			"DirectML"
		} else if CoreMLExecutionProvider::default().register(&mut session).is_ok() {
			"CoreML"
		} else if XNNPACKExecutionProvider::default().register(&mut session).is_ok() {
			"XNNPACK"
		} else {
			CPUExecutionProvider::default().register(&mut session)?;

			"CPU"
		});

		Ok(DeepQNetwork {
			model: session.commit_from_memory_directly(include_bytes!("../model.onnx"))?
		})
	}
}

impl<'a> Evictor for DeepQNetwork<'a> {
	fn select_victim(self: &mut Self, entries: &HashMap<String, Entry>) -> Result<String> {
		let length: usize = entries.len();

		if length == 0 {
			return Err(Box::from("entries length must be greater than 0"));
		}

		let mut keys: Vec<&String> = Vec::with_capacity(length);
		let mut inputs: Vec<f32> = Vec::with_capacity(length * 4);
		let capacity: f32 = log1p(entries.capacity() as u64);

		for entry in entries {
			keys.push(entry.0);
			inputs.push(log1p(unix_epoch()? - entry.1.accessed_at));
			inputs.push(log1p(entry.1.access_count));
			inputs.push(log1p(entry.1.value.len() as u64));
			inputs.push(capacity);
		}

		let output: SessionOutputs = self.model.run(vec![("args_0", Value::from_array((([length, 4]), inputs))?)])?;
		let output: &[f32] = output[0].try_extract_tensor::<f32>()?.1;

		let mut i: usize = 0;
		let mut minimum_score: f32 = f32::MAX;
		let mut minimum_index: usize = 0;

		if ARGUMENT.is_verbose {
			let mut key_scores: Vec<(&&String, &f32)> = zip(&keys, output).collect::<Vec<(&&String, &f32)>>();

			key_scores.sort_by(|a: &(&&String, &f32), b: &(&&String, &f32)| a.1.total_cmp(b.1));

			debug!("scored with {:#?}\n", key_scores);
		}

		for score in output {
			if *score < minimum_score {
				minimum_score = *score;
				minimum_index = i;
			}

			i += 1;
		}

		Ok(keys[minimum_index].clone())
	}
}

pub struct LeastRecentlyUsed {}

impl LeastRecentlyUsed {
	pub fn new() -> Self {
		info!("initializing model using LeastRecentlyUsed\n");

		LeastRecentlyUsed {}
	}
}

impl Evictor for LeastRecentlyUsed {
	fn select_victim(self: &mut Self, entries: &HashMap<String, Entry>) -> Result<String> {
		if entries.len() == 0 {
			return Err(Box::from("entries length must be greater than 0"));
		}

		let mut minimum_accessed_at: u64 = u64::MAX;
		let mut minimum_key: &String = &String::new();

		for entry in entries {
			if entry.1.accessed_at < minimum_accessed_at {
				minimum_accessed_at = entry.1.accessed_at;
				minimum_key = entry.0;
			}
		}

		Ok(minimum_key.clone())
	}
}

pub struct LeastFrequentlyUsed {}

impl LeastFrequentlyUsed {
	pub fn new() -> Self {
		info!("initializing model using LeastFrequentlyUsed\n");

		LeastFrequentlyUsed {}
	}
}

impl Evictor for LeastFrequentlyUsed {
	fn select_victim(self: &mut Self, entries: &HashMap<String, Entry>) -> Result<String> {
		if entries.len() == 0 {
			return Err(Box::from("entries length must be greater than 0"));
		}

		let mut minimum_access_count: u64 = u64::MAX;
		let mut minimum_key: &String = &String::new();

		for entry in entries {
			if entry.1.access_count < minimum_access_count {
				minimum_access_count = entry.1.access_count;
				minimum_key = entry.0;
			}
		}

		Ok(minimum_key.clone())
	}
}
