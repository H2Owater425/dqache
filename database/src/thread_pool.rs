use std::{
	sync::{
		mpsc::{channel, Receiver, SendError, Sender},
		Arc,
		Mutex
	},
	thread::{spawn, JoinHandle}
};
use crate::common::{Job, Result, ARGUMENT};

pub struct ThreadPool {
	threads: Vec<JoinHandle<()>>,
	sender: Option<Sender<Job>>
}

impl ThreadPool {
	pub fn new(size: usize) -> Result<ThreadPool> {
		if size == 0 {
			return Err(Box::from("size must be greater than zero"));
		}

		let (sender, receiver): (Sender<Job>, Receiver<Job>) = channel();
		let receiver: Arc<Mutex<Receiver<Job>>> = Arc::new(Mutex::new(receiver));

		let mut threads: Vec<JoinHandle<()>> = Vec::with_capacity(size);

		for id in 0..size {
			let receiver: Arc<Mutex<Receiver<Job>>> = receiver.clone();

			threads.push(spawn(move || loop {
				let job: Job = if let Ok(job) = (match receiver.lock() {
					Ok(guard) => guard,
					Err(error) => {
						if ARGUMENT.is_verbose {
							eprint!("{} from thread {}\n", error, id);
						}

						break; // break if lock is poisoned (extremely rare)
					}
				}).recv() {
					job
				} else {
					if ARGUMENT.is_verbose {
						print!("thread {} shutdown\n", id);
					}

					break;
				};

				if ARGUMENT.is_verbose {
					print!("thread {} got job\n", id);
				}

				job();

				if ARGUMENT.is_verbose {
					print!("thread {} finished job\n", id);
				}
			}));
		}

		Ok(ThreadPool {
			threads: threads,
			sender: Some(sender)
		})
	}

	pub fn execute<F>(self: &Self, function: F) -> Result<(), SendError<Job>> where F: FnOnce() + Send + 'static {
		if let Some(sender) = &self.sender {
			sender.send(Box::new(function))?;
		}

		Ok(())
	}
}

impl Drop for ThreadPool {
	fn drop(self: &mut Self) {
		drop(self.sender.take());

		while let Some(thread) = self.threads.pop() {
			thread.join().unwrap();
		}
	}
}