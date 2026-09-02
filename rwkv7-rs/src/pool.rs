//! A minimal fixed-size thread pool for data-parallel loops.
//!
//! `parallel(f)` calls `f(worker_index)` on every worker and does not return until
//! all of them have finished.  Because the call blocks for the whole duration, the
//! closure is allowed to borrow from the caller's stack; we erase the lifetime
//! through a raw pointer and re-materialise it inside the workers.
//!
//! Rolling our own (instead of pulling in rayon) keeps the crate dependency-free,
//! and a persistent pool avoids paying ~30us of `thread::spawn` on every matmul.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};
use std::thread;

type Task = dyn Fn(usize) + Send + Sync;

/// Raw pointer to the currently published task. Only ever dereferenced while
/// `Pool::parallel` is blocked on the completion condvar, so the referent is live.
struct TaskPtr(*const Task);
unsafe impl Send for TaskPtr {}
unsafe impl Sync for TaskPtr {}

struct Slot {
    task: Option<TaskPtr>,
    /// Bumped for every published task so workers can tell "new work" from a spurious wakeup.
    epoch: u64,
    shutdown: bool,
}

pub struct Pool {
    threads: usize,
    slot: Mutex<Slot>,
    have_work: Condvar,
    work_done: Condvar,
    outstanding: Mutex<usize>,
}

static POOL: OnceLock<&'static Pool> = OnceLock::new();
static REQUESTED_THREADS: AtomicUsize = AtomicUsize::new(0);

/// Set the worker count. Must be called before the first `pool()` use to have effect.
pub fn set_threads(n: usize) {
    REQUESTED_THREADS.store(n, Ordering::SeqCst);
}

pub fn pool() -> &'static Pool {
    POOL.get_or_init(|| {
        let mut n = REQUESTED_THREADS.load(Ordering::SeqCst);
        if n == 0 {
            n = thread::available_parallelism().map(|v| v.get()).unwrap_or(1);
        }
        let n = n.max(1);

        // Leaked on purpose: the pool lives for the whole process, and a 'static
        // reference is what lets worker threads hold on to it without lifetimes.
        let pool: &'static Pool = Box::leak(Box::new(Pool {
            threads: n,
            slot: Mutex::new(Slot { task: None, epoch: 0, shutdown: false }),
            have_work: Condvar::new(),
            work_done: Condvar::new(),
            outstanding: Mutex::new(0),
        }));

        // Worker 0 is the calling thread, so we only spawn n-1 helpers.
        for idx in 1..n {
            thread::Builder::new()
                .name(format!("rwkv7-worker-{idx}"))
                .stack_size(1 << 20)
                .spawn(move || pool.worker_loop(idx))
                .expect("failed to spawn worker thread");
        }
        pool
    })
}

impl Pool {
    pub fn threads(&self) -> usize {
        self.threads
    }

    fn worker_loop(&'static self, idx: usize) {
        let mut seen: u64 = 0;
        loop {
            let task = {
                let mut slot = self.slot.lock().unwrap();
                while !slot.shutdown && slot.epoch == seen {
                    slot = self.have_work.wait(slot).unwrap();
                }
                if slot.shutdown {
                    return;
                }
                seen = slot.epoch;
                match &slot.task {
                    Some(TaskPtr(p)) => *p,
                    None => continue,
                }
            };

            // SAFETY: `parallel` publishes the pointer, then blocks until `outstanding`
            // drops to zero, which happens strictly after this call returns. The
            // referent therefore outlives every dereference.
            unsafe { (*task)(idx) };

            let mut left = self.outstanding.lock().unwrap();
            *left -= 1;
            if *left == 0 {
                self.work_done.notify_all();
            }
        }
    }

    /// Run `f(0..threads)` across the pool, blocking until every worker is done.
    pub fn parallel<F: Fn(usize) + Send + Sync>(&'static self, f: F) {
        if self.threads == 1 {
            f(0);
            return;
        }

        // Erase the closure's lifetime so it can be handed to the workers. This is
        // the whole reason `parallel` blocks below: the borrow must not outlive it.
        let dyn_ref: &(dyn Fn(usize) + Send + Sync) = &f;
        let raw: *const Task = unsafe {
            std::mem::transmute::<*const (dyn Fn(usize) + Send + Sync), *const Task>(
                dyn_ref as *const _,
            )
        };

        *self.outstanding.lock().unwrap() = self.threads - 1;
        {
            let mut slot = self.slot.lock().unwrap();
            slot.task = Some(TaskPtr(raw));
            slot.epoch += 1;
        }
        self.have_work.notify_all();

        // The calling thread pulls its share instead of idling.
        f(0);

        let mut left = self.outstanding.lock().unwrap();
        while *left != 0 {
            left = self.work_done.wait(left).unwrap();
        }
    }
}

/// Split `len` items into `parts` near-equal contiguous ranges and return range `i`.
#[inline]
pub fn chunk(len: usize, parts: usize, i: usize) -> (usize, usize) {
    let base = len / parts;
    let rem = len % parts;
    let start = base * i + i.min(rem);
    let end = start + base + if i < rem { 1 } else { 0 };
    (start, end)
}
