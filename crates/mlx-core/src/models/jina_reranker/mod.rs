//! Inference-only listwise ranking. No chat sessions or process-global model state.
pub(crate) mod config;
mod inference;
pub(crate) mod weights;
use crate::{
    array,
    model_thread::{ModelThread, ResponseTx},
};
use config::{Options, invalid};
use inference::{Model, Request};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::{
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};
use tokio::sync::oneshot;

enum Command {
    Rank {
        request: Request,
        cancel: Arc<AtomicBool>,
        reply: ResponseTx<String>,
    },
}
struct Handle {
    thread: Mutex<Option<ModelThread<Command>>>,
    shutdown: Arc<AtomicBool>,
    disposal: tokio::sync::Mutex<()>,
}
#[napi]
pub struct JinaRerankerModel {
    handle: Arc<Handle>,
    info: String,
}
#[napi]
impl JinaRerankerModel {
    #[napi]
    pub fn load<'env>(
        env: &'env Env,
        model_path: String,
        options_json: Option<String>,
    ) -> Result<PromiseRaw<'env, JinaRerankerModel>> {
        let options: Options =
            serde_json::from_str(options_json.as_deref().unwrap_or("{}")).map_err(invalid)?;
        env.spawn_future(async move {
            let shutdown = Arc::new(AtomicBool::new(false));
            let stopping = shutdown.clone();
            let (thread, ready) = ModelThread::spawn_with_init(
                move || {
                    let model = Model::load(&PathBuf::from(model_path), options)?;
                    let info = serde_json::to_string(&model.info).map_err(invalid)?;
                    Ok((model, info))
                },
                move |model, command| match command {
                    Command::Rank {
                        request,
                        cancel,
                        reply,
                    } => {
                        let result = model
                            .rerank(request, &cancel, &stopping)
                            .and_then(|value| serde_json::to_string(&value).map_err(invalid));
                        // Completion is a resource boundary, including on cancellation/error.
                        array::synchronize();
                        let _ = reply.send(result);
                    }
                },
            );
            let info = await_reply(ready).await?;
            Ok(Self {
                handle: Arc::new(Handle {
                    thread: Mutex::new(Some(thread)),
                    shutdown,
                    disposal: tokio::sync::Mutex::new(()),
                }),
                info,
            })
        })
    }
    #[napi(getter)]
    pub fn info_json(&self) -> String {
        self.info.clone()
    }
    #[napi]
    pub fn start_rerank(&self, request_json: String) -> Result<JinaRerankerTask> {
        let request: Request = serde_json::from_str(&request_json)
            .map_err(|e| Error::from_reason(format!("rerank_invalid_request: {e}")))?;
        let thread = self
            .handle
            .thread
            .lock()
            .map_err(|_| invalid("thread lock poisoned"))?;
        if self.handle.shutdown.load(Ordering::Acquire) {
            return Err(Error::from_reason("rerank_closed: model disposed"));
        }
        let cancel = Arc::new(AtomicBool::new(false));
        let (reply, receiver) = oneshot::channel();
        thread
            .as_ref()
            .ok_or_else(|| Error::from_reason("rerank_closed: model disposed"))?
            .send(Command::Rank {
                request,
                cancel: cancel.clone(),
                reply,
            })?;
        Ok(JinaRerankerTask {
            cancel,
            receiver: Mutex::new(Some(receiver)),
        })
    }
    #[napi]
    pub fn dispose<'env>(&self, env: &'env Env) -> Result<PromiseRaw<'env, ()>> {
        let handle = self.handle.clone();
        handle.shutdown.store(true, Ordering::Release);
        env.spawn_future(async move {
            let _guard = handle.disposal.lock().await;
            let thread = handle
                .thread
                .lock()
                .map_err(|_| invalid("thread lock poisoned"))?
                .take();
            if let Some(mut thread) = thread {
                spawn_blocking(move || thread.shutdown_and_join().map_err(Error::from_reason))
                    .await
                    .map_err(|e| Error::from_reason(e.to_string()))??;
            }
            Ok(())
        })
    }
}
impl Drop for JinaRerankerModel {
    fn drop(&mut self) {
        self.handle.shutdown.store(true, Ordering::Release);
    }
}

#[napi]
pub struct JinaRerankerTask {
    cancel: Arc<AtomicBool>,
    receiver: Mutex<Option<oneshot::Receiver<Result<String>>>>,
}
#[napi]
impl JinaRerankerTask {
    #[napi]
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Release);
    }
    #[napi]
    pub fn result<'env>(&self, env: &'env Env) -> Result<PromiseRaw<'env, String>> {
        let rx = self
            .receiver
            .lock()
            .map_err(|_| invalid("task lock poisoned"))?
            .take()
            .ok_or_else(|| invalid("task result already consumed"))?;
        env.spawn_future(await_reply(rx))
    }
}
impl Drop for JinaRerankerTask {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Release);
    }
}
async fn await_reply<T>(rx: oneshot::Receiver<Result<T>>) -> Result<T> {
    rx.await
        .map_err(|_| Error::from_reason("rerank_worker_failed: model thread exited"))?
}
