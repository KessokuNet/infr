//! `POST /v1/embeddings` — the OpenAI feature-extraction endpoint: one vector per input text, from
//! an embedding model (EmbeddingGemma 2 — `docs/plan/22-embedding-models.md`).
//!
//! The vectors are the model's own pooled output: mean over the prompt's tokens, L2-normalized.
//! `dimensions` truncates to the leading entries (Matryoshka) and renormalizes, matching what the
//! model card requires of a shortened vector.

use super::*;

/// Inputs one request may carry. Each one is a full forward over the whole prompt, and the slot
/// semaphore bounds how many requests run at once, not how much work a single request queues.
const MAX_EMBEDDING_INPUTS: usize = 256;

#[derive(Debug, Deserialize)]
pub(crate) struct EmbeddingsRequest {
    #[serde(default)]
    model: String,
    /// A string, or an array of strings. Token-id arrays are refused: embedding a caller's ids
    /// would skip the model's own `<bos>`/`<eos>` framing, which changes the vector.
    input: serde_json::Value,
    #[serde(default)]
    dimensions: Option<usize>,
    #[serde(default)]
    encoding_format: Option<String>,
}

impl EmbeddingsRequest {
    /// The texts to embed, validated: non-empty, string-only, and within the batch cap.
    fn texts(&self) -> Result<Vec<String>, ParamError> {
        let bad = |message: String| ParamError {
            param: "input",
            message,
        };
        let one = |v: &serde_json::Value| match v {
            serde_json::Value::String(s) if s.is_empty() => {
                Err(bad("input: an empty string cannot be embedded".to_string()))
            }
            serde_json::Value::String(s) => Ok(s.clone()),
            serde_json::Value::Number(_) | serde_json::Value::Array(_) => Err(bad(
                "input: token ids are not supported — send text, so the model's own <bos>/<eos> \
                 framing is applied"
                    .to_string(),
            )),
            _ => Err(bad(
                "input: expected a string or an array of strings".to_string()
            )),
        };
        let texts = match &self.input {
            serde_json::Value::String(_) => vec![one(&self.input)?],
            serde_json::Value::Array(items) => {
                if items.is_empty() {
                    return Err(bad("input: the array is empty".to_string()));
                }
                if items.len() > MAX_EMBEDDING_INPUTS {
                    return Err(bad(format!(
                        "input: {} inputs exceeds the per-request cap of {MAX_EMBEDDING_INPUTS}",
                        items.len()
                    )));
                }
                items.iter().map(one).collect::<Result<Vec<_>, _>>()?
            }
            _ => {
                return Err(bad(
                    "input: expected a string or an array of strings".to_string()
                ));
            }
        };
        if self.dimensions == Some(0) {
            return Err(ParamError {
                param: "dimensions",
                message: "dimensions: must be at least 1".to_string(),
            });
        }
        Ok(texts)
    }

    /// Whether the vectors go out as base64 of the little-endian f32 bytes (OpenAI's non-default
    /// `encoding_format`) rather than as a JSON array.
    fn base64_encoding(&self) -> Result<bool, ParamError> {
        match self.encoding_format.as_deref() {
            None | Some("float") => Ok(false),
            Some("base64") => Ok(true),
            Some(other) => Err(ParamError {
                param: "encoding_format",
                message: format!("encoding_format: `{other}` is not supported (float | base64)"),
            }),
        }
    }
}

/// One input's little-endian f32 bytes, base64'd.
fn base64_vector(v: &[f32]) -> String {
    use base64::Engine as _;
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

pub(crate) async fn embeddings_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Result<Json<EmbeddingsRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    if let Some(denied) = auth_gate(&state.cfg, &headers) {
        return denied;
    }
    let Json(req) = match body {
        Ok(j) => j,
        Err(e) => return param_error(None, e.body_text()),
    };
    let texts = match req.texts() {
        Ok(t) => t,
        Err(e) => return param_error(Some(e.param), e.message),
    };
    let base64 = match req.base64_encoding() {
        Ok(b) => b,
        Err(e) => return param_error(Some(e.param), e.message),
    };
    let entry = state.route(&req.model);
    let Some(engine) = entry.engine.clone() else {
        return json_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "embeddings: no model is loaded".to_string(),
        );
    };
    // One permit per request, the same bound generation uses: a CPU embedding forward would
    // otherwise pile a burst of requests onto the CPU at once.
    let permit = match entry.slots.clone().acquire_owned().await {
        Ok(p) => p,
        Err(_) => {
            return json_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "embeddings: the server is shutting down".to_string(),
            );
        }
    };
    let dims = req.dimensions;
    let started = std::time::Instant::now();
    let rows = tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<(Vec<f32>, usize)>> {
        let _permit = permit;
        let mut out = Vec::with_capacity(texts.len());
        for text in &texts {
            out.push(engine.embed(text, dims)?);
        }
        Ok(out)
    })
    .await;
    let rows = match rows {
        Ok(Ok(rows)) => rows,
        Ok(Err(e)) => {
            let message = format!("embeddings: {e}");
            return match e.downcast_ref::<ParamError>() {
                Some(p) => param_error(Some(p.param), p.message.clone()),
                None => json_error(StatusCode::INTERNAL_SERVER_ERROR, message),
            };
        }
        Err(e) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("embeddings: the worker panicked: {e}"),
            );
        }
    };
    let data: Vec<serde_json::Value> = rows
        .iter()
        .enumerate()
        .map(|(index, (vector, _))| {
            let embedding = if base64 {
                serde_json::Value::String(base64_vector(vector))
            } else {
                serde_json::json!(vector)
            };
            serde_json::json!({
                "object": "embedding",
                "index": index,
                "embedding": embedding,
            })
        })
        .collect();
    let prompt_tokens: usize = rows.iter().map(|(_, tokens)| *tokens).sum();
    tracing::debug!(
        inputs = rows.len(),
        prompt_tokens,
        dimensions = rows.first().map(|(v, _)| v.len()).unwrap_or(0),
        elapsed_ms = started.elapsed().as_millis() as u64,
        "embeddings: served"
    );
    let body = serde_json::json!({
        "object": "list",
        "data": data,
        "model": &*entry.id,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens,
        },
    });
    (StatusCode::OK, Json(body)).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    /// A generator that "embeds" by putting a fixed marker first and the text length second, so a
    /// test can tell which input produced which row without any model.
    struct EmbedGen;

    impl ChatGenerator for EmbedGen {
        fn status(&self) -> GeneratorStatus {
            GeneratorStatus {
                backend: "cpu",
                device: None,
                ctx_per_slot: None,
                memory: None,
                kv_pages: None,
            }
        }

        fn vocab_size(&self) -> usize {
            3
        }

        fn embed(
            &self,
            text: &str,
            dimensions: Option<usize>,
        ) -> anyhow::Result<(Vec<f32>, usize)> {
            let mut v = vec![11.0, text.len() as f32, 0.0];
            if let Some(d) = dimensions {
                v.truncate(d);
            }
            Ok((v, text.len()))
        }

        fn chat(
            &self,
            _messages: &[infr_engine::ChatMessage],
            _tools: Option<&serde_json::Value>,
            _tool_choice: Option<&str>,
            _params: &GenParams,
            _cancel: &Arc<AtomicBool>,
            _on_delta: &mut dyn FnMut(infr_engine::Delta),
        ) -> anyhow::Result<ChatOutcome> {
            unreachable!("embedding-only mock")
        }
    }

    /// A generator that keeps the trait's default `embed`, which refuses.
    struct NoEmbedGen;

    impl ChatGenerator for NoEmbedGen {
        fn status(&self) -> GeneratorStatus {
            GeneratorStatus {
                backend: "cpu",
                device: None,
                ctx_per_slot: None,
                memory: None,
                kv_pages: None,
            }
        }

        fn vocab_size(&self) -> usize {
            3
        }

        fn chat(
            &self,
            _messages: &[infr_engine::ChatMessage],
            _tools: Option<&serde_json::Value>,
            _tool_choice: Option<&str>,
            _params: &GenParams,
            _cancel: &Arc<AtomicBool>,
            _on_delta: &mut dyn FnMut(infr_engine::Delta),
        ) -> anyhow::Result<ChatOutcome> {
            unreachable!("embedding-only mock")
        }
    }

    fn router(gen: Arc<dyn ChatGenerator>) -> Router {
        build_router(AppState::new(
            gen,
            "embed-model",
            1,
            Arc::new(Config::default()),
        ))
    }

    async fn post(router: Router, body: &str) -> (StatusCode, serde_json::Value) {
        let resp = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/embeddings")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_owned()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .unwrap();
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null),
        )
    }

    #[tokio::test]
    async fn a_string_input_is_one_embedding_with_usage() {
        let (status, body) = post(router(Arc::new(EmbedGen)), r#"{"input":"hello"}"#).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["object"], "list");
        assert_eq!(body["model"], "embed-model");
        assert_eq!(body["data"][0]["object"], "embedding");
        assert_eq!(body["data"][0]["index"], 0);
        assert_eq!(
            body["data"][0]["embedding"],
            serde_json::json!([11.0, 5.0, 0.0])
        );
        assert_eq!(body["usage"]["prompt_tokens"], 5);
        assert_eq!(body["usage"]["total_tokens"], 5);
    }

    #[tokio::test]
    async fn an_array_keeps_its_order_and_sums_the_token_usage() {
        let (status, body) = post(router(Arc::new(EmbedGen)), r#"{"input":["a","bcd"]}"#).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["data"][0]["index"], 0);
        assert_eq!(body["data"][1]["index"], 1);
        assert_eq!(body["data"][0]["embedding"][1], 1.0);
        assert_eq!(body["data"][1]["embedding"][1], 3.0);
        assert_eq!(body["usage"]["prompt_tokens"], 4);
    }

    #[tokio::test]
    async fn dimensions_truncate_through_the_generator() {
        let (status, body) = post(
            router(Arc::new(EmbedGen)),
            r#"{"input":"hello","dimensions":2}"#,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["data"][0]["embedding"], serde_json::json!([11.0, 5.0]));
    }

    #[tokio::test]
    async fn base64_encoding_is_the_little_endian_f32_bytes() {
        use base64::Engine as _;
        let (status, body) = post(
            router(Arc::new(EmbedGen)),
            r#"{"input":"hello","encoding_format":"base64"}"#,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let encoded = body["data"][0]["embedding"].as_str().unwrap();
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .unwrap();
        let floats: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        assert_eq!(floats, vec![11.0, 5.0, 0.0]);
    }

    #[tokio::test]
    async fn bad_requests_name_the_offending_parameter() {
        for (body, param) in [
            (r#"{"input":[]}"#, "input"),
            (r#"{"input":[1,2,3]}"#, "input"),
            (r#"{"input":""}"#, "input"),
            (r#"{"input":"x","dimensions":0}"#, "dimensions"),
            (
                r#"{"input":"x","encoding_format":"hex"}"#,
                "encoding_format",
            ),
        ] {
            let (status, resp) = post(router(Arc::new(EmbedGen)), body).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
            assert_eq!(resp["error"]["param"], param, "{body}");
        }
    }

    #[tokio::test]
    async fn a_model_without_embeddings_is_a_500() {
        let (status, body) = post(router(Arc::new(NoEmbedGen)), r#"{"input":"x"}"#).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        let message = body["error"]["message"].as_str().unwrap_or_default();
        assert!(message.contains("does not support embeddings"), "{message}");
    }
}
