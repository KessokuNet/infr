//! `POST /v1/embeddings` — the OpenAI feature-extraction endpoint: one vector per input, from an
//! embedding model (EmbeddingGemma 2 — `docs/plan/22-embedding-models.md`).
//!
//! The vectors are the model's own pooled output: mean over the prompt's tokens, L2-normalized.
//! `dimensions` truncates to the leading entries (Matryoshka) and renormalizes, matching what the
//! model card requires of a shortened vector. An input is a text string or an `image_url` part
//! (a base64 `data:` URI), the same image shape a chat message carries.

use super::*;

/// Inputs one request may carry. Each one is a full forward over the whole prompt, and the slot
/// semaphore bounds how many requests run at once, not how much work a single request queues.
const MAX_EMBEDDING_INPUTS: usize = 256;

#[derive(Debug, Deserialize)]
pub(crate) struct EmbeddingsRequest {
    #[serde(default)]
    model: String,
    /// A string, an array of strings, or an array of content parts (`{"type":"text","text":…}`,
    /// `{"type":"image_url","image_url":{"url":"data:…"}}`). Token-id arrays are refused:
    /// embedding a caller's ids would skip the model's own `<bos>`/`<eos>` framing, which changes
    /// the vector.
    input: serde_json::Value,
    #[serde(default)]
    dimensions: Option<usize>,
    #[serde(default)]
    encoding_format: Option<String>,
}

/// One decoded input: a text to embed, or an image's bytes for the vision encoder.
pub(crate) enum EmbedItem {
    Text(String),
    Image(Arc<[u8]>),
}

impl EmbeddingsRequest {
    /// The inputs to embed, validated: non-empty strings or data-URI images, within the request's
    /// batch and image caps.
    fn items(&self) -> Result<Vec<EmbedItem>, ParamError> {
        let bad = |message: String| ParamError {
            param: "input",
            message,
        };
        let one = |v: &serde_json::Value| -> Result<EmbedItem, ParamError> {
            match v {
                serde_json::Value::String(s) if s.is_empty() => Err(bad(
                    "input: an empty string cannot be embedded".to_string(),
                )),
                serde_json::Value::String(s) => Ok(EmbedItem::Text(s.clone())),
                serde_json::Value::Number(_) | serde_json::Value::Array(_) => Err(bad(
                    "input: token ids are not supported — send text, so the model's own <bos>/<eos> \
                     framing is applied"
                        .to_string(),
                )),
                serde_json::Value::Object(_) => match v.get("type").and_then(|t| t.as_str()) {
                    Some("text") => match v.get("text").and_then(|t| t.as_str()) {
                        Some(t) if !t.is_empty() => Ok(EmbedItem::Text(t.to_string())),
                        _ => Err(bad("input: a text part needs a non-empty `text`".to_string())),
                    },
                    Some("image_url") | Some("image") => {
                        let url = v
                            .get("image_url")
                            .and_then(|u| u.get("url").or(Some(u)))
                            .and_then(|u| u.as_str())
                            .unwrap_or("");
                        Ok(EmbedItem::Image(decode_image_data_uri(url, "input")?))
                    }
                    // An OpenAI-style `image_url` object without a `type` is still unambiguous.
                    _ if v.get("image_url").is_some() => {
                        let url = v
                            .get("image_url")
                            .and_then(|u| u.get("url").or(Some(u)))
                            .and_then(|u| u.as_str())
                            .unwrap_or("");
                        Ok(EmbedItem::Image(decode_image_data_uri(url, "input")?))
                    }
                    _ => Err(bad(
                        "input: a part needs `type` `text` or `image_url`".to_string(),
                    )),
                },
                _ => Err(bad(
                    "input: expected a string, or an array of strings or content parts".to_string(),
                )),
            }
        };
        let items = match &self.input {
            serde_json::Value::String(_) | serde_json::Value::Object(_) => vec![one(&self.input)?],
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
                    "input: expected a string, or an array of strings or content parts".to_string(),
                ));
            }
        };
        let images = items
            .iter()
            .filter(|i| matches!(i, EmbedItem::Image(_)))
            .count();
        if images > infr_engine::MAX_REQUEST_IMAGES {
            return Err(bad(format!(
                "input: {images} images exceeds the {}-image cap",
                infr_engine::MAX_REQUEST_IMAGES
            )));
        }
        if self.dimensions == Some(0) {
            return Err(ParamError {
                param: "dimensions",
                message: "dimensions: must be at least 1".to_string(),
            });
        }
        Ok(items)
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
    let items = match req.items() {
        Ok(i) => i,
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
    // A model without a vision projector refuses images here, on the headers, before any work:
    // the same `validate_images` the chat route gates on.
    let images: Vec<Arc<[u8]>> = items
        .iter()
        .filter_map(|i| match i {
            EmbedItem::Image(b) => Some(b.clone()),
            EmbedItem::Text(_) => None,
        })
        .collect();
    if !images.is_empty() {
        if let Err(e) = engine.validate_images(&images) {
            return param_error(Some("input"), format!("input: {e}"));
        }
    }
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
        let mut out = Vec::with_capacity(items.len());
        for item in &items {
            out.push(match item {
                EmbedItem::Text(text) => engine.embed(text, dims)?,
                EmbedItem::Image(bytes) => engine.embed_image(bytes, dims)?,
            });
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

        fn validate_images(&self, _images: &[Arc<[u8]>]) -> anyhow::Result<()> {
            Ok(())
        }

        fn embed_image(
            &self,
            image: &[u8],
            dimensions: Option<usize>,
        ) -> anyhow::Result<(Vec<f32>, usize)> {
            // A marker distinct from the text path's, plus the byte length, so a test can tell
            // an image row from a text row.
            let mut v = vec![22.0, image.len() as f32, 0.0];
            if let Some(d) = dimensions {
                v.truncate(d);
            }
            Ok((v, image.len()))
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

    /// `data:image/png;base64,` + four bytes.
    const TINY_PNG_URI: &str = "data:image/png;base64,iVBORw0KGgo=";

    #[tokio::test]
    async fn an_image_input_is_embedded_through_the_vision_hook() {
        let body_json = serde_json::json!({
            "input": [{"type": "image_url", "image_url": {"url": TINY_PNG_URI}}]
        });
        let (status, body) = post(router(Arc::new(EmbedGen)), &body_json.to_string()).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        // The image marker (22.0) and the decoded byte count (8), not the text path's (11.0).
        assert_eq!(
            body["data"][0]["embedding"],
            serde_json::json!([22.0, 8.0, 0.0])
        );
        assert_eq!(body["usage"]["prompt_tokens"], 8);
    }

    #[tokio::test]
    async fn text_and_image_parts_keep_their_order() {
        let body_json = serde_json::json!({
            "input": [
                {"type": "text", "text": "hi"},
                {"type": "image_url", "image_url": {"url": TINY_PNG_URI}},
            ]
        });
        let (status, body) = post(router(Arc::new(EmbedGen)), &body_json.to_string()).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["data"][0]["embedding"][0], 11.0);
        assert_eq!(body["data"][1]["embedding"][0], 22.0);
    }

    #[tokio::test]
    async fn a_model_without_vision_refuses_an_image_before_any_work() {
        let body_json = serde_json::json!({
            "input": [{"type": "image_url", "image_url": {"url": TINY_PNG_URI}}]
        });
        let (status, body) = post(router(Arc::new(NoEmbedGen)), &body_json.to_string()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(body["error"]["param"], "input");
    }

    #[tokio::test]
    async fn a_malformed_image_data_uri_is_a_400() {
        for url in [
            "data:image/png;base64",
            "data:image/png,notbase64",
            "https://example.com/x.png",
        ] {
            let body_json = serde_json::json!({
                "input": [{"type": "image_url", "image_url": {"url": url}}]
            });
            let (status, body) = post(router(Arc::new(EmbedGen)), &body_json.to_string()).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{url}: {body}");
            assert_eq!(body["error"]["param"], "input", "{url}");
        }
    }
}
