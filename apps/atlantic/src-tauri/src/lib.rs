use serde::Serialize;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RuntimeStatus {
    stage: &'static str,
    inference_ready: bool,
    judge_ready: bool,
    trainer_ready: bool,
    reason: &'static str,
}

#[tauri::command]
fn runtime_status() -> RuntimeStatus {
    RuntimeStatus {
        stage: "design-preview",
        inference_ready: false,
        judge_ready: false,
        trainer_ready: false,
        reason: "Native inference, the exact judge encoder and platform services are not linked to this shell yet.",
    }
}

// Every future turn command must call the common gated C++ host.
// Do not add a raw-completion command or an HTTP server alongside it.
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![runtime_status])
        .run(tauri::generate_context!())
        .expect("Atlantic could not start");
}
