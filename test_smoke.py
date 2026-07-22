import importlib.util

file_path = "canyon_final_v12_sp500_top_model_FINAL_RUNNABLE.py"

spec = importlib.util.spec_from_file_location("canyon_v12", file_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

res = mod.canyon_v12_offline_smoke_test(
    n_assets=50,
    n_days=220,
    seed=7
)

print("\n=== SMOKE TEST DONE ===")
print("version:", res.get("version"))
print("data quality passed:", res["data_quality"]["passed"])
print("alpha report shape:", res["alpha_report"].shape)
print("positions:", int((res["target_weights"].abs() > 1e-6).sum()))
print("gross:", round(float(res["target_weights"].abs().sum()), 4))
print("net:", round(float(res["target_weights"].sum()), 4))
print("self audit score:", res["self_audit"]["score"])
