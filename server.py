"""
カルマンフィルタ学習サーバー

Usage:
    pip install flask numpy
    python server.py
    → http://localhost:5000 をブラウザで開く
"""
from __future__ import annotations

import math
from flask import Flask, jsonify, request, send_from_directory
import numpy as np

app = Flask(__name__, static_folder=".")


# ── Kalman filter (1D, constant velocity model) ──────────────────────────────

def run_kalman(
    obs: list[float],
    R: float,
    Q: float,
    P0: float,
) -> dict:
    """
    1 次元カルマンフィルタ（定数モデル: x̂ₖ = x̂ₖ₋₁）

    Args:
        obs: 観測列
        R  : 観測ノイズ分散
        Q  : プロセスノイズ分散
        P0 : 初期推定誤差分散

    Returns:
        estimates, gains, Ps の辞書
    """
    x_est = float(obs[0])
    P = float(P0)
    estimates, gains, Ps = [], [], []

    for z in obs:
        # ① 予測
        P_pred = P + Q

        # ② カルマンゲイン
        K = P_pred / (P_pred + R)

        # ③ 更新
        x_est = x_est + K * (z - x_est)
        P = (1.0 - K) * P_pred

        estimates.append(x_est)
        gains.append(K)
        Ps.append(P)

    return {"estimates": estimates, "gains": gains, "Ps": Ps}


def generate_truth(N: int) -> np.ndarray:
    """真値（正弦波の重ね合わせ）"""
    t = np.linspace(0, 4 * math.pi, N)
    return np.sin(t) + 0.3 * np.sin(3 * t)


# ── エンドポイント ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/compute")
def compute():
    """
    カルマンフィルタを実行して結果を返す。

    Query params:
        R     : 観測ノイズ分散       (default 1.0)
        Q     : プロセスノイズ分散   (default 0.1)
        P0    : 初期推定誤差分散     (default 1.0)
        sigma : 観測ノイズの標準偏差  (default 1.5)
        seed  : 乱数シード           (default 42)
        N     : ステップ数           (default 120)
    """
    R     = float(request.args.get("R",     1.0))
    Q     = float(request.args.get("Q",     0.1))
    P0    = float(request.args.get("P0",    1.0))
    sigma = float(request.args.get("sigma", 1.5))
    seed  = int(request.args.get("seed",   42))
    N     = int(request.args.get("N",      120))

    # データ生成
    rng   = np.random.default_rng(seed)
    truth = generate_truth(N)
    obs   = (truth + rng.normal(0.0, sigma, N)).tolist()
    truth = truth.tolist()

    # カルマンフィルタ
    kf = run_kalman(obs, R, Q, P0)

    # RMSE
    rmse = math.sqrt(
        sum((e - x) ** 2 for e, x in zip(kf["estimates"], truth)) / N
    )

    return jsonify({
        "truth":     truth,
        "obs":       obs,
        "estimates": kf["estimates"],
        "gains":     kf["gains"],
        "Ps":        kf["Ps"],
        "rmse":      rmse,
    })


if __name__ == "__main__":
    print("http://localhost:5000 をブラウザで開いてください")
    app.run(debug=True, port=5000)
