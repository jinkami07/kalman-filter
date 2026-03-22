"""
ロボット関節モータ — カルマンフィルタ推定サーバー

シナリオ:
  1軸関節をPD制御で正弦波軌道に追従させる。
  センサ（エンコーダ）は角度 θ のみ観測可能。
  カルマンフィルタで [θ, ω] を同時推定する。

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

# ── 固定システムパラメータ ────────────────────────────────────────────────────
DT      = 0.02   # サンプリング周期 [s]  (50 Hz)
J       = 0.1    # 慣性モーメント [kg·m²]
B_FRIC  = 0.5    # 粘性摩擦係数 [N·m·s/rad]
KP      = 10.0   # PD 比例ゲイン
KD      = 2.0    # PD 微分ゲイン
N       = 200    # ステップ数 (= 4 秒)
T_PERIOD = N * DT  # 目標軌道の周期 [s]


# ── シミュレーション ──────────────────────────────────────────────────────────

def simulate_joint(sigma_sensor: float, seed: int) -> dict:
    """
    PD制御による正弦波軌道追従シミュレーション。
    観測はエンコーダ（角度のみ）。
    """
    rng = np.random.default_rng(seed)
    t = np.arange(N) * DT

    # 目標軌道（1周期の正弦波）
    theta_ref = np.sin(2 * math.pi * t / T_PERIOD)
    omega_ref = (2 * math.pi / T_PERIOD) * np.cos(2 * math.pi * t / T_PERIOD)

    theta  = np.zeros(N)
    omega  = np.zeros(N)
    torque = np.zeros(N)

    for k in range(N - 1):
        u = KP * (theta_ref[k] - theta[k]) + KD * (omega_ref[k] - omega[k])
        u = float(np.clip(u, -5.0, 5.0))
        torque[k] = u

        # オイラー積分 + 微小プロセスノイズ
        alpha        = (u - B_FRIC * omega[k]) / J
        omega[k + 1] = omega[k] + DT * alpha + rng.normal(0.0, 0.02)
        theta[k + 1] = theta[k] + DT * omega[k]

    torque[-1] = torque[-2]

    # エンコーダ観測（角度のみ、ガウスノイズ付加）
    obs = theta + rng.normal(0.0, sigma_sensor, N)

    return {
        "theta":     theta.tolist(),
        "omega":     omega.tolist(),
        "theta_ref": theta_ref.tolist(),
        "omega_ref": omega_ref.tolist(),
        "obs":       obs.tolist(),
        "torque":    torque.tolist(),
        "t":         t.tolist(),
    }


# ── 2次元カルマンフィルタ ─────────────────────────────────────────────────────

def run_kalman_joint(
    obs: list[float],
    controls: list[float],
    R: float,
    Q_theta: float,
    Q_omega: float,
    P0: float,
) -> dict:
    """
    状態: x = [θ, ω]^T  (角度, 角速度)
    観測: z = θ (エンコーダ)
    入力: u = トルク指令 [N·m]

    システム行列:
        A = [[1,  dt           ],
             [0,  1-b*dt/J     ]]
        B = [0, dt/J]^T
        H = [1, 0]
    """
    A  = np.array([[1.0, DT],
                   [0.0, 1.0 - B_FRIC * DT / J]])
    Bm = np.array([0.0, DT / J])        # 制御入力行列 (1-D)
    H  = np.array([[1.0, 0.0]])         # 観測行列
    Q  = np.diag([Q_theta, Q_omega])    # プロセスノイズ共分散

    x = np.array([float(obs[0]), 0.0])  # 初期状態推定
    P = np.eye(2) * P0                  # 初期誤差共分散

    theta_est, omega_est = [], []
    K_theta_list, K_omega_list, P_theta_list = [], [], []

    for z, u in zip(obs, controls):
        # ① 予測
        x_pred = A @ x + Bm * u
        P_pred = A @ P @ A.T + Q

        # ② カルマンゲイン
        S = (H @ P_pred @ H.T).item() + R  # スカラー
        K = (P_pred @ H.T) / S             # shape (2, 1)

        # ③ 更新
        innov = z - (H @ x_pred).item()    # イノベーション（スカラー）
        x = x_pred + K.flatten() * innov
        P = (np.eye(2) - K @ H) @ P_pred

        theta_est.append(float(x[0]))
        omega_est.append(float(x[1]))
        K_theta_list.append(float(K[0, 0]))
        K_omega_list.append(float(K[1, 0]))
        P_theta_list.append(float(P[0, 0]))

    return {
        "theta_est": theta_est,
        "omega_est": omega_est,
        "K_theta":   K_theta_list,
        "K_omega":   K_omega_list,
        "P_theta":   P_theta_list,
    }


# ── エンドポイント ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/compute")
def compute():
    """
    Query params:
        R       : 観測ノイズ分散 (エンコーダ精度)     default 0.01
        Q_theta : 角度プロセスノイズ分散              default 0.001
        Q_omega : 角速度プロセスノイズ分散             default 0.1
        P0      : 初期推定誤差分散                    default 1.0
        sigma   : シミュレーション用センサノイズ標準偏差 default 0.05
        seed    : 乱数シード                         default 42
    """
    R       = float(request.args.get("R",       0.01))
    Q_theta = float(request.args.get("Q_theta", 0.001))
    Q_omega = float(request.args.get("Q_omega", 0.1))
    P0      = float(request.args.get("P0",      1.0))
    sigma   = float(request.args.get("sigma",   0.05))
    seed    = int(request.args.get("seed",      42))

    sim = simulate_joint(sigma, seed)
    kf  = run_kalman_joint(sim["obs"], sim["torque"], R, Q_theta, Q_omega, P0)

    # RMSE（角度）
    rmse_theta = math.sqrt(
        sum((e - x) ** 2 for e, x in zip(kf["theta_est"], sim["theta"])) / N
    )
    # RMSE（角速度）— 直接観測していないのに推定できる！
    rmse_omega = math.sqrt(
        sum((e - x) ** 2 for e, x in zip(kf["omega_est"], sim["omega"])) / N
    )

    return jsonify({
        "t":          sim["t"],
        "theta":      sim["theta"],
        "omega":      sim["omega"],
        "theta_ref":  sim["theta_ref"],
        "obs":        sim["obs"],
        "theta_est":  kf["theta_est"],
        "omega_est":  kf["omega_est"],
        "K_theta":    kf["K_theta"],
        "K_omega":    kf["K_omega"],
        "P_theta":    kf["P_theta"],
        "rmse_theta": rmse_theta,
        "rmse_omega": rmse_omega,
        "N":          N,
        "dt":         DT,
    })


if __name__ == "__main__":
    print("http://localhost:5000 をブラウザで開いてください")
    app.run(debug=True, port=5000)
