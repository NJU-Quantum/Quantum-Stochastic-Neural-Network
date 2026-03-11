import torch
import time
import models
import qsw

# 定义要对比的演化函数
evolve_methods = {
    'expm': qsw.evolve_expm,
    'rk4': qsw.evolve_vec_rk4,
    'krylov': qsw.evolve_from_operators
}

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda"

def benchmark_qsnn_function(N_in=20, T=1.0, num_runs=10):
    results = {}
    for name, evolve_func in evolve_methods.items():
        # 临时替换evolve函数
        original_evolve = models.evolve
        models.evolve = evolve_func

        forward_times = []
        backward_times = []
        for _ in range(num_runs):
            model = models.QSNNFunction(N_in=N_in, T=T, device=device)  # 每次重新初始化以避免梯度累积
            x = torch.randn(1, device=device)
            target = torch.randn(1, device=device)  # 随机目标

            # Forward
            start_time = time.time()
            yhat, rho_out = model(x)
            torch.cuda.synchronize() if device == "cuda" else None
            forward_end_time = time.time()
            forward_times.append(forward_end_time - start_time)

            # Backward
            loss = (yhat - target) ** 2
            start_time = time.time()
            loss.backward()
            torch.cuda.synchronize() if device == "cuda" else None
            backward_end_time = time.time()
            backward_times.append(backward_end_time - start_time)

        avg_forward = sum(forward_times) / len(forward_times)
        avg_backward = sum(backward_times) / len(backward_times)
        results[name] = {'forward': avg_forward, 'backward': avg_backward}
        print(f"{name}: 平均 forward 时间 {avg_forward:.4f} 秒, 平均 backward 时间 {avg_backward:.4f} 秒")

        # 恢复原始evolve
        models.evolve = original_evolve

    return results

def benchmark_qsnn_2d(N_in=20, T_u=1.0, T_d=1.0, num_runs=10):
    results = {}
    for name, evolve_func in evolve_methods.items():
        # 临时替换evolve函数
        original_evolve = models.evolve
        models.evolve = evolve_func

        forward_times = []
        backward_times = []
        for _ in range(num_runs):
            model = models.QSNN2D(N_in=N_in, T_u=T_u, T_d=T_d, device=device)  # 每次重新初始化
            xy = [torch.randn(1, device=device), torch.randn(1, device=device)]
            target = torch.tensor([0.5, 0.5], device=device)  # 固定目标概率

            # Forward
            start_time = time.time()
            probs, rho_out = model(xy)
            torch.cuda.synchronize() if device == "cuda" else None
            forward_end_time = time.time()
            forward_times.append(forward_end_time - start_time)

            # Backward
            loss = torch.nn.functional.mse_loss(probs, target)
            start_time = time.time()
            loss.backward()
            torch.cuda.synchronize() if device == "cuda" else None
            backward_end_time = time.time()
            backward_times.append(backward_end_time - start_time)

        avg_forward = sum(forward_times) / len(forward_times)
        avg_backward = sum(backward_times) / len(backward_times)
        results[name] = {'forward': avg_forward, 'backward': avg_backward}
        print(f"{name}: 平均 forward 时间 {avg_forward:.4f} 秒, 平均 backward 时间 {avg_backward:.4f} 秒")

        # 恢复原始evolve
        models.evolve = original_evolve

    return results

if __name__ == "__main__":
    print("device:", "cuda" if torch.cuda.is_available() else "cpu")

    print("Benchmarking QSNNFunction:")
    benchmark_qsnn_function()

    print("\nBenchmarking QSNN2D:")
    benchmark_qsnn_2d()