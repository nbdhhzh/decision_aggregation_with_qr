import numpy as np
from scipy.special import comb
from scipy.optimize import bisect, minimize, minimize_scalar
import matplotlib.pyplot as plt

eps=1e-15

def calc_gn(n):
    
    def L(m, beta, q):
        return (q * (1 - q))**m * (1 - 2 * q) / (1 / (1 + np.exp(-beta)) - q)

    # 定义 R 函数
    def R(m, beta, q):
        return L(m, beta, q) * (beta + np.log(1 - q) - np.log(q)) / (beta - np.log(1 - q) + np.log(q))

    def calc_q0_q1(m, beta):
        q1 = minimize_scalar(lambda q1, m=m, beta=beta: -L(m, beta, q1), 
                            bounds=(1 / (1 + np.exp(beta)), 0.5), method='bounded', options={'xatol': 1e-18}).x
        q0 = minimize_scalar(lambda q0, m=m, beta=beta: R(m, beta, q0), 
                            bounds=(1 / (1 + np.exp(beta)), q1), method='bounded', options={'xatol': 1e-18}).x
        return q0, q1, R(m, beta, q0) - L(m, beta, q1)

    def calc_beta(n):
        m = (n - 1) // 2
        beta_root = bisect(lambda beta: calc_q0_q1(m, beta)[2], 0.1, 10, xtol=1e-15)
        return beta_root
    if n <= 2:
        return 1e9
    return calc_beta(n)

def plot_gn_curve_final():
    n_values = range(3, 21)
    gn_values = [calc_gn(n) for n in n_values]

    plt.figure(figsize=(5, 3.5))  # Making the figure smaller
    plt.plot(n_values, gn_values, label=r"$g(n)$", zorder=5, marker='o')  # Adding points

    # Marking the odd points
    for n, gn in zip(n_values, gn_values):
        if n % 2 == 1:
            # Move the text label to the right of the point, between odd and even points
            plt.text(n + 0.5, gn + 0.25, f'{gn:.2f}', ha='center', va='center', fontsize=12, color='black')

    plt.xlabel("n", fontsize=13)
    plt.ylabel(r"$\beta$", fontsize=13)  # Labeling the y-axis as beta
    plt.xticks([n for n in n_values if n % 2 == 1], fontsize=12)  # Only showing odd values on x-axis
    plt.yticks(np.arange(0, 7, 1), fontsize=12)  # Only showing odd values on x-axis
    # plt.ylim(0, 6)
    # plt.yticks(np.arange(0, 6.5, 0.5))  # Y-axis from 0 to 6
    # plt.grid(True)
    plt.tight_layout()
    
    # Adding legend
    plt.legend()

    # Save the figure to a PDF file
    plt.savefig("gn_20.pdf", bbox_inches='tight')
    plt.savefig("gn_20.png", bbox_inches='tight')
    plt.show()


def loss(x, y):
    return x * y + (1 - x) * (1 - y)

def calc_pq_prime(beta, mu, p, q):
    if (mu * p + (1 - mu) * q == 0):
        p1 = 0.5
    elif abs(-beta * (mu * p - (1 - mu) * q) / (mu * p + (1 - mu) * q)) < 300:
        p1 = 1 / (1 + np.exp(-beta * (mu * p - (1 - mu) * q) / (mu * p + (1 - mu) * q)))
    else:
        p1 = 0 if -beta * (mu * p - (1 - mu) * q) / (mu * p + (1 - mu) * q) > 0 else 1
    if ((mu * (1 - p) + (1 - mu) * (1 - q)) == 0):
        p2 = 0.5
    elif abs(-beta * (mu * (1 - p) - (1 - mu) * (1 - q)) / (mu * (1 - p) + (1 - mu) * (1 - q))) < 300:
        p2 = 1 / (1 + np.exp(-beta * (mu * (1 - p) - (1 - mu) * (1 - q)) / (mu * (1 - p) + (1 - mu) * (1 - q))))
    else:
        p2 = 0 if -beta * (mu * (1 - p) - (1 - mu) * (1 - q)) / (mu * (1 - p) + (1 - mu) * (1 - q)) > 0 else 1
    p_prime = p * p1 + (1 - p) * p2
    q_prime = q * p1 + (1 - q) * p2
    return p_prime, q_prime

def majority(n, beta, mu, p, q, k=None):
    p_prime, q_prime = calc_pq_prime(beta, mu, p, q)
    # print(p_prime, q_prime)
    if (k is None) or (k >= n / 2):
        k = 1 - ((n % 2) / 2)
    else:
        k = n / 2 - k
    g_val = 0
    for i in range(n + 1):
        bin_coeff = comb(n, i)
        g_i = (i - n / 2 + k) / k / 2
        # print(i, n, k, g_i)
        g_i = min(max(g_i, 0), 1)

        # if (i * 2 == n):
        #     g_i = .5
        # else:
        #     g_i = 0 if i < n / 2 else 1
        #     if (abs(i * 2 - n) <= 2 * k):
        #         g_i = 1 - g_i
        denom = mu * p_prime**i * (1 - p_prime)**(n - i) + (1 - mu) * q_prime**i * (1 - q_prime)**(n - i)
        if denom == 0:
            continue
        num = mu * p_prime**i * (1 - p_prime)**(n - i)
        g_val += bin_coeff * denom * loss(num / denom, g_i)
    
    return g_val

def randomfollow(n, beta, mu, p, q):
    p_prime, q_prime = calc_pq_prime(beta, mu, p, q)
    
    g_val = 0
    for i in range(n + 1):
        bin_coeff = comb(n, i)
        g_i = i / n
        denom = mu * p_prime**i * (1 - p_prime)**(n - i) + (1 - mu) * q_prime**i * (1 - q_prime)**(n - i)
        if denom == 0:
            continue
        num = mu * p_prime**i * (1 - p_prime)**(n - i)
        g_val += bin_coeff * denom * loss(num / denom, g_i)
    
    return g_val

def minimum(n, beta, mu, p, q):
    p_prime, q_prime = calc_pq_prime(beta, mu, p, q)
    
    g_val = 0
    for i in range(n + 1):
        bin_coeff = comb(n, i)
        denom = mu * p_prime**i * (1 - p_prime)**(n - i) + (1 - mu) * q_prime**i * (1 - q_prime)**(n - i)
        num = mu * p_prime**i * (1 - p_prime)**(n - i)
        num2 = mu * p_prime**(n - i) * (1 - p_prime)**i
        denom2 = mu * p_prime**(n - i) * (1 - p_prime)**i + (1 - mu) * q_prime**(n - i) * (1 - q_prime)**i
        g_i = 1 if (2 * num - denom + denom2 - 2 * num2 > eps) else (0 if (2 * num - denom + denom2 - 2 * num2 < -eps) else .5)
        # if i * 2 <= n:
        #     g_i = min(g_i, .5)
        # if i * 2 >= n:
        #     g_i = max(g_i, .5)
        if denom == 0:
            continue
        g_val += bin_coeff * denom * loss(num / denom, g_i)
    return g_val

def minimum_s(n, beta, mu, p, q):
    p_prime, q_prime = calc_pq_prime(beta, mu, p, q)
    
    g_val = 0
    for i in range(n + 1):
        bin_coeff = comb(n, i)
        denom = mu * p_prime**i * (1 - p_prime)**(n - i) + (1 - mu) * q_prime**i * (1 - q_prime)**(n - i)
        num = mu * p_prime**i * (1 - p_prime)**(n - i)
        # num2 = mu * p_prime**(n - i) * (1 - p_prime)**i
        # denom2 = mu * p_prime**(n - i) * (1 - p_prime)**i + (1 - mu) * q_prime**(n - i) * (1 - q_prime)**i
        g_i = 1 if (2 * num - denom > eps) else (0 if (2 * num - denom < -eps) else .5)
        # if i * 2 <= n:
        #     g_i = min(g_i, .5)
        # if i * 2 >= n:
        #     g_i = max(g_i, .5)
        if denom == 0:
            continue
        g_val += bin_coeff * denom * loss(num / denom, g_i)
    return g_val

def opt_abs(n, beta, mu, p, q):
    return 0
    
def opt_sig(n, beta, mu, p, q):
    h_val = 0
    for i in range(n + 1):
        bin_coeff = comb(n, i)
        denom = mu * p**i * (1 - p)**(n - i) + (1 - mu) * q**i * (1 - q)**(n - i)
        if denom == 0:
            continue
        num = mu * p**i * (1 - p)**(n - i)
        h_val += bin_coeff * denom * max(loss(num / denom, 0), loss(num / denom, 1))
    
    return h_val

def opt_rep(n, beta, mu, p, q, p_prime=None, q_prime=None):
    if p_prime==None:
        p_prime, q_prime = calc_pq_prime(beta, mu, p, q)
    h_val = 0
    for i in range(n + 1):
        bin_coeff = comb(n, i)
        denom = mu * p_prime**i * (1 - p_prime)**(n - i) + (1 - mu) * q_prime**i * (1 - q_prime)**(n - i)
        if denom == 0:
            continue
        num = mu * p_prime**i * (1 - p_prime)**(n - i)
        h_val += bin_coeff * denom * max(loss(num / denom, 0), loss(num / denom, 1))
    
    return h_val

# Objective function to minimize
def objective(params, n, beta, g, h):
    mu, p = params
    q = 1
    return g(n, beta, mu, p, q) - h(n, beta, mu, p, q)

def plot_bounded_advantage():
    ns = [1, 2, 3]
    beta_range = np.linspace(0, 20, 1001)  # beta 的取值范围

    # 创建绘图
    fig, axes = plt.subplots(1, len(ns), figsize=(4 * len(ns), 3))

    for i, n in enumerate(ns):
        ax = axes[i]
        
        majority_curve = [majority(n, beta, 0.25, 1, 0.5) * 2 - 1 for beta in beta_range]
        minimum_curve = [minimum_s(n, beta, 0.25, 1, 0.5) * 2 - 1 for beta in beta_range]

        ax.plot(beta_range, minimum_curve, label=f'Omniscient Aggregator', alpha=0.7, linewidth=2.5)
        ax.plot(beta_range, majority_curve, label=f'Majority Vote', alpha=0.7, linewidth=2.5)
        ax.set_ylim(0.415, 0.565)
        ax.set_title(f'n={n}', fontsize=14)
        ax.set_xlabel('$\\beta$', fontsize=13)
        ax.set_ylabel('Utility', fontsize=13)
        if n == ns[0]:
            ax.legend(fontsize=13)

    # 保存图像为 PDF 文件
    plt.tight_layout()
    plt.savefig('bounded_advantage.pdf')

    # 显示图形
    plt.show()

print(majority(1, 5, 0.25, 1, 0.5) * 2 - 1)
print(majority(3, 5, 0.25, 1, 0.5) * 2 - 1)
print(minimum_s(1, 5, 0.25, 1, 0.5) * 2 - 1)
print(minimum_s(2, 5, 0.25, 1, 0.5) * 2 - 1)
print(minimum_s(3, 5, 0.25, 1, 0.5) * 2 - 1)

plot_bounded_advantage()
plot_gn_curve_final()

# Initial guess
initial_guess = [0.4, 0.8]
bounds = [(0, 1), (0, 1)]

# Range of beta values
beta_values = np.linspace(0, 10, 1001)

# ns = [3, 5, 10, 20]

funcs = {
    "Optimal Robust": minimum,
    "Majority Vote": majority,
    # "Trimmed Mean 3": lambda n, beta, mu, p, q: majority(n, beta, mu, p, q, k=3),
    # "Trimmed Mean 2": lambda n, beta, mu, p, q: majority(n, beta, mu, p, q, k=2),
    # "Trimmed Mean 1": lambda n, beta, mu, p, q: majority(n, beta, mu, p, q, k=1),
    # "Random Follow": randomfollow,
}
opts = {
    # "opt_sig": opt_sig,
    "opt_rep": opt_rep,
    # "opt_abs": opt_abs,
}
# for func in funcs:
#     for beta in [100]:
#         mu_values = np.linspace(0, 1, 300)
#         p_values = np.linspace(0, 1, 300)
#         mu_grid, p_grid = np.meshgrid(mu_values, p_values)
#         heatmap_data = np.zeros_like(mu_grid)

#         for opt in opts:
#             for i in range(len(mu_values)):
#                 for j in range(len(p_values)):
#                     heatmap_data[j, i] = -objective([mu_values[i], p_values[j]], n, beta, funcs[func], opts[opt])

#             plt.figure(figsize=(10, 8))
#             plt.imshow(heatmap_data, extent=(0, 1, 0, 1), origin='lower', aspect='auto', cmap='viridis')
#             plt.colorbar(label='Objective Value')
#             plt.xlabel('mu')
#             plt.ylabel('p')
#             plt.title(f'Heatmap of Objective Function ({opt})')
#             plt.savefig("heatmaps/" + func + "_" + opt + "_" + str(beta) + "_heatmap.png")


# Define the number of rows and columns for the subplots
ns = [1, 3, 5]
fig, axes = plt.subplots(1, len(ns), figsize=(4 * len(ns), 3))  # Adjust figsize as needed

# Loop over each n value and create the plot in the corresponding subplot
for idx, n in enumerate(ns):
    ax = axes[idx]
    gn = calc_gn(n)
    for opt in opts:

    # Prepare the results for each function
        res = []
        for beta in beta_values:
            res.append({})
            for key in funcs:
                res[-1].update({key: minimize(objective, initial_guess, args=(n, beta, funcs[key], opts[opt]), bounds=bounds, tol=eps)})

        for i in range(len(beta_values) - 1):
            for key in funcs:
                result = minimize(objective, res[i][key].x, args=(n, beta_values[i + 1], funcs[key], opts[opt]), bounds=bounds, tol=eps)
                if result.fun < res[i + 1][key].fun:
                    res[i + 1][key] = result

        for i in range(len(beta_values) - 1, 0, -1):
            for key in funcs:
                result = minimize(objective, res[i][key].x, args=(n, beta_values[i - 1], funcs[key], opts[opt]), bounds=bounds, tol=eps)
                if result.fun < res[i - 1][key].fun:
                    res[i - 1][key] = result

    # Plotting
        for key in funcs:
            ax.plot(beta_values, [-res[i][key].fun for i in range(len(res))], label=key, alpha=0.7, linewidth=2.5)
        if gn <= 10:
            ax.axvline(x=gn, color='grey', linestyle='--', linewidth=1, label="$g(n)$")  # Draw a vertical line at g(n)
        else:
            ax.plot([], [], color='grey', linestyle='--', linewidth=1, label="$g(n)$")
        # else:
        #     ax.axvline(x=gn, color='grey', linestyle='--', linewidth=1, label="$g(n)$")  # Draw a vertical line at g(n)
            # ax.text(gn + 0.2, max([-res[i][key].fun for i in range(len(res))]) * 0.9, f"$g(n)$", 
            #         ha='left', va='center', fontsize=12, color='grey', alpha=0.7)

        ax.set_title(f'n={n}', fontsize=14)
        ax.set_xlabel('$\\beta$', fontsize=13)
        ax.set_ylabel('Regret', fontsize=13)
        if n == ns[0]:
            ax.legend(fontsize=13)

# Adjust layout for better spacing
fig.tight_layout()

# Save the combined figure as a single image
plt.savefig('combined_plots.png', bbox_inches='tight')
plt.savefig('combined_plots.pdf', bbox_inches='tight')

# Show the figure (optional)
plt.show()



