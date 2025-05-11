import json 
import itertools
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import random
import statsmodels.api as sm
import math

# Load JSON data from the file
def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)

# Clean and merge data
def merge_data(data, model='gpt-4o-mini-2024-07-18'):
    merged_data = {}
    for item in data:
        key = (
            item['left_red'], item['left_blue'], item['right_red'], item['right_blue'],
            item['left_probability'], item['right_probability']
        )
        color = item['color']

        cleaned_data = {
            k: [{'answer': entry['answer']} for entry in v]
            for k, v in item[model].items()
        }

        if key not in merged_data:
            merged_data[key] = {}

        merged_data[key][color] = cleaned_data

    for key, value in merged_data.items():
        if 'red' not in value:
            value['red'] = {k: [{'answer': 'R'} for _ in v] for k, v in value['blue'].items()}
        elif 'blue' not in value:
            value['blue'] = {k: [{'answer': 'R'} for _ in v] for k, v in value['red'].items()}

    return merged_data

# Convert merged data into the target format
def convert_to_result_format(merged_data):
    result = []
    for key, value in merged_data.items():
        result.append({
            'left_red': key[0],
            'left_blue': key[1],
            'right_red': key[2],
            'right_blue': key[3],
            'left_probability': key[4],
            'right_probability': key[5],
            'color': value
        })
    return result

# Calculate weighted average
def average(lst):
    return sum(lst) / len(lst)

# Calculate majority vote
def maj(lst):
    if sum(lst) * 2 > len(lst):
        return 1.0
    elif sum(lst) * 2 < len(lst):
        return 0.0
    return 0.5

# Calculate weighted sum at a given temperature
def calculate_sum(f, T, red, blue, temperatures, repeat=10, order=None):
    results = {}

    for temp in temperatures:
        total_sum = 0
        P = [1.0]
        for i in range(repeat):
            new_P = [0 for j in range(i + 2)]
            for j in range(i + 1):
                new_P[j + int(T[temp][0][order[0][i]])] += P[j] * blue / 100.0
                new_P[j + int(T[temp][1][order[1][i]])] += P[j] * red / 100.0
            P = new_P
        for i in range(repeat + 1):
            total_sum += P[i] * (1 - f([1 for j in range(i)] + [0 for j in range(repeat - i)]))

        results[temp] = total_sum

    return results

# Calculate final results based on the given function and temperatures
def calculate_final_results(data, f, temperatures, repeat=10, order=None):
    total_results = {temp: 0 for temp in temperatures}
    total_entries = len(data)
    results = {temp: [] for temp in total_results}
    for entry in data:
        left_red = entry['left_red']
        left_blue = entry['left_blue']
        right_red = entry['right_red']
        right_blue = entry['right_blue']
        p = entry['left_probability'] / 100.0
        opt = 0
        for i in range(repeat + 1):
            opt += math.comb(repeat, i) * max(((left_red / 100) ** i) * ((left_blue / 100) ** (repeat - i)) * p,
                                              ((right_red / 100) ** i) * ((right_blue / 100) ** (repeat - i)) * (1 - p))
        T = entry['T']

        results0 = calculate_sum(f, T, left_red, left_blue, temperatures, repeat, order)
        results1 = calculate_sum(f, T, right_red, right_blue, temperatures, repeat, order)
        for temp in total_results:
            results[temp].append(((p * results0[temp]) + (1.0 - p) * (1 - results1[temp])))
            total_results[temp] += ((p * results0[temp]) + (1.0 - p) * (1 - results1[temp]))

    return {temp: np.mean(result) for temp, result in results.items()}

# Plot results by repeat values
def plot_results_by_repeat(repeats, temperatures, avg_results, maj_results, betas):
    plt.figure(figsize=(5, 3))

    # Define color mapping
    colors = [(68/255, 114/255, 196/255), (255/255, 192/255, 0/255)]  # Blue to Orange
    n_colors = len(temperatures)  # Number of colors matches the number of temperatures
    color_map = LinearSegmentedColormap.from_list("blue_to_orange", colors, N=n_colors)
    color_cycle = color_map(np.linspace(0, 1, n_colors))

    bar_width = 0.2  # Set the bar width
    bar_positions = np.arange(len(repeats))  # Positions for each repeat

    ymin = 1
    ymax = 0

    # Plot avg and maj bar charts for each temperature
    for idx, temp in enumerate(temperatures):
        color = color_cycle[idx % len(color_cycle)]  # Assign color to current temperature
        avg_values = []
        avg_error = []
        maj_values = []
        maj_error = []

        for repeat in repeats:
            if repeat in avg_results:
                avg_values.append(np.mean(avg_results[repeat].get(temp, [0])))
                avg_error.append(np.std(avg_results[repeat].get(temp, [0])) / np.sqrt(len(avg_results[repeat].get(temp, None))))
            if repeat in maj_results:
                maj_values.append(np.mean(maj_results[repeat].get(temp, [0])))
                maj_error.append(np.std(maj_results[repeat].get(temp, [0])) / np.sqrt(len(maj_results[repeat].get(temp, None))))

        # Plot avg bars
        beta = f'$\\beta={betas[temp]:.2f}$'
        if betas[temp] > 1e8:
            beta = '$\\beta\\rightarrow\infty$'
        if avg_values:
            plt.bar(
                bar_positions + idx * bar_width,
                avg_values,
                bar_width,
                yerr=avg_error,
                label=f'Avg t={float(temp):.1f} {beta}',
                color=color,
                alpha=0.5,
                capsize=5,
                edgecolor='black'
            )
            ymin = min(ymin, min(avg_values))
            ymax = max(ymax, max(avg_values))

        # Plot maj bars
        if maj_values:
            plt.bar(
                bar_positions + idx * bar_width,
                maj_values,
                bar_width,
                yerr=maj_error,
                label=f't={float(temp):.1f}, {beta}',
                color=color,
                alpha=0.8,
                capsize=5,
                edgecolor='black'
            )
            ymin = min(ymin, min(maj_values))
            ymax = max(ymax, max(maj_values))

    # Set up the plot
    plt.xlabel('Number of Experts $n$')
    plt.ylabel('Expected Utility')
    plt.xticks(bar_positions + bar_width * (len(temperatures) - 1) / 2, repeats)

    plt.ylim(1.05 * ymin - 0.05 * ymax, 1.05 * ymax - 0.05 * ymin)
    plt.legend(loc='upper left')
    plt.tight_layout()

    # Save the plot
    plt.savefig("results_by_repeat_plot.pdf")  # Save as PDF
    plt.savefig("results_by_repeat_plot.png")  # Save as PNG
    plt.show()

# Estimate beta values
def estimate_beta(n, temperatures, data):
    res = {}
    num_temps = len(temperatures)
    plt.figure(figsize=(4 * num_temps, 3))  # Set up the figure size

    sorted_temperatures = sorted(temperatures)
    colors = [(68/255, 114/255, 196/255), (255/255, 192/255, 0/255)]  # Blue to Orange
    n_colors = len(temperatures)  # Number of colors matches the number of temperatures
    color_map = LinearSegmentedColormap.from_list("blue_to_orange", colors, N=n_colors)
    color_cycle = color_map(np.linspace(0, 1, n_colors))

    for i, t in enumerate(sorted_temperatures):
        X = []
        P = []
        for entry in data:
            left_red = entry['left_red'] / 100.0
            right_red = entry['right_red'] / 100.0
            p = entry['left_probability'] / 100.0
            for color in entry["color"]:
                if color == "red":
                    if (left_red * p + right_red * (1 - p)) == 0:
                        continue
                    poster = left_red * p / \
                        (left_red * p + right_red * (1 - p))
                else:
                    if ((1 - left_red) * p + (1 - right_red) * (1 - p)) == 0:
                        continue
                    poster = (1 - left_red) * p / ((1 - left_red)
                                                   * p + (1 - right_red) * (1 - p))

                X.append(sum([1 if entry["color"][color][t][i]
                         ["answer"] == "L" else 0 for i in range(n)]) / n)
                P.append(poster)
                X.append(sum([0 if entry["color"][color][t][i]
                         ["answer"] == "L" else 1 for i in range(n)]) / n)
                P.append(1 - poster)

        X = np.array(X)
        P = np.array(P).reshape(-1, 1)  # Ensure P is a column vector

        X_with_intercept = sm.add_constant(P)

        logit_model = sm.Logit(X, X_with_intercept)
        result = logit_model.fit()

        coef = result.params
        std_err = result.bse  # Standard errors

        z_values = coef / std_err
        p_values = result.pvalues

        # Print regression results for each temperature
        print(f"Results for temperature {t}:")
        print("Coefficients (betas):", coef)
        print("Standard errors:", std_err)
        print("z-values:", z_values)
        print("p-values:", p_values)

        plt.subplot(1, num_temps, i + 1)
        plt.scatter(P, X, alpha=0.5, s=40, color='gray',
                    edgecolors='none')  # Scatter points
        p_values_curve = np.linspace(0, 1, 1000).reshape(-1, 1)
        X_values = result.predict(sm.add_constant(p_values_curve))
        if result.mle_retvals['converged']:
            beta = f'$\\beta={-coef[0]:.2f}$'
            res[t] = -coef[0]
        else:
            beta = '$\\beta\\rightarrow\infty$'
            res[t] = 1e9
        plt.plot(p_values_curve, X_values, color=color_cycle[i % len(color_cycle)], linewidth=2.5,
                 label=f"Fitted $\psi$, {beta}")

        plt.xlabel("Posterior Probability", fontsize=12)
        plt.ylabel("Decision Proportion", fontsize=12)
        plt.title(f"t={float(t):.1f}", fontsize=14)
        plt.legend(loc="upper left", fontsize=10)
        plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("estimate_beta.png", dpi=300)
    plt.savefig("estimate_beta.pdf", dpi=300)
    plt.close()

    return res

# Main function
def main():
    n = 20
    random.seed(42)
    data = load_data('results_5/data.json')
    merged_data = merge_data(data, model="openai/gpt-4o-mini-2024-07-18")

    result = convert_to_result_format(merged_data)
    temperatures = ['0', '0.5', '1']
    repeats = [1, 3, 5]
    betas = estimate_beta(n, temperatures, result)

    for item in result:
        T = {}
        for intensity in temperatures:
            T[intensity] = [[], []]
            for color in ['blue', 'red']:
                for answer in item['color'][color][intensity]:
                    T[intensity][0 if color == 'blue' else 1].append(0.0 if answer['answer'] == 'L' else 1.0)
        del item['color']
        item['T'] = T

    avg_results = {}
    maj_results = {}
    for repeat in repeats:
        print(f"n={repeat}")
        for _ in range(1000):
            order = [random.sample(range(n), repeat), random.sample(range(n), repeat)]
            avg_result = calculate_final_results(result, average, temperatures=temperatures, repeat=repeat, order=order)
            maj_result = calculate_final_results(result, maj, temperatures=temperatures, repeat=repeat, order=order)
            if repeat not in avg_results:
                avg_results[repeat] = {}
                maj_results[repeat] = {}
                for temp in avg_result:
                    avg_results[repeat][temp] = [avg_result[temp]]
                for temp in avg_result:
                    maj_results[repeat][temp] = [maj_result[temp]]
            else:
                for temp in avg_result:
                    avg_results[repeat][temp].append(avg_result[temp])
                for temp in avg_result:
                    maj_results[repeat][temp].append(maj_result[temp])

    plot_results_by_repeat(repeats, temperatures, {}, maj_results, betas)

if __name__ == '__main__':
    main()
