import json 
import itertools
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np
import random
from matplotlib.colors import LinearSegmentedColormap
from collections import Counter

# Function to load data from a JSON file
def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)

# Function to convert the merged data into the target format
def convert_to_result_format(data, model):
    result = {}
    for problem in data:
        
        options = problem["options"]
        for temp in problem[model]:
            if temp not in result:
                result.update({temp: []})
            result[temp].append({
                "truth": 0,
                "answers": []
            })
            for i, option in enumerate(options):
                if option == problem["answers"]["text"][0]:
                    result[temp][-1]["truth"] = i
            if len(problem[model][temp]) < 20:
                print(problem, model, temp)
            for answer in problem[model][temp]:
                for i in range(len(options)):
                    if options[i] == answer["answer"]:
                        result[temp][-1]["answers"].append(i)
    return result

# Function to compute the weighted average
def average(truth, lst):
    counter = Counter(lst)
    return counter[truth] / len(lst)

# Function to compute the mode
def maj(truth, lst):
    counter = Counter(lst)
    most_common = counter.most_common()
    if counter[truth] != most_common[0][1]:
        return 0
    for i in range(1, len(most_common)):
        if most_common[i][1] != most_common[0][1]:
            return 1 / i
    return 1 / len(most_common)

# Function to calculate the weighted sum for a certain temperature
def calculate_sum(f, truth, answer, repeat=10, order=None):
    lst = [answer[i] for i in order]
    return f(truth, lst)

# Function to calculate the final results
def calculate_final_results(data, f, temperatures, repeat=10, order=None):
    results = {temp: 0 for temp in temperatures}
    for temp in temperatures:
        for entry in data[temp]:
            truth = entry["truth"]
            answers = entry["answers"]
            results[temp] += calculate_sum(f, truth, answers, repeat=repeat, order=order)
    results = {temp: results[temp] / len(data[temp]) for temp in temperatures}
    return results

# Function to plot the results by repeat
def plot_results_by_repeat(repeats, temperatures, avg_results, maj_results, path="."):
    plt.figure(figsize=(5, 3))

    # Define color map
    colors = [(68/255, 114/255, 196/255), (255/255, 192/255, 0/255)]  # Blue to Orange
    n_colors = len(temperatures)  # Number of colors matches the number of temperatures
    color_map = LinearSegmentedColormap.from_list(
        "blue_to_orange", colors, N=n_colors)
    color_cycle = color_map(np.linspace(0, 1, n_colors))

    bar_width = 0.2  # Set bar width
    bar_positions = np.arange(len(repeats))  # Positions for each repeat

    ymin = 1
    ymax = 0

    # Plot avg and maj bar charts for each temperature
    for idx, temp in enumerate(temperatures):
        color = color_cycle[idx % len(color_cycle)]  # Assign color to the current temperature
        avg_values = []
        avg_error = []
        maj_values = []
        maj_error = []

        for repeat in repeats:
            # Extract avg and maj data
            if repeat in avg_results:
                avg_values.append(np.mean(avg_results[repeat].get(temp, [0])))
                avg_error.append(np.std(avg_results[repeat].get(
                    temp, [0])) / np.sqrt(len(avg_results[repeat].get(temp, None))))
            if repeat in maj_results:
                maj_values.append(np.mean(maj_results[repeat].get(temp, [0])))
                maj_error.append(np.std(maj_results[repeat].get(temp, [0])) / np.sqrt(len(maj_results[repeat].get(temp, None))))

        # Plot avg bar chart
        if avg_values:
            plt.bar(
                bar_positions + idx * bar_width,
                avg_values,
                bar_width,
                yerr=avg_error,
                label=f'Avg t={float(temp):.1f}',
                color=color,
                alpha=0.5,
                capsize=5,
                edgecolor='black'
            )
            ymin = min(ymin, min(avg_values))
            ymax = max(ymax, max(avg_values))

        # Plot maj bar chart
        if maj_values:
            plt.bar(
                bar_positions + idx * bar_width,
                maj_values,
                bar_width,
                yerr=maj_error,
                label=f't={float(temp):.1f}',
                color=color,
                alpha=0.8,
                capsize=5,
                edgecolor='black'
            )
            ymin = min(ymin, min(maj_values))
            ymax = max(ymax, max(maj_values))

    # Chart settings
    plt.xlabel('Number of Experts $n$')
    plt.ylabel('Expected Accuracy')
    plt.xticks(bar_positions + bar_width *
               (len(temperatures) - 1) / 2, repeats)

    # Automatically adjust y-axis to include group range
    plt.ylim(1.05 * ymin - 0.05 * ymax, 1.05 * ymax - 0.05 * ymin)
    plt.legend(loc='upper left')
    plt.tight_layout()

    # Save the plot as PDF and PNG
    plt.savefig(f"{path}/results_by_repeat_plot_qa.pdf")
    plt.savefig(f"{path}/results_by_repeat_plot_qa.png")
    plt.show()

# Function to estimate beta for each temperature
def estimate_beta(n, temperatures, data):
    def log_likelihood(beta, p):
        """
        Calculate the log-likelihood function for a given beta value
        :param beta: The parameter beta
        :param p: Input data (list or np.array)
        :return: The value of the log-likelihood function
        """
        p = np.array(p)
        term = (1 - 2 * p) * beta
        log_likelihood_value = np.sum(np.log(1 + np.exp(term)))
        return log_likelihood_value

    res = {}
    for t in temperatures:
        P = []
        for entry in data[t]:
            for answer in entry["answers"]:
                P.append(1 if answer == entry["answers"][0] else 0)
        initial_guess = 1
        result = minimize(log_likelihood, initial_guess, args=(P))
        res.update({
            t: result.x[0]
        })
    return res

# Main program
def main():
    n = 20
    random.seed(42)
    path = "math_qa_500"
    data = load_data(f'{path}/data.json')
    result = convert_to_result_format(data, model="gpt-4o-mini")
    temperatures = ['0', '0.5', '1']
    repeats = [1, 3, 5]

    avg_results = {}
    maj_results = {}

    # Calculate and store results for different repeats
    for repeat in repeats:
        print(f"n={repeat}")
        for _ in range(1000):
            order = random.sample(range(n), repeat)
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

    # Plot the results and save as PDF and PNG
    plot_results_by_repeat(repeats, temperatures, {}, maj_results, path=path)

if __name__ == '__main__':
    main()
