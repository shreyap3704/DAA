import matplotlib.pyplot as plt
import numpy as np
import time

# ------------------ DYNAMIC PROGRAMMING FUNCTION ------------------
def dp_knapsack(values, weights, capacity):
    n = len(values)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    start_time = time.time()

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w],
                               dp[i - 1][w - weights[i - 1]] + values[i - 1])
            else:
                dp[i][w] = dp[i - 1][w]

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Backtracking to find selected items
    w = capacity
    selected_items = [0] * n
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected_items[i - 1] = 1
            w -= weights[i - 1]

    return dp, dp[n][capacity], selected_items, elapsed_time

# ------------------ VISUALIZATION FUNCTIONS ------------------
def plot_total_value_by_capacity(dp):
    final_values = dp[-1]  # Last row = best value for each capacity
    capacities = list(range(len(final_values)))

    plt.figure(figsize=(10, 5))
    plt.plot(capacities, final_values, marker='o', color='blue')
    plt.title("Maximum Value vs Capacity")
    plt.xlabel("Knapsack Capacity")
    plt.ylabel("Maximum Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_selected_items(values, weights, selected_items):
    item_indices = [i for i, selected in enumerate(selected_items) if selected]
    selected_vals = [values[i] for i in item_indices]
    selected_wts = [weights[i] for i in item_indices]

    x = np.arange(len(item_indices))

    plt.figure(figsize=(10, 5))
    bar1 = plt.bar(x - 0.2, selected_vals, width=0.4, label='Value', color='green')
    bar2 = plt.bar(x + 0.2, selected_wts, width=0.4, label='Weight', color='orange')

    plt.xlabel("Item Index")
    plt.ylabel("Amount")
    plt.title("Values and Weights of Selected Items")
    plt.xticks(x, item_indices)
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

# ------------------ SUMMARY FUNCTION ------------------
def summarize_result(values, weights, capacity, max_value, selected_items, elapsed_time):
    print(f"\nKnapsack Capacity: {capacity}")
    print(f"Maximum Value Achieved: {max_value}")
    print(f"Selected Items (0-based index): {[i for i, bit in enumerate(selected_items) if bit == 1]}")
    print(f"Corresponding Values: {[values[i] for i in range(len(values)) if selected_items[i] == 1]}")
    print(f"Corresponding Weights: {[weights[i] for i in range(len(weights)) if selected_items[i] == 1]}")
    print(f"Total Weight Used: {sum([weights[i] for i in range(len(weights)) if selected_items[i] == 1])}")
    print(f"Execution Time: {elapsed_time:.6f} seconds")

# ------------------ MAIN EXECUTION ------------------
if __name__ == "__main__":
    # Test case (you can customize this)
    values = [60, 100, 120]
    weights = [10, 20, 30]
    capacity = 50

    dp, max_value, selected_items, elapsed_time = dp_knapsack(values, weights, capacity)

    summarize_result(values, weights, capacity, max_value, selected_items, elapsed_time)
    plot_total_value_by_capacity(dp)
    plot_selected_items(values, weights, selected_items)

