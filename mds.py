import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")

plt.rcParams["figure.figsize"] = (15, 8)
plt.rcParams["font.size"] = 14
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.grid"] = True
plt.rcParams["axes.grid.axis"] = "both"
plt.rcParams["axes.grid.which"] = "both"
plt.rcParams["grid.alpha"] = 0.5
plt.rcParams["legend.fontsize"] = 14
plt.rcParams["legend.title_fontsize"] = 16
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["figure.dpi"] = 300

# Multidimensional Scaling (MDS) Analysis


# Define the function for MDS
def multidimensional_scaling_adjusted(D):
    # Calculate the matrix of squared distances
    D_squared = D**2
    # Calculate the centering matrix C
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    C = -0.5 * J @ D_squared @ J
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    # Sort the eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # Take the first two eigenvectors and eigenvalues
    E = np.diag(np.sqrt(np.abs(eigenvalues[:2])))
    P = eigenvectors[:, :2]
    # Flip the signs of the eigenvectors
    P[:, 0] = -P[:, 0]
    P[:, 1] = -P[:, 1]
    # Calculate the coordinates matrix X
    X = P @ E
    return X


D = np.array([[0, 97, 118, 60], [97, 0, 93, 84], [118, 93, 0, 167], [60, 84, 167, 0]])


X = multidimensional_scaling_adjusted(D)

print(X)

# Load the travel time data from the CSV file
file_path = "French_train_travel_time.csv"
travel_time_df = pd.read_csv(file_path)

# Convert the dataframe to a numpy array and handle missing values
travel_time_matrix = travel_time_df.drop("city", axis=1).to_numpy()
np.fill_diagonal(travel_time_matrix, 0)
nan_mask = np.isnan(travel_time_matrix)
column_means = np.nanmean(travel_time_matrix, axis=0)
travel_time_matrix[nan_mask] = np.take(column_means, np.where(nan_mask)[1])

# Apply the adjusted MDS function
coordinates_adjusted = multidimensional_scaling_adjusted(travel_time_matrix)

# Prepare the city labels for plotting
city_labels = travel_time_df["city"].values

# Plot the resulting two-dimensional points
plt.figure(figsize=(12, 8))
plt.scatter(coordinates_adjusted[:, 0], coordinates_adjusted[:, 1], marker="o")

# Annotate the points with city labels
for i, label in enumerate(city_labels):
    plt.annotate(
        label,
        (coordinates_adjusted[i, 0], coordinates_adjusted[i, 1]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
    )

# Set the title and labels
plt.title("2D Representation of French Cities Based on Train Travel Time")
plt.xlabel("Coordinate 1")
plt.ylabel("Coordinate 2")
plt.grid(True)
plt.axhline(y=0, color="k", linestyle="--")
plt.axvline(x=0, color="k", linestyle="--")
plt.tight_layout()
plt.savefig("mds_french_cities.png")
# Show the plot
plt.show()

# print the coordinates of the cities
print(coordinates_adjusted)


def calculate_strain(B, X):
    # Calculate the dot product matrix of X
    XXt = X @ X.T

    # Calculate the squared difference between B and XXt element-wise
    numerator = np.sum((B - XXt) ** 2)

    # Calculate the squared elements of B
    denominator = np.sum(B**2)

    # Compute the strain
    strain = np.sqrt(numerator / denominator)

    return strain


# Calculate the centering matrix
n = travel_time_matrix.shape[0]
J = np.eye(n) - np.ones((n, n)) / n

# Calculate matrix B for the strain calculation (Double centering)
D_squared = travel_time_matrix**2
B = -0.5 * J @ D_squared @ J

# Calculate strain using the coordinates from part (2)
strain_value = calculate_strain(B, coordinates_adjusted)

# Output the strain value
print(f"Strain Value: {strain_value}")
