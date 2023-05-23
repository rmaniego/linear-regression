import numpy as np

def main():
    # Load the data from CSV file.
    filename = "datasets/observations.csv"
    dataset = np.genfromtxt(filename, delimiter=",")
    

    header = dataset[0]     # Get the header from the dataset.
    dataset = dataset[1:]   # Remove the header from the dataset.
    
    # Get all rows in the specific colum.
    x = dataset[:, 0]
    y = dataset[:, 1]
    
    # Get the mean
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    # prepare the deviations from the mean
    deviation_from_mean_x = x - mean_x
    deviation_from_mean_y = y - mean_y

    # solve using numpy functions
    covariance_x_y = np.sum(np.dot(deviation_from_mean_x, deviation_from_mean_y))
    squared_deviation_x = np.sum(np.power(deviation_from_mean_x, 2))
    
    # slope (b₁) = Σ(x-x̄)(y-ȳ) / (x-x̄)²
    b1 = covariance_x_y / squared_deviation_x
    # intercept (b₀) = ȳ - b₁x̄
    b0 = mean_y - (b1*mean_x)

    regression_model = f"{b0:,.2f} + {b1:,.2f}x"
    print(f"Regression Model: {regression_model}")
    
    # Get each prediction for each observation.
    predictions = np.array([(b0+(b1*x1)) for x1 in x])
    
    print("\n[Measuring the Fit]")
    
    # Error Sum of Squares (SSE) = (yi-ŷ)²
    SSE = np.sum(np.power(np.subtract(y, predictions), 2))

    # Total Sum of Squares (SST) = (yi-ȳ)²
    SST = np.sum(np.power(np.subtract(y, mean_y), 2))

    # Regression Sum of Squares (SSR) = (ŷ-ȳ)²
    SSR = np.sum(np.power(np.subtract(predictions, mean_y), 2))
    
    # r-squared (r²) or goodness-of-fit measure
    # = SSR / SST or 1 - (SSE / SST)
    r_squared1 = SSR/SST
    r_squared2 = 1-(SSE/SST)
    
    print(f" * r-squared (method 1) = {r_squared1}")
    print(f" * r-squared (method 2) = {r_squared2}")
    
    r_squared_percent = r_squared1*100
    
    print(f"\nAbout {r_squared_percent:.2f}% of the data can be explained by the regression line y = {regression_model}")

if __name__ == "__main__":
    main()