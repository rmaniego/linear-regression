import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def main():
    # Load the data from CSV file.
    filename = "datasets/observations.csv"
    dataset = np.genfromtxt(filename, delimiter=",")

    header = dataset[0]     # Get the header from the dataset.
    dataset = dataset[1:]   # Remove the header from the dataset.
    
    # Get all rows in the specific colum.
    feature = dataset[:, 0].reshape((-1, 1)) # reshape to 2D array.
    target = dataset[:, 1]
    
    # Split dataset for training and validation (testing)
    # model_dataset = train_test_split(feature, target, test_size=0.1)
    # feature1, feature2, target1, target2 = model_dataset
    
    # Train the model
    print("\nTraining the model.")
    model = LinearRegression()
    model.fit(feature, target)
    
    # get the prediction and calculate the error
    print("\nCalculating the MSE the model.")
    predictions = model.predict(feature)
    
    # mean_square_error = np.mean((predictions - target) ** 2)
    # print(f"MSE: {mean_square_error}")
    
    slope = model.coef_[0]
    intercept = model.intercept_

    regression_model = f"{intercept:,.2f} + {slope:,.2f}x"
    print(f"Regression Model: {regression_model}")
    
    r_squared = model.score(feature, target)
    r_squared_percent = r_squared*100
    
    print(f"\nAbout {r_squared_percent:.2f}% of the data can be explained by the regression line y = {regression_model}")
    
    show_regression_line(feature, target, slope, intercept)

def show_regression_line(feature, target, slope, intercept):
    # Create scatter plot
    plt.scatter(feature, target, color="blue", label="Data Points")
    
    # Draw regression line
    plt.plot(feature, (slope*feature + intercept), color="red", label="Regression Line")

    # Set labels and title
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Simple Linear Regression")

    # Show the legend and plot
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()