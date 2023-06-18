
function test_load_and_split_data()
    # Define the file path
    filepath = "/path/to/your/data.csv"

    # Call the function
    train_data, test_data = load_and_split_data(filepath)

    # Load the full data
    full_data = CSV.read(filepath, DataFrame)

    # Check the number of samples
    @assert size(train_data, 1) == floor(Int, size(full_data, 1) * 0.8)
    @assert size(test_data, 1) == size(full_data, 1) - size(train_data, 1)

    println("Test passed.")
end

# Call the test function
test_load_and_split_data()

using DataFrames

function test_split_data()
    # Create a DataFrame with known values
    data = DataFrame(A = 1:100, B = 101:200, C = 201:300)

    # Call the split_data function
    (train_features, train_labels), (test_features, test_labels) = split_data(data)

    # Check the number of samples in the training and testing sets
    @assert size(train_features, 1) == 80
    @assert size(test_features, 1) == 20

    # Check the number of features and labels
    @assert size(train_features, 2) == 2
    @assert size(train_labels, 1) == 80
    @assert size(test_features, 2) == 2
    @assert size(test_labels, 1) == 20

    # Check the values in the training and testing sets
    @assert train_features[1, "A"] == 1
    @assert train_labels[1] == 201
    @assert test_features[1, "A"] == 81
    @assert test_labels[1] == 281

    println("Test passed.")
end

# Call the test function
test_split_data()
