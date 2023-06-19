using Test

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

# Define a test function
function test_main()
    # Call the main function
    main()

    # Check that the model file was created
    @test isfile("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-analyze-fin-data/model.bson")

    # Check that the model file was created
    model_path = "/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-analyze-fin-data/model.bson"
    @test isfile(model_path)

    # Load the model
    model = BSON.load(model_path)

    # Generate some dummy input data
    # The shape and type of this data will depend on your specific model
    input_data = rand(128, 128, 1, 64)

    # Make a prediction
    prediction = model(input_data)

    # Check that the prediction has the expected shape
    # Again, the expected shape will depend on your specific model
    @test size(prediction) == (1, 64)
end    

# Call the test function
test_split_data()
test_load_and_split_data()
test_main()
