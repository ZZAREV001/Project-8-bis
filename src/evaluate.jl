# Training logic
using Flux

# Load the model
function load_model()
    model = Flux.load("path_to_model_weights/model.bson")
    return model
end

# Load the test dataset
function load_data()
    filepath = "/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Datasets-examples/df_order_book_20200817.csv"
    test_data = load_data(filepath)  # load_data is defined in data.jl
    return test_data
end

# Preprocess the dataset
function preprocess_data(data)
    # Convert DataFrame to Matrix
    data = Matrix(data)

    # Convert Matrix to array of appropriate type for Flux
    data = Array{Float32}(data)

    return data
end

# Evaluate the model
function evaluate_model(model, data)
    predictions = model(data)
    return predictions
end

# Calculate the metrics
function calculate_metrics(predictions, ground_truth)
    mse = Flux.mse(predictions, ground_truth)
    # Add more metrics as needed
    return mse
end

# Main function to orchestrate the evaluation
function main()
    # Load the model and data
    model = load_model()
    test_data = load_data()

    # Preprocess the data
    test_data = preprocess_data(test_data)

    # Evaluate the model
    predictions = evaluate_model(model, test_data)

    # Calculate and print the metrics
    mse = calculate_metrics(predictions, test_data)
    println("Mean Squared Error: ", mse)
end

main()
