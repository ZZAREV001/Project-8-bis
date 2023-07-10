# Data loading and preprocessing logic 
using Flux
using Flux: @epochs
using Flux.Data: DataLoader
using JSON
using BSON
include("data.jl")
include("model.jl")

# Load the dataset
function load_data(filepath)
    train_data, test_data = load_and_split_data(filepath)  # Returns a tuple of a DataFrame and a Vector

    # Preprocess the training and testing data separately
    train_features, train_labels = train_data
    test_features, test_labels = test_data

    train_features = preprocess_data(train_features)  # Expects a DataFrame
    test_features = preprocess_data(test_features)  # Expects a DataFrame

    return (train_features, train_labels), (test_features, test_labels)
end

# Preprocess the dataset
function preprocess_data(data)
    # Convert DataFrame to Matrix
    data = Matrix(data)

    # Convert Matrix to array of appropriate type for Flux
    data = Array{Float32}(data)

    return data
end

# Define the loss function
loss(model, x, y) = Flux.mse(model(x), y)

# Define the callback for evaluating the model
evalcb(loss) = @show(loss)

# Define the training process
function train_model(model, train_data, test_data, epochs, opt)
    for epoch in 1:epochs
        for (x, y) in train_data
            gs = Flux.gradient(params(model)) do
                l = loss(model, x, y)
                evalcb(l)
                return l
            end
            Flux.Optimise.update!(opt, params(model), gs)
        end
    end
end

# Main function to orchestrate the training
function main()
    filepath = "/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Datasets-examples/churn-bigml-80.csv" 
    (train_features, train_labels), (test_features, test_labels) = load_data(filepath)

    # Preprocess the data
    train_features = preprocess_data(train_features)
    test_features = preprocess_data(test_features)

    # Wrap data into DataLoader
    train_data = DataLoader((train_features, train_labels), batchsize=64)
    test_data = DataLoader((test_features, test_labels), batchsize=64)

    # Load the hyperparameters
    hyperparameters = JSON.parsefile("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-analyze-data/config/hyperparameters.json")

    # Initialize the variables
    input_dim = Tuple(hyperparameters["input_shape"])
    cnn_output_dim = hyperparameters["num_filters"]
    lstm_hidden_dim = hyperparameters["lstm_hidden"]
    kernel_size = Tuple(hyperparameters["kernel_size"])
    output_dim = hyperparameters["output_dim"]

    # Define the model
    model = CNNTOLSTM(input_dim, cnn_output_dim, kernel_size, lstm_hidden_dim, output_dim)

    # Define the optimizer
    opt = ADAM(0.01)

    # Train the model
    train_model(model, train_data, test_data, 100, opt)

    # Save the trained model
    model_path = "/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-analyze-data/model.bson"
    try
        BSON.@save model_path model
        println("Model saved successfully.")
    catch e
        println("Failed to save the model.")
        println("Error: ", e)
    end
end

main()
