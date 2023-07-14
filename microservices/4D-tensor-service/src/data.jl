# Data loading and preprocessing logic
using Flux
using CSV
using DataFrames
using Statistics
using MLJ: coerce, OneHotEncoder, Multiclass
using MLJBase: machine, fit!, transform
include("model.jl")

# Load the data and split
function load_and_split_data(filepath)
    data = CSV.read(filepath, DataFrame)
    data = preprocess_data(data)

    # Split the data into training and testing sets
    train_data, test_data = split_data(data)
    return train_data, test_data
end

# Split the data into training and testing sets
function split_data(data)
    # Assuming data is a DataFrame and the last column is the target variable
    features = data[:, 1:end-1]
    labels = data[:, end]

    # Calculate the number of training samples (e.g., 80% of the data)
    num_train = floor(Int, size(data, 1) * 0.8)

    # Split the features
    train_features = features[1:num_train, :]
    test_features = features[num_train+1:end, :]

    # Split the labels
    train_labels = labels[1:num_train]
    test_labels = labels[num_train+1:end]

    return (train_features, train_labels), (test_features, test_labels)
end

# Preprocess the data
function preprocess_data(data::DataFrame)
    # Identify categorical columns
    cat_cols = names(data)[[eltype(data[!, col]) <: AbstractString for col in names(data)]]

    # Convert categorical columns to one-hot encoding
    for col in cat_cols
        data[!, col] = coerce(data[!, col], Multiclass)
    end

    hot = machine(OneHotEncoder(), data)
    fit!(hot)
    data = transform(hot, data)

    # Normalize numerical columns
    for column in names(data)
        if eltype(data[!, column]) <: Number
            data[!, column] = (data[!, column] .- mean(data[!, column])) ./ std(data[!, column])
        end
    end    

    return data
end

# Main function to orchestrate data loading and preprocessing
function main()
    filepath = "/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Datasets-examples/churn-bigml-80.csv" 
    train_data, test_data = load_and_split_data(filepath)

    # Wrap data into DataLoader
    train_data = DataLoader(train_data, batchsize=64)
    test_data = DataLoader(test_data, batchsize=64)

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
    Flux.save("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-analyze-data/model.bson", params(model))
end
