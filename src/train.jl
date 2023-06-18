# Data loading and preprocessing logic 
using Flux
using Flux: @epochs
using Flux.Data: DataLoader
using JSON
include("data.jl")
include("model.jl")

# Load the dataset
function load_data(filepath)
    data = load_and_split_data(filepath)  # Attention: risk of infinite loop here
    data = preprocess_data(data)

    # Split the data into training and testing sets
    train_data, test_data = split_data(data)
    return train_data, test_data
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
    filepath = "/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Datasets-examples/df_order_book_20200817.csv" 
    train_data, test_data = load_data(filepath)

    # Load and preprocess the data
    train_data, test_data = load_data()
    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)

    # Wrap data into DataLoader
    train_data = DataLoader(train_data, batchsize=64)
    test_data = DataLoader(test_data, batchsize=64)

    # Load the hyperparameters
    hyperparameters = JSON.parsefile("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-analyze-fin-data/config/hyperparameters.json")

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
    Flux.save("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Datasets-examples/model.bson", params(model))
end

main()

