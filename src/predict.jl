# Prediction logic
using Flux
include("model.jl")  

# Define the model structure
input_shape = (128, 128)  
num_filters = 64  
kernel_size = (3, 3)  
lstm_hidden_dim = 128  
output_dim = 1  

model = CNNTOLSTM(input_shape, num_filters, kernel_size, lstm_hidden_dim, output_dim)

# Load the trained model weights into the model structure
println(isfile("/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Datasets-examples/model.bson"))
Flux.loadparams!(model, "/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Datasets-examples/model.bson")

# Define the prediction function
function predict(model, x)
    # Preprocess the input if necessary
    x = preprocess(x)  # Assuming preprocess is defined in train.jl or data.jl
    
    # Pass the input through the model
    y_pred = model(x)
    
    # Postprocess the prediction if necessary
    y_pred = postprocess(y_pred)  # Assuming postprocess is defined in train.jl or data.jl
    
    return y_pred
end



# Use the prediction function (optional: only if we load new data not contained in the initial csv file)
# x_new = ...  # Load or define the new input data here
# y_pred = predict(model, x_new)

# Display or save the prediction
# ("The prediction is ", y_pred)
