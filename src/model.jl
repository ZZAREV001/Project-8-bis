# Define a hybrid Convolutional Neural Network (CNN) 
# and Long Short-Term Memory (LSTM) model with Flux.jl
using Flux

struct CNNTOLSTM
    cnn
    lstm
end

function CNNTOLSTM(input_shape::Tuple{Int,Int}, num_filters::Int, kernel_size::Tuple{Int,Int}, lstm_hidden_dim::Int, output_dim::Int)
    return CNNTOLSTM(
        Flux.Chain(
            Flux.Conv(kernel_size, 1=>num_filters, relu),
            Flux.flatten
        ),
        Flux.Chain(
            Flux.LSTM(num_filters * prod(input_shape .÷ kernel_size), lstm_hidden_dim),
            Flux.Dense(lstm_hidden_dim, output_dim)
        )
    )
end

Flux.@functor CNNTOLSTM

function (m::CNNTOLSTM)(x)
    x = m.cnn(x)
    x = reshape(x, :, size(x, 4))
    return m.lstm(x)
end

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
