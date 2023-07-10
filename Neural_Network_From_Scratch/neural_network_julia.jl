using Random
using CSV

data = CSV.read("urpathto/train.csv")

data = Matrix(data)

# Shuffle data
Random.shuffle!(data)

# Separate development and training data
data_dev = data[:, 1:1000]'
Y_dev = data_dev[:, 1]
X_dev = data_dev[:, 2:end] / 255.0

data_train = data[:, 1001:end]'
Y_train = data_train[:, 1]
X_train = data_train[:, 2:end] / 255.0

function init_params()
    W1 = randn(10, 784) * sqrt(1/784)
    b1 = randn(10, 1) * sqrt(1/10)
    W2 = randn(10, 10) * sqrt(1/20)
    b2 = randn(10, 1) * sqrt(1/784)
    return W1, b1, W2, b2
end

function ReLU(Z)
    return max.(Z, 0)
end

function softmax(Z)
    A = exp.(Z) ./ sum(exp.(Z))
    return A
end

function forward_prop(W1, b1, W2, b2, X)
    Z1 = W1 * X + b1
    A1 = ReLU(Z1)
    Z2 = W2 * A1 + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2
end

function one_hot(Y)
    one_hot_Y = zeros(Y.size, maximum(Y) + 1)
    one_hot_Y[CartesianIndex.(1:Y.size, Y)] = 1
    one_hot_Y = transpose(one_hot_Y)
    return one_hot_Y
end

function deriv_ReLU(Z)
    return Z .> 0
end

function backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / size(Y, 1) * dZ2 * transpose(A1)
    db2 = 1 / size(Y, 1) * sum(dZ2, dims=2)
    dZ1 = transpose(W2) * dZ2 .* deriv_ReLU(Z1)
    dW1 = 1 / size(Y, 1) * dZ1 * transpose(X)
    db1 = 1 / size(Y, 1) * sum(dZ1, dims=2)
    return dW1, db1, dW2, db2
end

function update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2
end

function get_predictions(A2)
    return argmax(A2, dims=1)
end

function get_accuracy(predictions, Y)
    return sum(predictions .== Y) / size(Y, 1)
end

function gradient_descent(X, Y, alpha, iterations)
    W1, b1, W2, b2 = init_params()
    for i in 1:iterations
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0
            println("Iteration: ", i)
            println("Accuracy: ", get_accuracy(get_predictions(A2), Y))
        end
    end
    return W1, b1, W2, b2
end

Random.seed!(123)  # Set random seed for reproducibility

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.1, 1000)
