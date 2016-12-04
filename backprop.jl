function sigmoid(X, derivative)
  if derivative == 0
    return map(x->1 / (1 + exp(-x)), X)
  end
  if derivative == 1
    return map(x->(x*(1-x)), X)
  end
end

# % Train XOR
# % Input and bias
X = [0 0 1;0 1 1;1 0 1;1 1 1];
# % Output
y = [0;1;1;0];

# % Synapses weight
layer1 = rand(3,4);
layer2 = rand(4,1);

iterations = 10000000;
tic();
for i in collect(1:iterations)
    # % Forward propagation.
    l0 = X
    l1 = sigmoid(l0*layer1, 0);
    # % l2 is the output layer
    l2 = sigmoid(l1*layer2, 0);

    # calculate how far we're off
    output_error = y - l2;

    if i % 10000 == 0
        println("Error: ", mean(abs(output_error)));
    end
    # calculate deltas
    l2_delta = output_error .* sigmoid(l2, 1);
    l1_error = l2_delta*layer2';
    l1_delta = l1_error .* sigmoid(l1, 1);

    # % Update synapses weight
    layer2 = layer2 + l1'*l2_delta;
    layer1 = layer1 + l0'*l1_delta;

    if i == iterations
      println(l2);
      println("Number of iterations: ", iterations);
      toc();
    end
end
