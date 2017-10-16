from problem_1_classifications.training.softmax_nn import SoftmaxNeuralNetwork
from problem_1_classifications.preprocess.data_preprocess import DataCollector
from problem_1_classifications.postpreprocess.data_visualization import DataVisualization


if __name__ == "__main__":
    data_visualization = DataVisualization()
    data_collector = DataCollector()

    train_x, train_y = data_collector.get_train_data()
    test_x, test_y = data_collector.get_test_data()

    batch_size = 32
    epochs = 1000
    decay = 1e-6
    list_neurons = [10]

    softmax_nn = SoftmaxNeuralNetwork(train_x=train_x.as_matrix(), train_y=train_y,
                                      list_of_neuron_on_hidden_layer=list_neurons, decay=decay)
    softmax_nn.start_train(batch_size=batch_size, test_x=test_x, test_y=test_y, epochs=epochs)

    cost_train, prediction_train, exec_time, list_exec_time = softmax_nn.get_train_result()
    prediction_test = softmax_nn.get_test_result()

    data_visualization.show_plot(
        list_x_point=range(epochs), list_y_point=prediction_test,
        x_label="epochs", y_label="cost",
        title="test accuracy with multi layer neurons",
        figure_name="../image/classification/1/multilayer_neurons.png"
    )
