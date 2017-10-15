from training.softmax_nn import SoftmaxNeuralNetwork
from preprocess.data_preprocess import DataCollector
from postpreprocess.data_visualization import DataVisualizationWithLabels

from postpreprocess.data_visualization import DataVisualization

if __name__ == "__main__":
    # visualize
    data_visualization = DataVisualizationWithLabels()
    data_collector = DataCollector()

    train_x, train_y = data_collector.get_train_data()
    test_x, test_y = data_collector.get_test_data()

    list_neurons = [5, 10, 15, 20, 25]
    batch_size = 16
    epochs = 100

    costs = []
    predictions = []
    list_exec_times = []

    list_execution_time = []

    for neurons in list_neurons:
        softmax_nn = SoftmaxNeuralNetwork(train_x=train_x.as_matrix(), train_y=train_y,
                                          list_of_neuron_on_hidden_layer=[neurons], verbose=False)
        softmax_nn.start_train(batch_size=batch_size, test_x=test_x, test_y=test_y, epochs=epochs)

        cost_train, prediction_train, exec_time, list_exec_time = softmax_nn.get_train_result()
        prediction_test = softmax_nn.get_test_result()

        costs.append(cost_train)
        predictions.append(prediction_test)
        list_exec_times.append(list_exec_time)

        list_execution_time.append(exec_time)

        print "execution_time: %s " % exec_time

    list_x_point = []
    for cnt in range(len(list_neurons)):
        list_x_point.append(range(epochs))

    data_visualization.show_plot(
        list_x_point=list_x_point, list_y_point=costs,
        x_label="epochs", y_label="cost",
        title="Cross Entropy with neuron comparison",
        figure_name="../image/classification/23neuron_comparison_cross_entropy.png",
        labels=list_neurons
    )

    data_visualization.show_plot(
        list_x_point=list_x_point, list_y_point=predictions,
        x_label="epochs", y_label="prediction",
        title="Prediction with neuron comparison",
        figure_name="../image/classification/3/neuron_comparison_prediction.png",
        labels=list_neurons
    )

    data_visualization.show_plot(
        list_x_point=list_exec_times, list_y_point=costs,
        x_label="execution times", y_label="cost",
        title="Cross Entropy with neuron comparison",
        figure_name="../image/classification/3/neuron_comparison_cross_entropy_exec_times.png",
        labels=list_neurons
    )

    data_visualization.show_plot(
        list_x_point=list_exec_times, list_y_point=predictions,
        x_label="execution times", y_label="prediction",
        title="Prediction with neuron comparison",
        figure_name="../image/classification/3/neuron_comparison_prediction_exec_times.png",
        labels=list_neurons
    )

    data_visualization = DataVisualization()
    data_visualization.show_plot(
        list_x_point=list_neurons, list_y_point=list_execution_time,
        x_label="neurons", y_label="execution_time",
        title="Execution time comparison for different hidden layer network",
        figure_name="../image/classification/3/hidden_layer_exec_time_comparison.png"
    )
