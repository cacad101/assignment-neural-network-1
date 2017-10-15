from problem_1_classifications.training.softmax_nn import SoftmaxNeuralNetwork
from problem_1_classifications.preprocess.data_preprocess import DataCollector
from problem_1_classifications.postpreprocess.data_visualization import DataVisualizationWithLabels

from problem_1_classifications.postpreprocess.data_visualization import DataVisualization

if __name__ == "__main__":

    data_visualization = DataVisualizationWithLabels()
    data_collector = DataCollector()

    train_x, train_y = data_collector.get_train_data()
    test_x, test_y = data_collector.get_test_data()

    batch_size = 32
    epochs = 1000
    decay = 1e-6

    costs = []
    predictions = []
    list_exec_times = []
    list_execution_time = []

    list_layer = [4, 3]
    list_neurons_on_each_hidden_layer = [[10, 10], [10]]

    for list_neurons in list_neurons_on_each_hidden_layer:
        softmax_nn = SoftmaxNeuralNetwork(train_x=train_x.as_matrix(), train_y=train_y,
                                          list_of_neuron_on_hidden_layer=list_neurons, verbose=False,
                                          decay=decay
                                          )
        softmax_nn.start_train(batch_size=batch_size, test_x=test_x, test_y=test_y, epochs=epochs)

        cost_train, prediction_train, exec_time, list_exec_time = softmax_nn.get_train_result()
        prediction_test = softmax_nn.get_test_result()

        costs.append(cost_train)
        predictions.append(prediction_test)

        list_exec_times.append(list_exec_time)
        list_execution_time.append(exec_time)

        print "execution time: %s \n" % exec_time

    list_x_point = []
    for cnt in range(len(list_layer)):
        list_x_point.append(range(epochs))

    data_visualization.show_plot(
        list_x_point=list_x_point, list_y_point=costs,
        x_label="epochs", y_label="cost", title="Cross Entropy Comparison layers ",
        figure_name="../image/classification/5/4_3_layer_cross_entropy.png",
        labels=list_layer
    )

    data_visualization.show_plot(
        list_x_point=list_x_point, list_y_point=predictions,
        x_label="epochs", y_label="prediction", title="Prediction Comparison layers ",
        figure_name="../image/classification/5/4_3_layer_prediction.png",
        labels=list_layer
    )

    data_visualization.show_plot(
        list_x_point=list_exec_times, list_y_point=costs,
        x_label="execution times", y_label="cost", title="Cross Entropy Comparison layers ",
        figure_name="../image/classification/5/4_3_layer_cross_entropy_exec_time.png",
        labels=list_layer
    )

    data_visualization.show_plot(
        list_x_point=list_exec_times, list_y_point=predictions,
        x_label="execution times", y_label="prediction", title="Prediction Comparison layers ",
        figure_name="../image/classification/5/4_3_layer_prediction_exec_time.png",
        labels=list_layer
    )

    data_visualization = DataVisualization()
    data_visualization.show_plot(
        list_x_point=list_layer, list_y_point=list_execution_time,
        x_label="neurons", y_label="execution_time", title="Execution time comparison for different multilayer",
        figure_name="../image/classification/3/hidden_layer_exec_time_comparison.png"
    )
