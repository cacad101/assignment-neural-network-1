from problem_1_classifications.training.softmax_nn import SoftmaxNeuralNetwork
from problem_1_classifications.preprocess.data_preprocess import DataCollector
from problem_1_classifications.postpreprocess.data_visualization import DataVisualizationWithLabels


if __name__ == "__main__":

    data_visualization = DataVisualizationWithLabels()
    data_collector = DataCollector()

    train_x, train_y = data_collector.get_train_data()
    test_x, test_y = data_collector.get_test_data()

    learning_decays = [0, 1e-3, 1e-6, 1e-9, 1e-12]
    batch_size = 16
    epochs = 1000
    list_neurons_hidden_layer = [15]

    costs = []
    predictions = []
    list_exec_times = []

    list_execution_time = []

    for decay in learning_decays:
        print "decay: %s \n" % decay
        softmax_nn = SoftmaxNeuralNetwork(train_x=train_x.as_matrix(), train_y=train_y,
                                          list_of_neuron_on_hidden_layer=list_neurons_hidden_layer,
                                          verbose=False, decay=decay)
        softmax_nn.start_train(batch_size=batch_size, test_x=test_x,
                               test_y=test_y, epochs=epochs)

        cost_train, prediction_train, exec_time, list_exec_time = softmax_nn.get_train_result()
        prediction_test = softmax_nn.get_test_result()

        costs.append(cost_train)
        predictions.append(prediction_test)
        list_exec_times.append(list_exec_time)

        list_execution_time.append(exec_time)

    list_x_point = []
    for cnt in range(len(learning_decays)):
        list_x_point.append(range(epochs))

    data_visualization.show_plot(
        list_x_point=list_x_point, list_y_point=costs,
        x_label="epochs", y_label="cost",
        title="Cross Entropy with learning decay comparison",
        figure_name="../image/classification/4/learning_decay_comparison_cross_entropy.png",
        labels=learning_decays
    )

    data_visualization.show_plot(
        list_x_point=list_x_point, list_y_point=predictions,
        x_label="epochs", y_label="prediction",
        title="Prediction with learning decay comparison",
        figure_name="../image/classification/4/learning_decay_comparison_prediction.png",
        labels=learning_decays
    )

    data_visualization.show_plot(
        list_x_point=list_exec_times, list_y_point=costs,
        x_label="execution times", y_label="cost",
        title="Cross Entropy with learning decay comparison",
        figure_name="../image/classification/4/learning_decay_comparison_cross_entropy_exec_times.png",
        labels=learning_decays
    )

    data_visualization.show_plot(
        list_x_point=list_exec_times, list_y_point=predictions,
        x_label="execution times", y_label="prediction",
        title="Prediction with learning decay comparison",
        figure_name="../image/classification/4/learning_decay_comparison_prediction_exec_times.png",
        labels=learning_decays
    )
