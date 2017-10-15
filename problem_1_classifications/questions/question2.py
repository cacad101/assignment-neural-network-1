from problem_1_classifications.training.softmax_nn import SoftmaxNeuralNetwork
from problem_1_classifications.preprocess.data_preprocess import DataCollector
from problem_1_classifications.postpreprocess.data_visualization import DataVisualizationWithLabels, DataVisualization


if __name__ == "__main__":

    # Question A
    print ()

    data_visualization = DataVisualizationWithLabels()
    data_collector = DataCollector()
    train_x, train_y = data_collector.get_train_data()
    test_x, test_y = data_collector.get_test_data()

    # number_data = train_x.shape[0]
    list_batch = [4, 8, 16, 32, 64]
    list_execution_time = []

    epochs = 10

    costs = []
    predictions = []
    list_exec_times = []

    list_execution_time = []

    for batch in list_batch:
        softmax_nn = SoftmaxNeuralNetwork(train_x=train_x.as_matrix(), train_y=train_y,
                                          list_of_neuron_on_hidden_layer=[10], verbose=False)
        softmax_nn.start_train(batch_size=batch, test_x=test_x, test_y=test_y, epochs=epochs)

        cost_train, prediction_train, exec_time, list_exec_time = softmax_nn.get_train_result()
        prediction_test = softmax_nn.get_test_result()

        costs.append(cost_train)
        predictions.append(prediction_test)
        list_exec_times.append(list_exec_time)

        list_execution_time.append(exec_time)
        print "execution_time: %s " % exec_time

    list_x_point = []
    for cnt in range(len(list_batch)):
        list_x_point.append(range(epochs))

    data_visualization.show_plot(
        list_x_point=list_x_point, list_y_point=costs,
        x_label="epochs", y_label="cost",
        title="Cross Entropy with batch comparison",
        figure_name="../image/classification/2/batch_comparison_cross_entropy.png",
        labels=list_batch
    )

    data_visualization.show_plot(
        list_x_point=list_x_point, list_y_point=predictions,
        x_label="epochs",
        y_label="prediction",
        title="Prediction with batch comparison",
        figure_name="../image/classification/2/batch_comparison_prediction.png",
        labels=list_batch
    )

    data_visualization.show_plot(
        list_x_point=list_exec_times, list_y_point=costs,
        x_label="execution times", y_label="cost",
        title="Cross Entropy with batch comparison",
        figure_name="../image/classification/2/batch_comparison_cross_entropy_exec_time.png",
        labels=list_batch
    )

    data_visualization.show_plot(
        list_x_point=list_exec_times, list_y_point=predictions,
        x_label="execution times",
        y_label="prediction",
        title="Prediction with batch comparison",
        figure_name="../image/classification/2/batch_comparison_prediction_exec_time.png",
        labels=list_batch
    )

    # Question B
    data_visualization = DataVisualization()

    data_visualization.show_plot(
        list_x_point=list_batch, list_y_point=list_execution_time,
        x_label="batch_size", y_label="execution_time", title="Execution Time Comparison",
        figure_name="./image/classification/2/execution_time_comparison.png",
    )