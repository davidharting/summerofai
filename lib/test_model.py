from lib.model import LinearModel


class TestLinearModel:
    def test_linear_model_should_train(self):
        inputs = [[3, 4], [2, 0], [1, 3], [6, 8], [7, 10], [9, 9]]
        # If both inputs are odd, 1. Otherwise, 0.
        outputs = [[0], [0], [1], [0], [0], [1]]
        model = LinearModel()
        model.train(inputs, outputs)
        # We made it without throwing an error so yay

    def test_linear_model_test(self):
        train_inputs = [[4, 2], [3, 1], [5, 9], [2, 3], [6, 7]]
        train_outputs = [[6], [4], [14], [5], [13]]
        model = LinearModel()
        model.train(train_inputs, train_outputs, epochs=1001)

        predicted = model.test([[4, 9]])
        assert type(predicted) == type([])
        assert len(predicted) == 1
        assert len(predicted[0]) == 1
        assert type(predicted[0][0]) == type(13.00)

