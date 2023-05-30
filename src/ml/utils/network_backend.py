from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Dense,
    Dropout,
    LeakyReLU,
    Subtract,
)


class BackendNetwork:
    def __init__(self, nwk_type="shallow"):
        self.nwk_type = nwk_type

    def residual_block(self, nwk_input, neurons=50):
        hidden_1 = Dense(neurons, activation="relu")(nwk_input)
        hidden_2 = Dense(neurons, activation="relu")(hidden_1)
        hidden_3 = Dense(neurons, activation="relu")(hidden_2)
        sub = Subtract()([hidden_3, hidden_1])
        relu = Activation("relu")(sub)
        return relu

    def __call__(self, nwk_input):
        if self.nwk_type == "shallow":
            hidden_1 = Dense(1500, activation="relu")(nwk_input)
            hidden_1 = Dropout(0.5)(hidden_1)
            hidden_1 = Dense(200, activation="relu")(hidden_1)

        elif self.nwk_type == "residual":
            hidden_0 = Dense(300, activation="relu")(nwk_input)
            hidden_1 = Dense(150, activation="relu")(hidden_0)
            for i in range(3):
                hidden_1 = self.residual_block(hidden_1, neurons=100)

        elif self.nwk_type == "DNN":
            feature_1 = Dense(600, activation="relu")(nwk_input)
            hidden_1 = Dropout(0.5)(feature_1)
            hidden_1 = Dense(400, activation="relu")(feature_1)
            hidden_1 = Dense(200, activation="relu")(hidden_1)
            hidden_1 = Dense(200, activation="relu")(hidden_1)
            hidden_1 = Dense(50, activation="relu")(hidden_1)

        elif self.nwk_type == "linear":
            hidden_1 = Dense(400)(nwk_input)

        elif self.nwk_type == "DNN_2":
            feature_1 = Dense(1024, activation="relu")(nwk_input)
            hidden_1 = Dense(1024, activation="relu")(feature_1)
            hidden_1 = Dropout(0.5)(hidden_1)
            hidden_1 = Dense(512, activation="relu")(hidden_1)
            hidden_1 = Dense(100, activation="relu")(hidden_1)
            hidden_1 = Dropout(0.1)(hidden_1)

        elif self.nwk_type == "v3":
            hidden_1 = Dense(256, activation="relu")(nwk_input)
            hidden_1 = Dropout(rate=0.2)(hidden_1)
            hidden_1 = Dense(256, activation="relu")(hidden_1)
            hidden_1 = Dropout(rate=0.2)(hidden_1)
            hidden_1 = Dense(100, activation="relu")(hidden_1)

        elif self.nwk_type == "v3_batch":
            hidden_1 = BatchNormalization()(nwk_input)
            hidden_1 = Dense(256, kernel_initializer="he_normal")(hidden_1)
            hidden_1 = BatchNormalization()(hidden_1)
            hidden_1 = Activation("elu")(hidden_1)
            hidden_1 = Dropout(rate=0.3)(hidden_1)
            hidden_1 = Dense(256, kernel_initializer="he_normal")(hidden_1)
            hidden_1 = BatchNormalization()(hidden_1)
            hidden_1 = Activation("elu")(hidden_1)
            hidden_1 = Dropout(rate=0.3)(hidden_1)
            hidden_1 = Dense(100, kernel_initializer="he_normal", activation="relu")(
                hidden_1
            )

        elif self.nwk_type == "v3_batch_smaller":
            hidden_1 = BatchNormalization()(nwk_input)
            hidden_1 = Dense(128, kernel_initializer="he_normal")(hidden_1)
            hidden_1 = BatchNormalization()(hidden_1)
            hidden_1 = Activation("elu")(hidden_1)
            hidden_1 = Dropout(rate=0.3)(hidden_1)
            hidden_1 = Dense(128, kernel_initializer="he_normal")(hidden_1)
            hidden_1 = BatchNormalization()(hidden_1)
            hidden_1 = Activation("elu")(hidden_1)
            hidden_1 = Dropout(rate=0.3)(hidden_1)
            hidden_1 = Dense(64, kernel_initializer="he_normal", activation="relu")(
                hidden_1
            )

        elif self.nwk_type == "v3_deep":
            hidden_1 = Dense(256, activation="relu")(nwk_input)
            hidden_1 = Dropout(rate=0.2)(hidden_1)
            hidden_1 = Dense(256, activation="relu")(hidden_1)
            hidden_1 = Dropout(rate=0.2)(hidden_1)
            hidden_1 = Dense(256, activation="relu")(hidden_1)
            hidden_1 = Dropout(rate=0.2)(hidden_1)
            hidden_1 = Dense(256, activation="relu")(hidden_1)
            hidden_1 = Dropout(rate=0.2)(hidden_1)
            hidden_1 = Dense(100, activation="relu")(hidden_1)

        elif self.nwk_type == "v3_deep_batch":
            hidden_1 = BatchNormalization()(nwk_input)
            hidden_1 = Dense(256, kernel_initializer="he_normal")(hidden_1)
            hidden_1 = BatchNormalization()(hidden_1)
            hidden_1 = Activation("elu")(hidden_1)
            hidden_1 = Dropout(rate=0.2)(hidden_1)
            hidden_1 = Dense(256, kernel_initializer="he_normal")(hidden_1)
            hidden_1 = BatchNormalization()(hidden_1)
            hidden_1 = Activation("elu")(hidden_1)
            hidden_1 = Dropout(rate=0.2)(hidden_1)
            hidden_1 = Dense(256, kernel_initializer="he_normal")(hidden_1)
            hidden_1 = BatchNormalization()(hidden_1)
            hidden_1 = Activation("elu")(hidden_1)
            hidden_1 = Dropout(rate=0.2)(hidden_1)
            hidden_1 = Dense(256, kernel_initializer="he_normal")(hidden_1)
            hidden_1 = BatchNormalization()(hidden_1)
            hidden_1 = Activation("elu")(hidden_1)
            hidden_1 = Dropout(rate=0.2)(hidden_1)
            hidden_1 = Dense(100, kernel_initializer="he_normal", activation="relu")(
                hidden_1
            )

        elif self.nwk_type == "v3_deep_bigger_batch":
            hidden_1 = BatchNormalization()(nwk_input)
            hidden_1 = Dense(1024, kernel_initializer="he_normal")(hidden_1)
            hidden_1 = BatchNormalization()(hidden_1)
            hidden_1 = Activation("elu")(hidden_1)
            hidden_1 = Dropout(rate=0.4)(hidden_1)
            hidden_1 = Dense(512, kernel_initializer="he_normal")(hidden_1)
            hidden_1 = BatchNormalization()(hidden_1)
            hidden_1 = Activation("elu")(hidden_1)
            hidden_1 = Dropout(rate=0.4)(hidden_1)
            hidden_1 = Dense(256, kernel_initializer="he_normal")(hidden_1)
            hidden_1 = BatchNormalization()(hidden_1)
            hidden_1 = Activation("elu")(hidden_1)
            hidden_1 = Dropout(rate=0.2)(hidden_1)
            hidden_1 = Dense(128, kernel_initializer="he_normal")(hidden_1)
            hidden_1 = BatchNormalization()(hidden_1)
            hidden_1 = Activation("elu")(hidden_1)
            hidden_1 = Dropout(rate=0.2)(hidden_1)
            hidden_1 = Dense(64, activation="relu")(hidden_1)

        elif self.nwk_type == "v3_deep_bigger":
            hidden_1 = Dense(512, activation="relu")(nwk_input)
            hidden_1 = Dropout(rate=0.3)(hidden_1)
            hidden_1 = Dense(512, activation="relu")(hidden_1)
            hidden_1 = Dropout(rate=0.3)(hidden_1)
            hidden_1 = Dense(256, activation="relu")(hidden_1)
            hidden_1 = Dropout(rate=0.25)(hidden_1)
            hidden_1 = Dense(256, activation="relu")(hidden_1)
            hidden_1 = Dropout(rate=0.25)(hidden_1)
            hidden_1 = Dense(100, activation="relu")(hidden_1)

        elif self.nwk_type == "setup":
            hidden_1 = Dense(128, activation="relu")(nwk_input)
            hidden_1 = Dropout(rate=0.2)(hidden_1)
            hidden_1 = Dense(64, activation="relu")(hidden_1)
            hidden_1 = Dropout(rate=0.2)(hidden_1)
            hidden_1 = Dense(32, activation="relu")(hidden_1)
            hidden_1 = Dropout(rate=0.2)(hidden_1)
            hidden_1 = Dense(16, activation="relu")(hidden_1)
            hidden_1 = Dropout(rate=0.2)(hidden_1)
            hidden_1 = Dense(8, activation="relu")(hidden_1)

        elif self.nwk_type == "setup_batch":
            hidden_1 = BatchNormalization()(nwk_input)
            hidden_1 = Dense(128, kernel_initializer="he_normal")(hidden_1)
            hidden_1 = BatchNormalization()(hidden_1)
            hidden_1 = Activation("elu")(hidden_1)
            hidden_1 = Dropout(rate=0.2)(hidden_1)
            hidden_1 = Dense(64, kernel_initializer="he_normal")(hidden_1)
            hidden_1 = BatchNormalization()(hidden_1)
            hidden_1 = Activation("elu")(hidden_1)
            hidden_1 = Dropout(rate=0.2)(hidden_1)
            hidden_1 = Dense(32, kernel_initializer="he_normal")(hidden_1)
            hidden_1 = BatchNormalization()(hidden_1)
            hidden_1 = Activation("elu")(hidden_1)
            hidden_1 = Dropout(rate=0.2)(hidden_1)
            hidden_1 = Dense(16, kernel_initializer="he_normal")(hidden_1)
            hidden_1 = BatchNormalization()(hidden_1)
            hidden_1 = Activation("elu")(hidden_1)
            hidden_1 = Dropout(rate=0.2)(hidden_1)
            hidden_1 = Dense(8, kernel_initializer="he_normal")(hidden_1)

        elif self.nwk_type == "setup_bigger":
            hidden_1 = Dense(256, activation="relu")(nwk_input)
            hidden_1 = Dense(128, activation="relu")(hidden_0)
            hidden_1 = Dropout(rate=0.2)(hidden_1)
            hidden_1 = Dense(64, activation="relu")(hidden_1)
            hidden_1 = Dense(32, activation="relu")(hidden_1)
            hidden_1 = Dense(16, activation="relu")(hidden_1)
            hidden_1 = Dense(8, activation="relu")(hidden_1)

        elif self.nwk_type == "tests":
            hidden_1 = Dense(2, activation="relu")(nwk_input)

        return Dense(1, activation="relu")(hidden_1)
