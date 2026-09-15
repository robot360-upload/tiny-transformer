import numpy as np


class CrossEntropyLoss:
    def forward(self, probabilities, targets):
        batch_size = targets.shape[0]

        correct_probabilities = probabilities[
            np.arange(batch_size),
            targets
        ]

        correct_probabilities = np.clip(
            correct_probabilities,
            1e-12,
            1.0
        )

        loss = -np.mean(
            np.log(correct_probabilities)
        )

        return loss
