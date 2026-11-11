import numpy as np
import argparse


class VanillaTM:
    """Binary, unweighted Tsetlin Machine with flat conjunctions"""

    def __init__(self, n_features, clauses=100, threshold=15, states=100, seed=42, s=1.0):
        # Error checking
        if n_features < 1 or clauses < 2 or clauses % 2 or threshold < 1:
            raise ValueError("Use positive features and threshold, and an even clause count.")
        if not 1 <= states <= np.iinfo(np.int16).max // 2:
            raise ValueError("states is outside the supported range.")

        if not np.isfinite(s) or s < 1.0:
            raise ValueError("s must be finite and at least 1.0.")

        self.inverse_s = 1 / s
        self.states = states
        self.threshold = threshold
        self.rng = np.random.default_rng(seed)
        # Each clause has one automaton for each original and negated feature
        # States at or below the midpoint exclude the literal. Higher states include it
        self.automata = np.full((clauses, 2 * n_features), states, dtype=np.int16)
        # Alternate positive votes for class 1 and negative votes for class 0
        self.polarity = np.tile([1, -1], clauses // 2)

    def _literals(self, X):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] * 2 != self.automata.shape[1]:
            raise ValueError("X must be a matrix with the configured feature width.")
        if not np.all((X == 0) | (X == 1)):
            raise ValueError("X must contain binary values.")
        # For a binary feature, its negation is true exactly when the feature is zero
        return np.concatenate((X == 1, X == 0), axis=1)

    @staticmethod
    def _fires(included, literals):
        # A clause fires unless an included literal is false
        # An empty clause fires here, allowing feedback before any literals are included
        return ~np.any(included & ~literals, axis=1)

    def fit_epoch(self, x, y):
        literals = self._literals(x)
        y = np.asarray(y)
        if y.shape != (len(literals),) or not np.all((y == 0) | (y == 1)):
            raise ValueError("y must contain binary label per row.")

        # Shuffle the training examples each epoch and update after each example
        for index in self.rng.permutation(len(y)):
            row = literals[index]
            included = self.automata > self.states
            fires = self._fires(included, row)
            # Sum the signed votes. Clipping keeps the feedback probability in [0, 1]
            vote = np.clip(fires @ self.polarity, -self.threshold, self.threshold)
            target = 1 if y[index] == 1 else -1
            # Feedback becomes less likely as the vote moves toward the target threshold
            probability = (self.threshold - target * vote) / (2 * self.threshold)
            # Draw independently for every clause, even those that did not fire
            feedback = self.rng.random(fires.size) < probability

            # Type I trains clauses that vote for the correct class
            # Reinforce true literals in firing clauses with probability 1 - 1/s
            # Move other literals toward exclusion with probability 1/s
            type_i = feedback & (self.polarity == target)
            true_in_firing_clause = fires[:, None] & row
            type_i_random = self.rng.random(self.automata.shape)
            decrement = (
                type_i[:, None] & ~true_in_firing_clause
                & (type_i_random < self.inverse_s)
            )
            reinforce = (
                type_i[:, None] & true_in_firing_clause
                & (type_i_random >= self.inverse_s)
            )

            # Type II acts on firing clauses that vote for the wrong class
            # Move excluded false literals toward inclusion so they can block this input
            type_ii = feedback & (self.polarity != target) & fires
            suppress = type_ii[:, None] & ~included & ~row
            increment = reinforce | suppress
            self.automata -= decrement
            self.automata += increment
            # Keep automaton states within the valid range after both updates
            np.clip(self.automata, 1, 2 * self.states, out=self.automata)
        return self

    def predict(self, x):
        literals = self._literals(x)
        included = self.automata > self.states
        # Ignore empty clauses at prediction time. A tied vote predicts class 1
        nonempty = included.any(axis=1)
        return np.fromiter(
            ((self._fires(included, row) & nonempty) @ self.polarity >= 0
             for row in literals),
            dtype=np.uint8,
            count=len(literals),
        )


def main():
    parser = argparse.ArgumentParser(description="Train a binary vanilla Tsetlin Machine")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--clauses", type=int, default=100, help="Number of clauses")
    parser.add_argument("--threshold", type=int, default=15, help="Voting threshold")
    parser.add_argument("--s", type=float, default=21.1, help="Feedback specificity")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("Number of epochs must be positive.")


    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    from pair_generation import (
        X_train_pairs,
        y_train_pairs,
        X_val_pairs,
        y_val_pairs,
        X_test_pairs,
        test_pair_labels,
    )

    X_train = X_train_pairs.astype(np.uint8).toarray()
    X_val = X_val_pairs.astype(np.uint8).toarray()
    X_test = X_test_pairs.astype(np.uint8).toarray()
    model = VanillaTM(
        X_train.shape[1], args.clauses, args.threshold, seed=args.seed, s=args.s
    )
    best_f1 = -1
    print(f"s = {args.s}, clauses = {args.clauses}, threshold = {args.threshold}, seed={args.seed}")
    for epoch in range(1, args.epochs + 1):
        model.fit_epoch(X_train, y_train_pairs)
        predictions = model.predict(X_val)
        score = f1_score(y_val_pairs, predictions, average="macro")
        print(f"epoch {epoch}: Validation_macro_f1= {score:.4f}")

        # Copy the best validation states so subsequent updates cannot overwrite them
        if score > best_f1:
            best_f1 = score
            best_epoch = epoch
            best_automata = model.automata.copy()

    # Restore the best validation epoch before scoring the held-out test pairs
    model.automata = best_automata
    predictions = model.predict(X_test)
    print(f"best_epoch={best_epoch}")
    print(f"test_accuracy={accuracy_score(test_pair_labels, predictions):.4f}")
    print(f"test_macro_f1={f1_score(test_pair_labels, predictions, average='macro'):.4f}")
    print(confusion_matrix(test_pair_labels, predictions, labels=[0, 1]))


if __name__ == "__main__":
    main()
