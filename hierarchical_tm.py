import argparse
import numpy as np

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from PyHierarchicalTsetlinMachineCUDA.tm import MultiClassTsetlinMachine
import PyHierarchicalTsetlinMachineCUDA.tm as htm

from pair_generation import (
    X_train_pairs,
    y_train_pairs,
    X_val_pairs,
    y_val_pairs,
    X_test_pairs,
    test_pair_labels,
)


def main():
    parser = argparse.ArgumentParser(
        description="Train a Hierarchical Tsetlin Machine"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--clauses",
        type=int,
        default=20,
        help="Number of clauses"
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=128,
        help="Voting threshold T"
    )

    parser.add_argument(
        "--s",
        type=float,
        default=21.1,
        help="Feedback specificity"
    )

    parser.add_argument(
        "--alternatives",
        type=int,
        default=64,
        help="Number of OR alternatives"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("Number of epochs must be positive.")

    # Reproducibility
    np.random.seed(args.seed)

    # Convert sparse pair matrices to dense binary arrays
    X_train = X_train_pairs.astype(np.uint32).toarray()
    X_val = X_val_pairs.astype(np.uint32).toarray()
    X_test = X_test_pairs.astype(np.uint32).toarray()

    y_train = y_train_pairs.astype(np.uint32)
    y_val = y_val_pairs.astype(np.uint32)
    y_test = test_pair_labels.astype(np.uint32)

    print("Training:", X_train.shape, y_train.shape)
    print("Validation:", X_val.shape, y_val.shape)
    print("Testing:", X_test.shape, y_test.shape)

    # Hierarchy:
    # input features -> OR alternatives -> two learned concepts
    hierarchy = (
        (htm.AND_GROUP, X_train.shape[1]),
        (htm.OR_ALTERNATIVES, args.alternatives),
        (htm.AND_ALTERNATIVES, 2),
    )

    model = MultiClassTsetlinMachine(
        number_of_clauses=args.clauses,
        T=args.threshold,
        s=args.s,
        number_of_state_bits=8,
        boost_true_positive_feedback=0,
        append_negated=False,
        hierarchy_structure=hierarchy,
        seed=args.seed,
    )

    print()
    print(
        f"s={args.s}, "
        f"clauses={args.clauses}, "
        f"T={args.threshold}, "
        f"alternatives={args.alternatives}, "
        f"seed={args.seed}"
    )

    print("Hierarchy:", hierarchy)
    print()

    best_f1 = -1
    best_accuracy = -1
    best_epoch = 0
    best_state = None

    for epoch in range(1, args.epochs + 1):

        model.fit(X_train, y_train)

        val_predictions = model.predict(X_val)

        val_accuracy = accuracy_score(
            y_val,
            val_predictions
        )

        val_f1 = f1_score(
            y_val,
            val_predictions,
            average="macro"
        )

        print(
            f"epoch {epoch}: "
            f"Validation_accuracy={val_accuracy:.4f}, "
            f"Validation_macro_f1={val_f1:.4f}"
        )

        # Use macro F1 for model selection, like the Vanilla implementation
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_accuracy = val_accuracy
            best_epoch = epoch

            # Save the best HTM state
            best_state = model.save()

    print()
    print(f"best_epoch={best_epoch}")
    print(f"best_validation_accuracy={best_accuracy:.4f}")
    print(f"best_validation_macro_f1={best_f1:.4f}")

    # Restore best validation model
    if best_state is not None:
        model.load(best_state)

    # Final evaluation on held-out test data
    test_predictions = model.predict(X_test)

    print()
    print(
        f"test_accuracy="
        f"{accuracy_score(y_test, test_predictions):.4f}"
    )

    print(
        f"test_macro_f1="
        f"{f1_score(y_test, test_predictions, average='macro'):.4f}"
    )

    print("confusion_matrix:")
    print(
        confusion_matrix(
            y_test,
            test_predictions,
            labels=[0, 1]
        )
    )


if __name__ == "__main__":
    main()