"""Train the GitHub demo's CUDA TM on the project's Newsgroups pairs."""

import argparse
import math
from PyHierarchicalTsetlinMachineCUDA.tm import MultiClassTsetlinMachine
import PyHierarchicalTsetlinMachineCUDA.tm as tm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--clauses", type=int, default=200, help="Leaf clauses per class")
    parser.add_argument("--threshold", type=int, default=5)
    parser.add_argument("--s", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi-concept", action="store_true", help="Use the demo's AND-OR-AND hierarchy")
    parser.add_argument("--alternatives", type=int, default=4, help="OR alternatives per concept")
    parser.add_argument("--concepts", type=int, default=2, help="Concepts per hierarchical clause")
    parser.add_argument("--append-negated", action="store_true", help="Include negated features")
    args = parser.parse_args()

    if min(args.epochs, args.clauses, args.threshold, args.alternatives, args.concepts) < 1:
        parser.error("Epochs, clauses, threshold, alternatives, and concepts must be positive.")
    if not math.isfinite(args.s) or args.s < 1:
        parser.error("s must be finite and at least 1.0.")
    if not 0 <= args.seed < 2**32 - 1:
        parser.error("Seed must be between 0 and 4294967294.")
    leaves_per_clause = args.alternatives * args.concepts if args.multi_concept else 1
    if args.clauses % leaves_per_clause:
        parser.error("--clauses must be divisible by --alternatives times --concepts in multi-concept mode.")


    import numpy as np
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    from pair_generation import (
        X_train_pairs,
        X_val_pairs,
        X_test_pairs,
        y_train_pairs,
        y_val_pairs,
        test_pair_labels,
    )

    X_train = X_train_pairs.astype(np.uint32).toarray()
    X_val = X_val_pairs.astype(np.uint32).toarray()
    y_train = y_train_pairs.astype(np.uint32)
    features = X_train.shape[1]

    if args.multi_concept:
        architecture = "multi-concept"
        hierarchy = (
            (tm.AND_GROUP, features),
            (tm.OR_ALTERNATIVES, args.alternatives),
            (tm.AND_ALTERNATIVES, args.concepts),
        )
    else:
        architecture = "vanilla"
        hierarchy = ((tm.AND_GROUP, features), (tm.AND_GROUP, 1))

    # The demo matches the number of leaf clauses across architectures.
    root_clauses = args.clauses // leaves_per_clause
    # The CUDA class uses NumPy's global RNG to shuffle training examples.
    np.random.seed(args.seed)
    model = MultiClassTsetlinMachine(
        root_clauses,
        args.threshold,
        args.s,
        hierarchy_structure=hierarchy,
        boost_true_positive_feedback=0,
        append_negated=args.append_negated,
        seed=args.seed,
    )
    print(
        f"mode={architecture}, s={args.s}, threshold={args.threshold}, seed={args.seed}, "
        f"leaf_clauses_per_class={args.clauses}, root_clauses_per_class={root_clauses}, "
        f"hierarchy={hierarchy}, append_negated={args.append_negated}"
    )

    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        score = f1_score(y_val_pairs, predictions, average="macro")
        print(f"epoch {epoch}: Validation_macro_f1= {score:.4f}")
        if score > best_f1:
            best_f1 = score
            best_epoch = epoch
            best_state = model.save()

    model.load(best_state)
    predictions = model.predict(X_test_pairs.astype(np.uint32).toarray())
    print(f"best_epoch={best_epoch}")
    print(f"best_validation_macro_f1={best_f1:.4f}")
    print(f"test_accuracy={accuracy_score(test_pair_labels, predictions):.4f}")
    print(f"test_macro_f1={f1_score(test_pair_labels, predictions, average='macro'):.4f}")
    print(confusion_matrix(test_pair_labels, predictions, labels=[0, 1]))


if __name__ == "__main__":
    main()
