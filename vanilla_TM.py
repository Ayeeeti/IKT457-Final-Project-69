import argparse
import math

import numpy as np
from scipy import sparse
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import PyHierarchicalTsetlinMachineCUDA.tm as tm
from PyHierarchicalTsetlinMachineCUDA.tm import MultiClassTsetlinMachine


def to_dense(x, dtype=np.uint32):
    """Convert sparse/dense input to a dense array of the given dtype."""
    if sparse.issparse(x):
        x = x.toarray()
    return np.asarray(x).astype(dtype)


def to_labels(y, dtype=np.uint32):
    """Convert labels (possibly sparse / 2D) to a dense 1D array."""
    if sparse.issparse(y):
        y = y.toarray()
    return np.asarray(y).ravel().astype(dtype)


def main():
    parser = argparse.ArgumentParser(description="Vanilla Tsetlin Machine (hierarchical CUDA library)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--clauses", type=int, default=256, help="Number of clauses per class")
    parser.add_argument("--threshold", "--T", dest="threshold", type=int, default=128)
    parser.add_argument("--s", type=float, default=1.0)
    parser.add_argument("--number-of-state-bits", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--append-negated", action="store_true", help="Include negated features")
    parser.add_argument("--binary-inference", action="store_true")
    parser.add_argument("--constant-update-p", action="store_true")
    parser.add_argument("--and-group-normalization", action="store_true")
    parser.add_argument("--no-clipping", action="store_true")
    args = parser.parse_args()

    if min(args.epochs, args.clauses, args.threshold) < 1:
        parser.error("Epochs, clauses, and threshold must be positive.")
    if not math.isfinite(args.s) or args.s < 1:
        parser.error("s must be finite and at least 1.0.")
    if not 0 <= args.seed < 2**32 - 1:
        parser.error("Seed must be between 0 and 4294967294.")

    from pair_generation import (
        X_train_pairs,
        X_val_pairs,
        X_test_pairs,
        y_train_pairs,
        y_val_pairs,
        test_pair_labels,
    )

    X_train = to_dense(X_train_pairs)
    X_val = to_dense(X_val_pairs)
    X_test = to_dense(X_test_pairs)
    y_train = to_labels(y_train_pairs)
    y_val = to_labels(y_val_pairs)
    y_test = to_labels(test_pair_labels)
    features = X_train.shape[1]

    # Vanilla TM: one AND group over all features, then a single AND group of size 1
    hierarchy = (
        (tm.AND_GROUP, features),
        (tm.AND_GROUP, 1),
    )

    np.random.seed(args.seed)
    model = MultiClassTsetlinMachine(
        args.clauses,
        args.threshold,
        args.s,
        binary_inference=args.binary_inference,
        constant_update_p=args.constant_update_p,
        and_group_normalization=args.and_group_normalization,
        seed=args.seed,
        number_of_state_bits=args.number_of_state_bits,
        boost_true_positive_feedback=0,
        no_clipping=args.no_clipping,
        append_negated=args.append_negated,
        hierarchy_structure=hierarchy,
    )
    print(
        f"mode=vanilla, s={args.s}, threshold={args.threshold}, seed={args.seed}, "
        f"clauses_per_class={args.clauses}, state_bits={args.number_of_state_bits}, "
        f"hierarchy={hierarchy}, append_negated={args.append_negated}"
    )

    best_f1 = -1.0
    best_epoch = 0
    best_test_predictions = None

    for epoch in range(1, args.epochs + 1):
        model.fit(X_train, y_train)
        val_predictions = model.predict(X_val)
        score = f1_score(y_val, val_predictions, average="macro")
        print(f"epoch {epoch}: Validation_macro_f1= {score:.4f}")

        # Model selection uses validation only. Test predictions are just stored
        # at the best-validation epoch, so no save()/load() is needed.
        if score > best_f1:
            best_f1 = score
            best_epoch = epoch
            best_test_predictions = model.predict(X_test)

    print(f"best_epoch={best_epoch}")
    print(f"best_validation_macro_f1={best_f1:.4f}")
    print(f"test_accuracy={accuracy_score(y_test, best_test_predictions):.4f}")
    print(f"test_macro_f1={f1_score(y_test, best_test_predictions, average='macro'):.4f}")
    print(confusion_matrix(y_test, best_test_predictions, labels=[0, 1]))


if __name__ == "__main__":
    main()