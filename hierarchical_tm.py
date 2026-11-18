import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
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
    vectorizer,
)


# Convert a relational feature index back to a readable feature name.
# Features 0-999 represent words shared by both documents.
# Features 1000-1999 represent words appearing in only one document.
def relational_feature_name(feature_id, feature_names):
    vocabulary_size = len(feature_names)

    if feature_id < vocabulary_size:
        return f"shared:{feature_names[feature_id]}"

    return f"different:{feature_names[feature_id - vocabulary_size]}"


# Extract the learned literals from each HTM class and create
# visualizations showing which relational features are used most often.
def visualize_clauses(
    model,
    feature_names,
    output_directory="clause_visualizations",
    top_n=20,
):
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    class_names = {
        0: "different_category",
        1: "same_category",
    }

    total_features = len(feature_names) * 2

    print()
    print("Creating clause visualizations...")

    # MultiClassTsetlinMachine contains one Tsetlin Machine per class.
    for class_id, class_name in class_names.items():
        tm = model.tms[class_id]

        # Shape:
        # (number of clauses, number of hierarchy components, literals per component)
        literals = tm.get_literals()

        # Maps the literal positions in each hierarchy component
        # back to the original input feature indices.
        feature_map = tm.map_ta_id_to_feature_id()

        number_of_clauses = literals.shape[0]

        # Count how often every relational feature is included in each clause.
        feature_usage = np.zeros(
            (number_of_clauses, total_features),
            dtype=np.int64,
        )

        for clause_id in range(number_of_clauses):
            component_ids, literal_ids = np.nonzero(
                literals[clause_id]
            )

            mapped_features = feature_map[
                component_ids,
                literal_ids,
            ]

            np.add.at(
                feature_usage[clause_id],
                mapped_features,
                1,
            )

        # Sum usage over all clauses in this class.
        total_usage = feature_usage.sum(axis=0)

        # Keep only features that actually occur in learned clauses.
        used_features = np.flatnonzero(total_usage > 0)

        if len(used_features) == 0:
            print(
                f"No included literals found for class {class_id}."
            )
            continue

        # Select the most frequently included relational features.
        sorted_features = used_features[
            np.argsort(total_usage[used_features])[::-1]
        ]

        top_feature_ids = sorted_features[:top_n]

        top_feature_labels = [
            relational_feature_name(
                feature_id,
                feature_names,
            )
            for feature_id in top_feature_ids
        ]

        top_feature_counts = total_usage[top_feature_ids]

        print()
        print(
            f"CLASS {class_id} - "
            f"{class_name.replace('_', ' ').title()}"
        )

        for feature, count in zip(
            top_feature_labels[:10],
            top_feature_counts[:10],
        ):
            print(
                f"{feature}: {count} inclusions"
            )

        # ---------------------------------------------------------
        # Save total feature usage as CSV
        # ---------------------------------------------------------
        feature_csv_path = (
            output_path
            / f"class_{class_id}_feature_usage.csv"
        )

        with feature_csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            writer = csv.writer(csv_file)

            writer.writerow([
                "feature",
                "total_inclusions",
            ])

            for feature_id in sorted_features:
                writer.writerow([
                    relational_feature_name(
                        feature_id,
                        feature_names,
                    ),
                    int(total_usage[feature_id]),
                ])

        # ---------------------------------------------------------
        # Save a readable summary of every clause
        # ---------------------------------------------------------
        clause_csv_path = (
            output_path
            / f"class_{class_id}_clause_summary.csv"
        )

        with clause_csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            writer = csv.writer(csv_file)

            writer.writerow([
                "clause",
                "clause_weight",
                "total_literal_inclusions",
                "top_features",
            ])

            for clause_id in range(number_of_clauses):
                clause_usage = feature_usage[clause_id]

                clause_features = np.flatnonzero(
                    clause_usage > 0
                )

                clause_features = clause_features[
                    np.argsort(
                        clause_usage[clause_features]
                    )[::-1]
                ]

                # Keep the most important features for a readable summary.
                clause_top_features = []

                for feature_id in clause_features[:10]:
                    feature_label = relational_feature_name(
                        feature_id,
                        feature_names,
                    )

                    count = clause_usage[feature_id]

                    clause_top_features.append(
                        f"{feature_label} ({count})"
                    )

                writer.writerow([
                    clause_id,
                    tm.get_clause_weight(clause_id),
                    int(clause_usage.sum()),
                    "; ".join(clause_top_features),
                ])

        # ---------------------------------------------------------
        # Bar chart:
        # most frequently included relational features
        # ---------------------------------------------------------
        plt.figure(figsize=(10, 8))

        # Reverse the order so the largest feature appears at the top.
        plt.barh(
            top_feature_labels[::-1],
            top_feature_counts[::-1],
        )

        plt.xlabel(
            "Number of inclusions in learned clause components"
        )

        plt.ylabel(
            "Relational feature"
        )

        plt.title(
            f"Most frequently included HTM features - "
            f"Class {class_id} "
            f"({class_name.replace('_', ' ')})"
        )

        plt.tight_layout()

        bar_chart_path = (
            output_path
            / f"class_{class_id}_top_features.png"
        )

        plt.savefig(
            bar_chart_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        # ---------------------------------------------------------
        # Heatmap:
        # shows how the top features are distributed across clauses
        # ---------------------------------------------------------
        heatmap_data = feature_usage[
            :,
            top_feature_ids,
        ]

        plt.figure(figsize=(14, 8))

        image = plt.imshow(
            heatmap_data,
            aspect="auto",
        )

        plt.colorbar(
            image,
            label="Number of hierarchy components including feature",
        )

        plt.xticks(
            np.arange(len(top_feature_labels)),
            top_feature_labels,
            rotation=90,
        )

        plt.yticks(
            np.arange(number_of_clauses),
            [
                f"Clause {clause_id}"
                for clause_id in range(number_of_clauses)
            ],
        )

        plt.xlabel(
            "Relational feature"
        )

        plt.ylabel(
            "Clause"
        )

        plt.title(
            f"HTM clause feature usage - "
            f"Class {class_id} "
            f"({class_name.replace('_', ' ')})"
        )

        plt.tight_layout()

        heatmap_path = (
            output_path
            / f"class_{class_id}_clause_heatmap.png"
        )

        plt.savefig(
            heatmap_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

    print()
    print(
        f"Clause visualizations saved to: "
        f"{output_path.resolve()}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train a Hierarchical Tsetlin Machine"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
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
        default=40.0,
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

    np.random.seed(args.seed)

    # The pair generation file already converts each pair from
    # [Document A | Document B] into the relational representation
    # [shared words | different words].
    X_train = X_train_pairs.astype(np.uint32).toarray()
    X_val = X_val_pairs.astype(np.uint32).toarray()
    X_test = X_test_pairs.astype(np.uint32).toarray()

    y_train = y_train_pairs.astype(np.uint32)
    y_val = y_val_pairs.astype(np.uint32)
    y_test = test_pair_labels.astype(np.uint32)

    print(
        "Using relational features: "
        "[shared | different]"
    )

    print(
        "Training:",
        X_train.shape,
        y_train.shape,
    )

    print(
        "Validation:",
        X_val.shape,
        y_val.shape,
    )

    print(
        "Testing:",
        X_test.shape,
        y_test.shape,
    )

    # Hierarchy:
    # input features -> OR alternatives -> two learned concepts
    hierarchy = (
        (htm.AND_GROUP, X_train.shape[1]),
        (
            htm.OR_ALTERNATIVES,
            args.alternatives,
        ),
        (
            htm.AND_ALTERNATIVES,
            2,
        ),
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

    print(
        "Hierarchy:",
        hierarchy,
    )

    print()

    best_f1 = -1
    best_accuracy = -1
    best_epoch = 0
    best_state = None

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.fit(
            X_train,
            y_train,
        )

        val_predictions = model.predict(
            X_val
        )

        val_accuracy = accuracy_score(
            y_val,
            val_predictions,
        )

        val_f1 = f1_score(
            y_val,
            val_predictions,
            average="macro",
        )

        print(
            f"epoch {epoch}: "
            f"Validation_accuracy="
            f"{val_accuracy:.4f}, "
            f"Validation_macro_f1="
            f"{val_f1:.4f}"
        )

        # Select the best model using validation macro F1.
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_accuracy = val_accuracy
            best_epoch = epoch

            best_state = model.save()

    print()

    print(
        f"best_epoch="
        f"{best_epoch}"
    )

    print(
        f"best_validation_accuracy="
        f"{best_accuracy:.4f}"
    )

    print(
        f"best_validation_macro_f1="
        f"{best_f1:.4f}"
    )

    # Restore the model state that achieved the best validation F1.
    if best_state is not None:
        model.load(
            best_state
        )

    # Evaluate the selected model on the training set.
    train_predictions = model.predict(
        X_train
    )

    print()

    print(
        f"train_accuracy="
        f"{accuracy_score(y_train, train_predictions):.4f}"
    )

    print(
        f"train_macro_f1="
        f"{f1_score(y_train, train_predictions, average='macro'):.4f}"
    )

    # Final evaluation on the held-out test set.
    test_predictions = model.predict(
        X_test
    )

    print()

    print(
        f"test_accuracy="
        f"{accuracy_score(y_test, test_predictions):.4f}"
    )

    print(
        f"test_macro_f1="
        f"{f1_score(y_test, test_predictions, average='macro'):.4f}"
    )

    print(
        "confusion_matrix:"
    )

    print(
        confusion_matrix(
            y_test,
            test_predictions,
            labels=[0, 1],
        )
    )

    # Get the vocabulary that was fitted only on the training documents.
    feature_names = vectorizer.get_feature_names_out()

    # Visualize and save the learned clauses from the selected best model.
    visualize_clauses(
        model,
        feature_names,
        top_n=20,
    )


if __name__ == "__main__":
    main()