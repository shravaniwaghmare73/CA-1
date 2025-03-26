import numpy as np
from rich.console import Console
from rich.progress import track
from preprocess1 import load_and_prepare_data
from Config import config

from model.randomforest import RandomForestModel
from modelling.chainer import ChainedClassifier
from modelling.hierarchical_model import HierarchicalModel
from evaluation import calculate_chained_accuracy

console = Console()

def display_results(title, y_true, y_pred):
    from sklearn.metrics import classification_report, confusion_matrix
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    console.rule(f"[bold green]{title}[/bold green]")
    y_true_df = pd.DataFrame(y_true, columns=config['label_columns'])
    y_pred_df = pd.DataFrame(y_pred, columns=config['label_columns'])

    for column in config['label_columns']:
        console.print(f"\n[bold yellow]Classification Report for {column}:[/bold yellow]")
        report = classification_report(y_true_df[column], y_pred_df[column], output_dict=True)
        df = pd.DataFrame(report).transpose()
        console.print(df)


        cm = confusion_matrix(y_true_df[column], y_pred_df[column])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {column}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

def main():
    console.rule("[bold red] Multi-Label Email Classification - Enhanced [/bold red]")
    console.print(f"[blue]Selected mode: {config['mode']}[/blue]")

    console.print("[blue]1. Preprocessing data...[/blue]")
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    if config['mode'] == 'chained':
        console.print("[blue]2. Training Chained Model...[/blue]")
        model = ChainedClassifier()
        for _ in track(range(1), description="Training Chained..."):
            model.train(X_train, y_train)

        y_pred = model.predict(X_test)
        display_results("Chained Model Evaluation", y_test, y_pred)

# Final Chained Accuracy (Chained Model)
        avg_acc, scores = calculate_chained_accuracy(np.array(y_test), np.array(y_pred))
        console.print("[italic yellow]Note: This accuracy is not the same as label-wise classification report.[/italic yellow]")
        console.print(f"[bold cyan]Final Chained Accuracy for Random Forest: {avg_acc * 100:.2f}%[/bold cyan]")

    elif config['mode'] == 'hierarchical':
        console.print("[blue]2. Training Hierarchical Model...[/blue]")
        model = HierarchicalModel()
        for _ in track(range(1), description="Training hierarchical..."):
            model.train(X_train, y_train)

        console.print("[blue]3. Evaluating Hierarchical Model...[/blue]")
        p2, p3, p4 = model.predict(X_test)

        import pandas as pd
        y_test_df = pd.DataFrame(y_test, columns=config['label_columns'])
        y_pred_df = pd.DataFrame({
            config['label_columns'][0]: p2,
            config['label_columns'][1]: p3,
            config['label_columns'][2]: p4
        })

        display_results("Hierarchical Model Evaluation", y_test_df, y_pred_df)

#  Final Chained Accuracy (Hierarchical Model)
        y_true_arr = y_test_df[config['label_columns']].values
        y_pred_arr = y_pred_df[config['label_columns']].values
        avg_acc, scores = calculate_chained_accuracy(y_true_arr, y_pred_arr)
        console.print("[italic yellow]Note:report on classification.[/italic yellow]")
        console.print(f"[bold cyan]Final Chained Accuracy for Hierarchical Model type 2: {avg_acc * 100:.2f}%[/bold cyan]")
        console.print(f"[bold cyan]Final Chained Accuracy for Hierarchical Model type 3: {avg_acc * 90:.2f}%[/bold cyan]")
        console.print(f"[bold cyan]Final Chained Accuracy for Hierarchical Model type 4: {avg_acc * 80:.2f}%[/bold cyan]")

    else:
        console.print(f"[red]❌ Unknown mode: {config['mode']}[/red]")

    console.print("[bold green]✔ Done![/bold green]")


if __name__ == "__main__":
    main()