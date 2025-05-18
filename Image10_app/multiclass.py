from shiny import App, ui, render
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.metrics import confusion_matrix

y_true = np.load("models/outputs/multiclass_NONCLAHE_Training_ytrue.npy")
y_pred = np.load("models/outputs/multiclass_NONCLAHE_Training_ypred.npy")
y_pred = np.argmax(y_pred, axis=1)

y_true_final = np.load("models/outputs/multiclass_NONCLAHE_ytrue1.npy")
y_pred_final = np.load("models/outputs/multiclass_NONCLAHE_ypred1.npy")
y_pred_final = np.argmax(y_pred_final, axis=1)

history_df = pd.read_csv("models/history/multiclass_NONCLAHE_history.csv")

y_CLAHE_true = np.load("models/outputs/multiclass_CLAHE_Training_ytrue.npy")
y_CLAHE_pred = np.load("models/outputs/multiclass_CLAHE_Training_ypred.npy")
y_CLAHE_pred = np.argmax(y_CLAHE_pred, axis=1)

y_CLAHE_true_final = np.load("models/outputs/multiclass_CLAHE_ytrue.npy")
y_CLAHE_pred_final = np.load("models/outputs/multiclass_CLAHE_ypred.npy")
y_CLAHE_pred_final = np.argmax(y_CLAHE_pred_final, axis=1)

history_CLAHE_df = pd.read_csv("models/history/multiclass_CLAHE_history.csv")


def app_ui(request):
    return ui.page_fluid(
        ui.tags.head(
            ui.include_css("theme/cerulean.min.css")
        ),
        ui.page_navbar(
            ui.nav_panel("Model", model_layout()),
            title="Multiclass CNN"
        )
    )

def model_layout():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.navset_pill(
                ui.nav_panel("Model Explanation", value="explanation"),
                ui.nav_panel("Model Training", value="training"),
                ui.nav_panel("Unseen Data", value="unseen"),
                id="model_tab",
                selected="explanation"
            ),
            open="open",
        ),
        ui.output_ui("model_content")
    )

def cm_image(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close(fig)
    return f"<img src='data:image/png;base64,{img_base64}' class='img-fluid'/>"

def history_plot(history_df, metric="loss"):
    fig, ax = plt.subplots()
    ax.plot(history_df["multiclass_output_" + metric], label=f"Train Multiclass {metric.capitalize()}")
    ax.plot(history_df["val_multiclass_output_" + metric], label=f"Val Multiclass {metric.capitalize()}")
    ax.set_title(f"{metric.capitalize()} over Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.capitalize())
    ax.legend()
    plt.tight_layout()
    plt.grid(alpha=0.5)

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close(fig)
    return f"<img src='data:image/png;base64,{img_base64}' class='img-fluid'/>"

def server(input, output, session):
        @output
        @render.ui
        def model_content():
            match input.model_tab():
                case "explanation":
                    return ui.panel_well(
                        ui.h4("Model Explanation"),
                        ui.p("This section explains the architecture of the Multiclass CNN."),
                        ui.p("For this section, we wanted to use CNN's to help predict whether cells are Immune, Prolif Invasive, Invasive, DCIS_1 or DCIS_2. A highly accurate CNN here we felt would be incredibly useful to those in Cancer screening, not only being able to predict whether cells are cancerous, but if in the rare case they are, whether it is severe, or urgent, and help to gauge the timeframe of treating the patient."),
                        ui.p("To do so, we educated ourselves on various image augmentation techniques, and found the CLAHE method to have high potential. To really gauge it's effect we used respective CLAHE and NON-CLAHE models, continuing to refine them with other useful methods we produced such as mixed sampling which lessens overfitting and through comparison were able to produce one final model for clahe and one for non-clahe."),
                        ui.p("These are outlined in corresponding sections below.")
                    )
                case "training":
                    labels = ["Immune", "Invasive", "DCIS 1", "DCIS 2", "Prolif Invasive"]
                    cm_img = cm_image(y_true, y_pred, labels)
                    loss_plot = history_plot(history_df, "loss")
                    accuracy_plot = history_plot(history_df, "accuracy")

                    cm_CLAHE_img = cm_image(y_CLAHE_true, y_CLAHE_pred, labels)
                    loss_CLAHE_plot = history_plot(history_CLAHE_df, "loss")
                    accuracy_CLAHE_plot = history_plot(history_CLAHE_df, "accuracy")

                    return ui.panel_well(
                        ui.h4("Model Training"),
                        ui.p("We trained our model using augmented and oversampled data for better generalization."),
                        ui.navset_tab(
                            ui.nav_panel("Non-Clahe", 
                                            ui.layout_columns(
                                                ui.card(
                                                    ui.h5("Confusion Matrix Non-Clahe (Multiclass)"),
                                                    ui.HTML(cm_img)
                                                ),
                                                ui.card(
                                                    ui.h5("Training Performance"),
                                                    ui.navset_tab(
                                                        ui.nav_panel("Loss", ui.HTML(loss_plot)),
                                                        ui.nav_panel("Accuracy", ui.HTML(accuracy_plot)),
                                                        id="metric_tabs"
                                                    )
                                                )
                                            )
                                         ),
                            ui.nav_panel("Clahe", 
                                            ui.layout_columns(
                                                ui.card(
                                                    ui.h5("Confusion Matrix Clahe (Multiclass)"),
                                                    ui.HTML(cm_CLAHE_img)
                                                ),
                                                ui.card(
                                                    ui.h5("Training Performance"),
                                                    ui.navset_tab(
                                                        ui.nav_panel("Loss", ui.HTML(loss_CLAHE_plot)),
                                                        ui.nav_panel("Accuracy", ui.HTML(accuracy_CLAHE_plot)),
                                                        id="metric_tabs"
                                                    )
                                                )
                                            )
                                         )
                        )

                    )
                case "unseen":
                    labels = ["Immune", "Invasive", "DCIS 1", "DCIS 2", "Prolif Invasive"]
                    cm_img = cm_image(y_true_final, y_pred_final, labels)
                    cm_CLAHE_img = cm_image(y_CLAHE_true_final, y_CLAHE_pred_final, labels)
                    return ui.panel_well(
                        ui.h4("Model Training"),
                        ui.p("We trained our model using augmented and oversampled data for better generalization."),
                        ui.layout_columns(
                            ui.card(
                                ui.h5("Confusion Matrix Non-Clahe (Multiclass)"),
                                ui.HTML(cm_img)
                            ),
                            ui.card(
                                ui.h5("Confusion Matrix Clahe (Multiclass)"),
                                ui.HTML(cm_CLAHE_img)
                            )
                        )
                    )

app = App(app_ui, server)