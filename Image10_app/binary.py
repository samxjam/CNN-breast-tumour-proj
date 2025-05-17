from shiny import App, ui, render
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.metrics import confusion_matrix

# Missing: history of binary model (csv), y_pred output, y_true output, and confusion matrix (npy files)
# y_true = np.load("models/outputs/binary_MS_CLAHE_ytrue.npy")
# y_pred = np.load("models/outputs/binary_MS_CLAHE_ypred.npy")
# history_df = pd.read_csv("models/history/binary_MS_CLAHE_history.csv")

def app_ui(request):
    return ui.page_fluid(
        ui.tags.head(
            ui.include_css("theme/cerulean.min.css")
        ),
        ui.page_navbar(
            ui.nav_panel("Model", model_layout()),
            title="Binary CNN"
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

'''
Copied directly from fellow group mates
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
'''

# history plot needs to be made with reference to the binary model - I might need to see the original code

def server(input, output, session):
        @output
        @render.ui
        def model_content():
            match input.model_tab():
                case "explanation":
                    return ui.panel_well(
                        ui.h4("Model Explanation"),
                        ui.p("This section explains the architecture of the Binary CNN."),
                        ui.p("The binary CNN was the first and most basic model we built. Its purpose was to distinguish between cancer cells and immune cells - if it was unable to do this reliably, it would have made further work impossible."),
                        ui.p("The description of the training process and the final evalutation of this model's effectiveness can be found in the other sections of this app.")
                    )
                case "training":
                    return ui.panel_well(
                        ui.h4("Model Training"),
                        ui.p("We trained our model using augmented and oversampled data for better generalization."),
                        ui.p("The immune class consists of images from four immune cell types - B Cells, Stromal cells, and T cells (TD4+ and TD8+). The tumour class consists of images of cells from different stages of cancer - DCIS-1, DCIS-2, Invasive, and Prolif-Invasive."),
                        ui.layout_columns(
                            ui.card(
                                ui.h5("Confusion Matrix (Binary CNN)"),
                                # place image here
                            ),
                            ui.card(
                                ui.h5("Training Performance"),
                                ui.navset_tab(
                                    ui.nav_panel("Loss"), # insert plot inside brackets
                                    ui.nav_panel("Accuracy"), # insert plot inside brackets
                                    id="metric_tabs"
                                )
                            )
                        )
                    )
                case "unseen":
                    return ui.panel_well(
                        ui.h4("Unseen Data"),
                        ui.p("This section tests the model's performance on unseen data."),
                        ui.p("When asking the model to distinguish between these immune cells and the main tumour cells, the model is highly reliable. However, it does not cover all the immune cell types, so it may not be as accurate when dealing with cells not used for its training.")
                    )

app = App(app_ui, server)
