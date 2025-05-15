from shiny import App, ui, render
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.metrics import confusion_matrix

y_true = np.load("models/outputs/dlcnn_MS_CLAHE_ytrue.npy")
y_pred = np.load("models/outputs/dlcnn_MS_CLAHE_ypred.npy")

def app_ui(request):
    return ui.page_fluid(
        ui.tags.head(
            ui.include_css("theme/cerulean.min.css")
        ),
        ui.page_navbar(
            ui.nav_panel("Overview", overview_layout()),
            ui.nav_panel("Model", model_layout()),
            title="Double Layer CNN"
        )
    )

def overview_layout():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.navset_pill(
                ui.nav_panel("Cell Images", value="cell"),
                ui.nav_panel("Sampling Methods", value="sampling"),
                ui.nav_panel("CLAHE", value="clahe"),
                id="overview_tab",
                selected="sampling"
            ),
            open="open",
        ),
        ui.output_ui("overview_content")
    )

def model_layout():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.navset_pill(
                ui.nav_panel("Model Explanation", value="explanation"),
                ui.nav_panel("Model Training", value="training"),
                ui.nav_panel("Unseen Data", value="unseen"),
                id="model_tab",
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

def server(input, output, session):

    @output
    @render.ui
    def overview_content():
        match input.overview_tab():
            case "cell":
                return ui.panel_well(
                    ui.h4("Cell Images"),
                    ui.p("In this project, we work with a dataset of Breast Cancer cell images stained using Hematoxylin and Eosin (H&E), provided as part of the Data Bundle on Dropbox. Each image is annotated with a cell identity label (\"Cluster\"), representing its classification based on morphology and type."),
                    ui.HTML('''
<p>
  The dataset originates from the study published in 
  <a href="https://www.nature.com/articles/s41467-023-43458-x" target="_blank">
    Nature Communications (Pati et al., 2023)
  </a>, which provides a high-resolution, spatially-resolved characterization of the tumour microenvironment.
</p>
'''),
                    ui.p("The cell images chosen from the Data Bundle included 4 Immune and 4 Tumour type cells. These individual cell types can be seen in Table 1 below, with their relative class sizes. Although we see a large class imbalance on both the high and low extremes, the following section addresses how this was overcome for the modelling process."),
                    ui.output_table("table1"),
                    ui.HTML('''<p><span style="font-weight: bold; font-size: 0.9rem;">Table 1:</span>
                    <span style="font-size: 0.9rem;">Cell Types and Class Sizes</span></p>''')
                )
            case "sampling":
                return ui.panel_well(
                    ui.h4("Sampling Methods"),
                    ui.p("To address the class imbalances we see between Cell types, specific sampling methods such as Undersampling, Oversampling, and Moderate sampling were used. Each of these sampling methods were completed on all models to see how they can increase or decrease model accuracy when it comes to classifying Cell types. The results of these methods are visualised and explained in the 'Models' tab found in the navigation bar above, and an explanation of how each sampling method was implemented can be seen below."),
                    ui.h5("Undersampling"),
                    ui.p("Between our 8 individual cell types, we have a large range of class sizes, going from ~4,000 for Prolif Invasive Tumours to ~41,000 for Stromal Immune cells. If left untouched, this can create problems during the modelling process as visual models such as CNN read this class imbalance as a level of feature importance meaning it will place higher importance on Stromal features rather than treating all cells the same."),
                    ui.p("To prevent this modelling fault, undersampling can be used to bring all class sizes down to a consistent level, which for our dataset, would be the Prolif Invasive Tumour class  size of 3,775 images. To implement this method, we created an undersampling function which consistently selected random indexes to be removed by first calculating the reduction required for each cell type. By doing this process consistently and randomly, we ensured all models used the same undersampled image set without selection bias via removal."),
                    ui.p("Table 2 shows how each cell type was reduced, as well as how much information was lost as a percentage of the original class size."),
                    ui.output_table("table2"),
                    ui.HTML('''<p><span style="font-weight: bold; font-size: 0.9rem;">Table 2:</span>
                    <span style="font-size: 0.9rem;">Undersampled sizes of Cell Types and Information Loss (%)</span></p>'''),
                    ui.br(),
                    ui.p("It is clear that many cell types, particularly larger types like Stromal and Invasive, lose extreme amount of information. This is problematic, especially when as much data as possible is required to build an accurate classification model. Hence we require a different sampling method such as oversampling which can stop this problem altogether."),
                    ui.h5("Oversampling"),
                    ui.p("")
                )
            case "clahe":
                return ui.panel_well(
                    ui.h4("CLAHE"),
                    ui.p("Contrast Limited Adaptive Histogram Equalization improves the contrast in cell images.")
                )

    @output
    @render.ui
    def table1():
        data = {
            "Individual Cell Type": ["CD4 + T Cells", "CD8 + T Cells", "B Cells", "Stromal", "Invasive", "DCIS 1",
                                     "DCIS 2", "Prolif Invasive"],
            "Binary Cell Type": ["Immune", "Immune", "Immune", "Immune", "Tumour", "Tumour", "Tumour", "Tumour"],
            "Class Size": ["8,453", "6,941", "4,987", "41,424", "34,398", "12,925", "11,719", "3,775"]
        }
        df = pd.DataFrame(data)
        html_table = df.to_html(classes="table table-striped table-bordered table-hover", index=False, border=0)
        
        style = """<style>.table th, .table td {text-align: left !important;}</style>"""
        
        return ui.HTML(f'''{style}<div class="table-responsive">{html_table}</div>''')

    @output
    @render.ui
    def table2():
        data = {
            "Individual Cell Type": ["CD4 + T Cells", "CD8 + T Cells", "B Cells", "Stromal", "Invasive", "DCIS 1",
                                     "DCIS 2", "Prolif Invasive"],
            "Original Class Size": ["8,453", "6,941", "4,987", "41,424", "34,398", "12,925", "11,719", "3,775"],
            "Undersampled Class Size": ["3,775", "3,775", "3,775", "3,775", "3,775", "3,775", "3,775", "3,775"],
            "Information Loss": ["55.3%", "45.6%", "24.3%", "90.9%", "89.0%", "70.8%", "67.8%", "0.00%"]
        }
        df = pd.DataFrame(data)
        html_table = df.to_html(classes="table table-striped table-bordered table-hover", index=False, border=0)
        
        style = """<style>.table th, .table td {text-align: left !important;}</style>"""
        
        return ui.HTML(f'''{style}<div class="table-responsive">{html_table}</div>''')

    @output
    @render.ui
    def model_content():
        match input.model_tab():
            case "explanation":
                return ui.panel_well(
                    ui.h4("Model Explanation"),
                    ui.p("This section explains the architecture of the double-layer CNN.")
                )
            case "training":
                labels = ["Invasive", "DCIS 1", "DCIS 2", "Prolif Invasive"]
                cm_img = cm_image(y_true, y_pred, labels)
                return ui.panel_well(
                    ui.h4("Model Training"),
                    ui.p("We trained our model using augmented and oversampled data for better generalization."),
                    ui.HTML("<h5>Confusion Matrix (Tumour Multiclass)</h5>"),
                    ui.HTML(cm_img)
                )
            case "unseen":
                return ui.panel_well(
                    ui.h4("Unseen Data"),
                    ui.p("This section tests the model's performance on unseen data.")
                )

app = App(app_ui, server)