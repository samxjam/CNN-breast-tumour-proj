from shiny import App, ui, render
import pandas as pd

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
                id="overview_tab"
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

def server(input, output, session):

    @output
    @render.ui
    def overview_content():
        match input.overview_tab():
            case "cell":
                return ui.panel_well(
                    ui.h4("Cell Images"),
                    ui.p("In this project, we work with a dataset of Breast Cancer cell images stained using Hematoxylin and Eosin (H&E), provided as part of the Data Bundle. Each image is annotated with a cell identity label (\"Cluster\"), representing its classification based on morphology and type."),
                    ui.HTML('''
<p>
  The dataset originates from the study published in 
  <a href="https://www.nature.com/articles/s41467-023-43458-x" target="_blank">
    Nature Communications (Pati et al., 2023)
  </a>, which provides a high-resolution, spatially-resolved characterization of the tumour microenvironment.
</p>
'''),
                    ui.p("The cell images chosen from the Data Bundle included 4 Immune and 4 Tumour type cells. These individual cell types can be seen in Table 1 below, with their relative class sizes. Although we see a large class imbalance on both the high and low extremes, the following section addresses how this was overcome for the modelling process."),
                    ui.output_table("my_table"),
                    ui.HTML('''<p><span style="font-weight: bold; font-size: 0.9rem;">Table 1:</span>
                    <span style="font-size: 0.9rem;">Cell Types and Class Sizes</span></p>''')
                )
            case "sampling":
                return ui.panel_well(
                    ui.h4("Sampling Methods"),
                    ui.p("We use moderate oversampling to handle class imbalance in the dataset.")
                )
            case "clahe":
                return ui.panel_well(
                    ui.h4("CLAHE"),
                    ui.p("Contrast Limited Adaptive Histogram Equalization improves the contrast in cell images.")
                )

    @output
    @render.ui
    def my_table():
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
    def model_content():
        match input.model_tab():
            case "explanation":
                return ui.panel_well(
                    ui.h4("Model Explanation"),
                    ui.p("This section explains the architecture of the double-layer CNN.")
                )
            case "training":
                return ui.panel_well(
                    ui.h4("Model Training"),
                    ui.p("We trained our model using augmented and oversampled data for better generalization.")
                )
            case "unseen":
                return ui.panel_well(
                    ui.h4("Unseen Data"),
                    ui.p("This section tests the model's performance on unseen data.")
                )

app = App(app_ui, server)