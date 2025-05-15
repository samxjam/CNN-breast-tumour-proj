from shiny import App, ui, render

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