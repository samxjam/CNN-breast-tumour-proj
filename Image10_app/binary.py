from shiny import App, ui, render

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

def server(input, output, session):
        @output
        @render.ui
        def model_content():
            match input.model_tab():
                case "explanation":
                    return ui.panel_well(
                        ui.h4("Model Explanation"),
                        ui.p("This section explains the architecture of the Binary CNN.")
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