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
                        ui.p("The binary CNN was the first and most basic model we built. Its purpose was to distinguish between cancer cells and immune cells - if it was unable to do this reliably, it would have made further work impossible.")
                        ui.p("The description of the training process and the final evalutation of this model's effectiveness can be found in the other sections of this app.")
                    )
                case "training":
                    return ui.panel_well(
                        ui.h4("Model Training"),
                        ui.p("We trained our model using augmented and oversampled data for better generalization.")
                        ui.p("The immune class consists of images from four immune cell types - B Cells, Stromal cells, and T cells (TD4+ and TD8+). The tumour class consists of images of cells from different stages of cancer - DCIS-1, DCIS-2, Invasive, and Prolif-Invasive.")
                    )
                case "unseen":
                    return ui.panel_well(
                        ui.h4("Unseen Data"),
                        ui.p("This section tests the model's performance on unseen data.")
                        ui.p("When asking the model to distinguish between these immune cells and the main tumour cells, the model turned out to be highly reliable. However, it does not cover all the immune cell types, so it may not be as accurate when dealing with cells not used for its training.")
                    )

app = App(app_ui, server)
