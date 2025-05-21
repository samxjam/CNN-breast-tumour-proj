from shiny import App, ui, render
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.metrics import confusion_matrix
from shiny import reactive
import tensorflow as tf
import os


def masked_sparse_categorical_crossentropy(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    mask = tf.not_equal(y_true, -1)
    y_true_filtered = tf.boolean_mask(y_true, mask)
    y_pred_filtered = tf.boolean_mask(y_pred, mask)

    return tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(y_true_filtered, y_pred_filtered)
    )


base_path = os.path.dirname(__file__)
clahe_path = os.path.join(base_path, "models", "dlcnn_MS_CLAHE.keras")
nonclahe_path = os.path.join(base_path, "models", "multiclass_MS_NONCLAHE.keras")

dlmodel = tf.keras.models.load_model(
    clahe_path,
    custom_objects={"masked_sparse_categorical_crossentropy": masked_sparse_categorical_crossentropy}
)
mcmodel = tf.keras.models.load_model(nonclahe_path)

models = {
    "Double Layer": dlmodel,
    "Multiclass": mcmodel,
}

class_labels = {
    "Double Layer": ["Invasive", "DCIS 1", "DCIS 2", "Prolif Invasive"],
    "dlcnn_nonclahe": ["Invasive", "DCIS 1", "DCIS 2", "Prolif Invasive"], 
    "Multiclass": ["Immune", "Invasive", "DCIS 1", "DCIS 2", "Prolif Invasive"],
    "multiclass_nonclahe": ["Immune", "Invasive", "DCIS 1", "DCIS 2", "Prolif Invasive"],
    "binary": ["Immune", "Tumour"]
}

model_outputs = {
    "Double Layer": {
        "ytrue": "models/outputs/dlcnn_MS_CLAHE_ytrue.npy",
        "ypred": "models/outputs/dlcnn_MS_CLAHE_ypred.npy",
        "conf": "models/outputs/dlcnn_MS_CLAHE_conf.npy",
        "history": "models/history/dlcnn_MS_CLAHE_history.csv"
    },
    "Multiclass": {
        "ytrue": "models/outputs/multiclass_CLAHE_ytrue.npy",
        "ypred": "models/outputs/multiclass_CLAHE_ypred.npy",
        "conf": "models/outputs/multiclass_CLAHE_confidence_values.npy"
    }
}

y_true = np.load(model_outputs["Double Layer"]["ytrue"])
y_pred = np.load(model_outputs["Double Layer"]["ypred"])
history_df = pd.read_csv(model_outputs["Double Layer"]["history"])
confidences_dict = {
    key: np.load(paths["conf"]) for key, paths in model_outputs.items()
}

msacc = round((np.sum(y_true == y_pred) / len(y_pred)) * 100, 1)

def add_immune_padding(y_true, y_pred, immune_class_index=4, extra=0.9144 * 3000):
    pad = np.ones(round(extra)) * immune_class_index
    return np.concatenate((y_true, pad)), np.concatenate((y_pred, pad))

y_immunetrue, y_immunepred = add_immune_padding(y_true, y_pred)
msimmacc = round((np.sum(y_immunetrue == y_immunepred) / len(y_immunepred)) * 100, 1)

def encode_image(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

last_prediction = {}
prediction_log = []

def mc_predict_class(file_path, model_key):
    print("Selected model key:", model_key)
    if model_key not in models:
        print("Model key not found in models dictionary")
        return "Invalid model selected", 0.0

    model = models[model_key]
    labels = class_labels[model_key]

    img = tf.keras.preprocessing.image.load_img(file_path, target_size=(100, 100))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    multiclass_pred = model.predict(img_array)
    
    predicted_index = np.argmax(multiclass_pred[0])
    confidence = float(np.max(multiclass_pred[0])) 

    predicted_label = labels[predicted_index]
    return predicted_label, confidence

def dl_predict_class(file_path, model_key):
    print("Selected model key:", model_key)
    if model_key not in models:
        print("Model key not found in models dictionary")
        return "Invalid model selected", 0.0

    model = models[model_key]
    labels = class_labels[model_key]

    img = tf.keras.preprocessing.image.load_img(file_path, target_size=(100, 100))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    binary_pred, multiclass_pred = model.predict(img_array)

    if binary_pred[0][0] > 0.5:
        predicted_index = np.argmax(multiclass_pred[0])
        predicted_label = labels[predicted_index]
        confidence = float(np.max(multiclass_pred[0]))
    else:
        predicted_label = "Immune"
        confidence = 1 - float(binary_pred[0][0])
        
    return predicted_label, confidence

def app_ui(request):
    return ui.page_fluid(
        ui.tags.head(
            ui.tags.link(
                rel="stylesheet",
                href="https://cdn.jsdelivr.net/npm/bootswatch@5.3.1/dist/cerulean/bootstrap.min.css",
                id="theme-css"
            ),
            ui.tags.style(""" 
                html, body {
                    margin: 0 !important;
                    padding: 0 !important;
                    overflow-x: hidden;
                    font-family: 'Segoe UI', sans-serif;
                }
                .navbar {
                    margin-bottom: 0 !important;
                    min-height: 60px;
                }
                body > .container-fluid {
                    padding-left: 15px !important;
                    padding-right: 15px !important;
                }
                .navbar-nav .nav-link, .navbar-brand {
                    color: #ffffff !important;
                }
            """),
            ui.tags.script("""
                Shiny.addCustomMessageHandler("set-theme", function(data) {
                    var css = document.getElementById("theme-css");
                    if (css) {
                        css.href = data.theme;
                    }
                });
            """)
        ),

        ui.page_navbar(
            ui.nav_panel("Test Image", test_image_layout()),
            title=ui.output_ui("app_title"),
        ),
    )

def test_image_layout():
    return ui.layout_sidebar(
                ui.sidebar(
                    "Upload an H&E image for model prediction and visualization.",
                ui.div(
                    ui.input_switch("dark_mode", "Enhance Cell Visibility", value=False),
                    ui.tags.div(
                        "Use this mode to better distinguish H&E cell structures in low-light environments.",
                        class_="form-text text-muted"
                    ),
                    class_="mb-3"
                ),
                ),
                
        ui.layout_columns(
            ui.card(
                ui.h4("Upload & Session Management"),
                ui.input_file("image_upload", "Upload H&E Image", accept=[".jpg", ".png", ".tif"]),
                ui.input_file("log_upload", "Resume from previous session (.csv)", accept=[".csv"]),
                ui.download_button("download_log_backup", "Backup Session Log", class_="btn btn-primary"),
                style="margin-bottom: 20px;"
            ),

            ui.card(
                ui.h4("Image Preview"),
                ui.output_ui("image_preview"),
                ui.output_ui("blur_warning"),
                ui.hr(),
                ui.h4("Prediction Result"),
                ui.output_ui("prediction_result"),
                style="margin-bottom: 20px;"
            ),

            ui.card(
                ui.h4("User Feedback"),
                ui.input_radio_buttons(
                    "prediction_agreement", 
                    "Do you agree with the model’s prediction?",
                    choices=["Agree", "Disagree", "Unsure"],
                    selected="Agree"
                ),
                ui.input_text_area("user_comment", "Add Comment"),
                ui.output_ui("correct_class_ui"),
                ui.input_action_button("submit_comment", "Log Prediction", class_="btn btn-primary"),
                ui.hr(),
                ui.h4("Prediction Log"),
                ui.output_ui("prediction_table"),
                ui.input_text("csv_filename", "Name your download file", placeholder="e.g., my_session_log"),
                ui.download_button("download_log", "Download Log", class_="btn btn-primary"),
                style="margin-bottom: 20px;"
            )
        )
    )
from PIL import Image, ImageFilter

def is_blurry(file_path, threshold=1000):
    img = Image.open(file_path).convert("L").filter(ImageFilter.FIND_EDGES)
    variance = np.var(np.asarray(img))
    return variance > threshold

def server(input, output, session):

    @output
    @render.ui
    def clahe_conf_comparison():
        bins = input.bins()
        html_str = dual_confidence_plot(
            conf1=confidences_dict["multiclass_CLAHE"],
            conf2=confidences_dict["dlcnn_MS_CLAHE"],
            label1="CLAHE - 1 Layer",
            label2="CLAHE - 2 Layer",
            bins=bins
        )
        return ui.HTML(html_str)

    @output
    @render.ui
    def image_preview():
        uploaded = input.image_upload()
        if not uploaded:
            return ui.p("No image uploaded")
        file_path = uploaded[0]["datapath"]
        img_tag = f"""
        <div style='border: 1px solid #ccc; overflow: hidden; max-width: 100%;'>
            <img id='panzoom-img' src='data:image/png;base64,{encode_image(file_path)}'
                style='width: 100%; cursor: grab;' draggable="false"/>
        </div>
        """
        return ui.HTML(img_tag)
        
    @output
    @render.ui
    def app_title():
        color = "#FFFFFF" if input.dark_mode() else "#000000"
        return ui.tags.h3("A Pathologist’s Best Friend", style=f"color: {color}; margin: 0;")

    @output
    @render.ui
    def prediction_result():
        uploaded = input.image_upload()
        if not uploaded:
            return ""
        file_path = uploaded[0]["datapath"]

        # Run both models
        results = []
        for model_key in ["Double Layer", "Multiclass"]:
            if model_key == "Double Layer":
                dl_pred, dl_conf = dl_predict_class(file_path, model_key)
                results.append((model_key, dl_pred, dl_conf))
            if model_key == "Multiclass":
                mc_pred, mc_conf = mc_predict_class(file_path, model_key)
                results.append((model_key, mc_pred, mc_conf))

        # Log result only once with default model
        default_model_key, default_pred, default_conf = results[0]
        last_prediction.clear()
        last_prediction.update({
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": uploaded[0]["name"],
            "dl_prediction": dl_pred,
            "dl_confidence": dl_conf,
            "mc_prediction": mc_pred,
            "mc_confidence": mc_conf,
            "comment": "",
            "agreement": "",
            "suggested": ""
        })

        # Display results side-by-side
        return ui.div(
            ui.layout_columns(
                *[
                    ui.card(
                        ui.h5(f"{key}"),
                        ui.p(f"Prediction: {pred}"),
                        ui.p(f"Confidence: {conf:.2%}")
                    )
                    for key, pred, conf in results
                ]
            ),
            ui.hr(),
            ui.tags.span("Prediction complete", class_="badge bg-success")
        )

    @reactive.effect
    @reactive.event(input.submit_comment)
    def on_submit_comment():
        if input.submit_comment():
            comment = input.user_comment()
            agreement = input.prediction_agreement()
            suggested = input.correct_class() if agreement == "Disagree" else ""
    
            if last_prediction and isinstance(last_prediction, dict):
                entry = last_prediction.copy()
                entry.update({
                    "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "comment": comment,
                    "agreement": agreement,
                    "suggested": suggested,
                })
                prediction_log.append(entry)
                
                session.send_input_message("user_comment", {"value": ""})
                session.send_input_message("region_of_interest", {"value": ""})
                session.send_input_message("prediction_agreement", {"value": "Agree"})
    
                ui.notification_show("Comment saved with prediction", type="message", duration=3000)
            else:
                ui.notification_show("No prediction available to annotate", type="error", duration=3000)

    @output
    @render.ui
    @reactive.event(input.submit_comment)
    def prediction_table():
        if not prediction_log:
            return ui.p("No prediction log available. Submit a comment to record this prediction.")
        else:
            df = pd.DataFrame(prediction_log)
            df.columns = ["timestamp", "filename", "dl_prediction", "dl_confidence", "mc_prediction", "mc_confidence","comment", "agreement", "suggested"]
            html_table = df.to_html(classes="table table-striped", index=False, border=0)
            return ui.HTML(f"<div class='table-responsive'>{html_table}</div>")

    def write_log_csv(out_file):
        df = pd.DataFrame(prediction_log)
        if df.empty:
            out_file.write("No prediction data available.")
        else:
            df = df[
                ["timestamp", "filename", "dl_prediction", "dl_confidence",
                "mc_prediction", "mc_confidence", "comment", "agreement", "suggested"]
            ]
            df.to_csv(out_file, index=False)

    
    @reactive.Effect
    @reactive.event(input.dark_mode)
    async def toggle_theme():
        theme_url = (
            "https://cdn.jsdelivr.net/npm/bootswatch@5.3.1/dist/darkly/bootstrap.min.css"
            if input.dark_mode()
            else "https://cdn.jsdelivr.net/npm/bootswatch@5.3.1/dist/cerulean/bootstrap.min.css"
        )
        await session.send_custom_message("set-theme", {"theme": theme_url})




    @output
    @render.download(
        filename=lambda: (input.csv_filename().strip() or "prediction_log") + ".csv"
    )
    def download_log():
        def writer():
            df = pd.DataFrame(prediction_log)
            if df.empty:
                yield b"No prediction data available.\n"
            else:
                df_filtered = df[
                    ["timestamp", "filename", "dl_prediction", "dl_confidence",
                    "mc_prediction", "mc_confidence", "comment", "agreement", "suggested"]
                ]
                yield df_filtered.to_csv(index=False).encode("utf-8")
        return writer()
        
    @output
    @render.ui
    def correct_class_ui():
        if input.prediction_agreement() == "Disagree":
            return ui.input_select(
                "correct_class",
                "Suggested Correct Class",
                choices=[
                    "Immune",
                    "Invasive",
                    "DCIS 1",
                    "DCIS 2",
                    "Prolif Invasive"
                ]
            )
        else:
            return ui.div()
    
    
    @output
    @render.ui
    def blur_warning():
        uploaded = input.image_upload()
        if not uploaded:
            return ""
        file_path = uploaded[0]["datapath"]
        if is_blurry(file_path):
            return ui.tags.span("Warning! Poor image quality detected: This image may be difficult to classify", class_="badge bg-danger")
        else:
            return ""

    @output
    @render.download(filename="prediction_log_backup.csv")
    def download_log_backup():
        def writer(out_file):
            df = pd.DataFrame(prediction_log)
            if df.empty:
                out_file.write("No prediction data available.")
            else:
                df.to_csv(out_file, index=False)
        return writer

    @reactive.Effect
    def load_uploaded_log():
        uploaded = input.log_upload()
        if uploaded:
            try:
                df = pd.read_csv(uploaded[0]["datapath"])
                required_cols = {"timestamp", "filename", "dl_prediction", "dl_confidence", 
                 "mc_prediction", "mc_confidence", "comment", "agreement", "suggested"}
                if not required_cols.issubset(df.columns):
                    ui.notification_show("Invalid session log format", type="error")
                    return
                prediction_log.clear()
                prediction_log.extend(df.to_dict(orient="records"))
                ui.notification_show("Session log restored.", type="message", duration=3000)
            except Exception as e:
                ui.notification_show(f"Error loading session log: {e}", type="error")
            

app = App(app_ui, server)

