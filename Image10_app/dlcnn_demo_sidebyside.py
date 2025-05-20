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
nonclahe_path = os.path.join(base_path, "models", "dlcnn_MS_NONCLAHE.keras")

clahemodel = tf.keras.models.load_model(
    clahe_path,
    custom_objects={"masked_sparse_categorical_crossentropy": masked_sparse_categorical_crossentropy}
)
nonclahemodel = tf.keras.models.load_model(nonclahe_path)

models = {
    "dlcnn_clahe": clahemodel,
    "dlcnn_nonclahe": nonclahemodel,
}

class_labels = {
    "dlcnn_clahe": ["Invasive", "DCIS 1", "DCIS 2", "Prolif Invasive"],
    "dlcnn_nonclahe": ["Invasive", "DCIS 1", "DCIS 2", "Prolif Invasive"], 
    "multiclass_clahe": ["Immune", "Invasive", "DCIS 1", "DCIS 2", "Prolif Invasive"],
    "multiclass_nonclahe": ["Immune", "Invasive", "DCIS 1", "DCIS 2", "Prolif Invasive"],
    "binary": ["Immune", "Tumour"]
}

model_outputs = {
    "dlcnn_MS_CLAHE": {
        "ytrue": "models/outputs/dlcnn_MS_CLAHE_ytrue.npy",
        "ypred": "models/outputs/dlcnn_MS_CLAHE_ypred.npy",
        "conf": "models/outputs/dlcnn_MS_CLAHE_conf.npy",
        "history": "models/history/dlcnn_MS_CLAHE_history.csv"
    },
    "multiclass_CLAHE": {
        "ytrue": "models/outputs/multiclass_CLAHE_ytrue.npy",
        "ypred": "models/outputs/multiclass_CLAHE_ypred.npy",
        "conf": "models/outputs/multiclass_CLAHE_confidence_values.npy"
    }
}

y_true = np.load(model_outputs["dlcnn_MS_CLAHE"]["ytrue"])
y_pred = np.load(model_outputs["dlcnn_MS_CLAHE"]["ypred"])
history_df = pd.read_csv(model_outputs["dlcnn_MS_CLAHE"]["history"])
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

def predict_class(file_path, model_key):
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
        confidence = float(np.max(multiclass_pred[0]))
    else:
        predicted_index = len(labels) - 1  
        confidence = 1 - float(binary_pred[0][0])  

    predicted_label = labels[predicted_index]
    return predicted_label, confidence

def gradcam_image(file_path):
    return encode_image(file_path)

def app_ui(request):
    return ui.page_fluid(
        ui.tags.head(
            ui.include_css("theme/cerulean.min.css"),
            ui.tags.script(src="https://cdn.jsdelivr.net/npm/@panzoom/panzoom@9.4.0/dist/panzoom.min.js"),
            ui.tags.script("""
                document.addEventListener('shiny:connected', function () {
                    const img = document.getElementById('panzoom-img');
                    if (img) {
                        const panzoom = Panzoom(img, {
                            maxScale: 5,
                            contain: 'outside',
                        });
                        img.parentElement.addEventListener('wheel', function(e) {
                            e.preventDefault();
                            panzoom.zoomWithWheel(e);
                        });
                    }
                });
            """)
        ),

        ui.page_navbar(
            ui.nav_panel("Overview", overview_layout()),
            ui.nav_panel("Model", model_layout()),
            ui.nav_panel("Test Image", test_image_layout()),
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
    labels = ["Invasive", "DCIS 1", "DCIS 2", "Prolif Invasive"]

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != -1
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    
    plt.xticks(rotation=45, fontsize=8)
    plt.yticks(rotation=45, fontsize=8)
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close(fig)
    return f"<img src='data:image/png;base64,{img_base64}' class='img-fluid'/>"

def history_plot(history_df, metric="loss"):
    fig, ax = plt.subplots()
    ax.plot(history_df["binary_output_" + metric], 
            label=f"Train Binary {metric.capitalize()}", marker='o', lw=1.5, ms=4)
    ax.plot(history_df["val_binary_output_" + metric], 
            label=f"Val Binary {metric.capitalize()}", marker='o', lw=1.5, ms=4)
    ax.plot(history_df["multiclass_output_" + metric], 
            label=f"Train Multiclass {metric.capitalize()}", marker='o', lw=1.5, ms=4)
    ax.plot(history_df["val_multiclass_output_" + metric], 
            label=f"Val Multiclass {metric.capitalize()}", marker='o', lw=1.5, ms=4)
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

def confidence_plot(confidences, bins=20):
    fig, ax = plt.subplots()
    
    ax.hist(confidences, bins=bins)
    ax.set_title("Distribution of Prediction Confidences", fontsize=14)
    ax.set_xlabel("Confidence Score", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_xlim([0, 1.0])
    ax.grid(axis='y', alpha=0.5)
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close(fig)
    return f"<img src='data:image/png;base64,{img_base64}' class='img-fluid'/>"

def dual_confidence_plot(conf1, conf2, label1="Model A", label2="Model B", bins=20):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    axes[0].hist(conf1, bins=bins, color='skyblue', edgecolor='black')
    axes[0].set_title(label1)
    axes[0].set_xlabel("Confidence")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(conf2, bins=bins, color='lightgreen', edgecolor='black')
    axes[1].set_title(label2)
    axes[1].set_xlabel("Confidence")

    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close(fig)

    return f"<img src='data:image/png;base64,{img_base64}' class='img-fluid'/>"

def test_image_layout():
    return ui.layout_sidebar(
        ui.sidebar("Upload an H&E image for model prediction and visualization."),

        ui.layout_columns(

            ui.card(
                ui.h4("Choose Classification Model"),
                ui.hr(),
                ui.h4("Upload & Session Management"),
                ui.input_file("image_upload", "Upload H&E Image", accept=[".jpg", ".png", ".tif"]),
                ui.input_file("log_upload", "Resume from previous session (.csv)", accept=[".csv"]),
                ui.download_button("download_log_backup", "Backup Session Log"),
                style="margin-bottom: 20px;"
            ),

            ui.card(
                ui.h4("Image Preview"),
                ui.output_ui("image_preview"),
                ui.hr(),
                ui.h4("Prediction Result"),
                ui.output_ui("prediction_result"),
                ui.output_text("blur_warning"),
                style="margin-bottom: 20px;"
            ),

            ui.card(
                ui.h4("User Feedback"),
                ui.input_radio_buttons(
                    "prediction_agreement", 
                    "Do you agree with the model’s prediction?",
                    choices=["Agree", "Disagree", "Neutral"],
                    selected="Agree"
                ),
                ui.input_text_area("user_comment", "Add Comment"),
                ui.input_action_button("submit_comment", "Submit Comment:"),
                ui.output_ui("correct_class_ui"),
                ui.input_text("region_of_interest", "Region of Interest (e.g., top-right, center)"),
                ui.hr(),
                ui.h4("Prediction Log"),
                ui.output_ui("prediction_table"),
                ui.download_button("download_log", "Download Prediction Log"),
                style="margin-bottom: 20px;"
            )
        )
    )
from PIL import Image, ImageFilter

def is_blurry(file_path, threshold=100):
    img = Image.open(file_path).convert("L").filter(ImageFilter.FIND_EDGES)
    variance = np.var(np.asarray(img))
    return variance < threshold

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
    def model_content():
        match input.model_tab():
            case "explanation":
                return ui.panel_well(
                    ui.h4("Model Explanation"),
                    ui.p("This section explains the architecture of the double-layer CNN.")
                )
            case "training":
                mclabels = ["Invasive", "DCIS 1", "DCIS 2", "Prolif Invasive"]
                mcimmlabels = ["Invasive", "DCIS 1", "DCIS 2", "Prolif Invasive", "Immune"]
                cm_img = cm_image(y_true, y_pred, mclabels)
                cm_imgimm = cm_image(y_immunetrue, y_immunepred, mcimmlabels)
                loss_plot = history_plot(history_df, "loss")
                accuracy_plot = history_plot(history_df, "accuracy")
                
                return ui.panel_well(
                    ui.h4("Model Training"),
                    ui.p("We trained our model using augmented and oversampled data for better generalization."),
                    ui.p(f"CLAHE model training accuracy: {msacc}%, With Immune: {msimmacc}%"),
                    ui.p(f"Training data: {len(y_true)} samples"),
                    ui.layout_columns(
                        ui.card(
                            ui.h5("Confusion Matrix"),
                            ui.navset_tab(
                                ui.nav_panel("Tumour Multiclass", ui.panel_well(
                                    ui.p(f"Accuracy: {msacc}%"),
                                    ui.HTML(cm_img))),
                                ui.nav_panel("With Immune", ui.panel_well(
                                    ui.p(f"Accuracy: {msimmacc}%"),
                                    ui.HTML(cm_imgimm))),
                                id="cm_tabs"
                            )
                        ),
                        ui.card(
                            ui.h5("Training Performance"),
                            ui.navset_tab(
                                ui.nav_panel("Accuracy", ui.HTML(accuracy_plot)),
                                ui.nav_panel("Loss", ui.HTML(loss_plot)),
                                ui.nav_panel("Confidence", ui.panel_well(
                                    ui.input_slider("bins", "Number of bins", 
                                                    min=1, max=100, value=20),
                                    ui.output_ui("clahe_conf_comparison"))),
                                id="metric_tabs"
                            )
                        )
                    )
                )
            

    @output
    @render.ui
    def image_preview():
        uploaded = input.image_upload()
        if not uploaded:
            return ui.p("No image uploaded.")
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
    def prediction_result():
        uploaded = input.image_upload()
        if not uploaded:
            return ""
        file_path = uploaded[0]["datapath"]

        # Run both models
        results = []
        for model_key in ["dlcnn_clahe", "dlcnn_nonclahe"]:
            pred, conf = predict_class(file_path, model_key)
            results.append((model_key, pred, conf))

        # Log result only once with default model
        default_model_key, default_pred, default_conf = results[0]
        last_prediction.clear()
        last_prediction.update({
            "filename": file_path.split("/")[-1],
            "model": "BOTH",
            "prediction": default_pred,
            "confidence": default_conf,
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "comment": ""
        })
        prediction_log.append(last_prediction.copy())

        # Display results side-by-side
        return ui.div(
            ui.layout_columns(
                *[
                    ui.card(
                        ui.h5(f"Model: {key.upper()}"),
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
    def on_submit_comment():
        if input.submit_comment():
            comment = input.user_comment()
            agreement = input.prediction_agreement()
            roi = input.region_of_interest().strip()
            suggested = input.correct_class() if agreement == "Disagree" else ""

            if last_prediction:
                last_prediction["comment"] = comment
                last_prediction["agree"] = agreement
                last_prediction["roi"] = roi
                last_prediction["suggested"] = suggested

                prediction_log.append(last_prediction.copy())

                session.send_input_message("user_comment", {"value": ""})
                session.send_input_message("region_of_interest", {"value": ""})
                session.send_input_message("prediction_agreement", {"value": "Agree"})

                ui.notification_show("Comment saved with prediction", type="message", duration=3000)

    @output
    @render.ui
    def prediction_table():
        if not prediction_log:
            return ui.p("No prediction log available. Submit a comment to record this prediction.")
        df = pd.DataFrame(prediction_log)
        df.columns = ["Timestamp", "Filename", "Model", "Prediction", "Confidence", "Agree", "Comment"]
        html_table = df.to_html(classes="table table-striped", index=False, border=0)
        return ui.HTML(f"<div class='table-responsive'>{html_table}</div>")


    @output
    @render.ui
    def gradcam_plot():
        uploaded = input.image_upload()
        if not uploaded:
            return ""
        file_path = uploaded[0]["datapath"]
        gradcam = gradcam_image(file_path)
        return ui.HTML(f"<img src='data:image/png;base64,{gradcam}' class='img-fluid'/>")
    
    @output
    @render.download(filename="prediction_log.csv")
    def download_log():
        df = pd.DataFrame(prediction_log)
        return df.to_csv(index=False)

    @output
    @render.ui
    def correct_class_ui():
        if input.prediction_agreement() == "Disagree":
            return ui.input_select(
                "correct_class",
                "Suggested Correct Class",
                choices=[
                    "Invasive",
                    "DCIS 1",
                    "DCIS 2",
                    "Prolif Invasive"
                ]
            )
        else:
            return ui.div()
    
    
    @output
    @render.text
    def blur_warning():
        uploaded = input.image_upload()
        if not uploaded:
            return ""
        file_path = uploaded[0]["datapath"]
        return "Image may be blurry" if is_blurry(file_path) else ""

    @output
    @render.download(filename="prediction_log_backup.csv")
    def download_log_backup():
        df = pd.DataFrame(prediction_log)
        return df.to_csv(index=False)

    @reactive.Effect
    def load_uploaded_log():
        uploaded = input.log_upload()
        if uploaded:
            try:
                df = pd.read_csv(uploaded[0]["datapath"])
                required_cols = {"Timestamp", "Filename", "Model", "Prediction", "Confidence", "Agree", "Comment"}
                if not required_cols.issubset(df.columns):
                    ui.notification_show("Invalid session log format", type="error")
                    return
                prediction_log.clear()
                prediction_log.extend(df.to_dict(orient="records"))
                ui.notification_show("Session log restored.", type="message", duration=3000)
            except Exception as e:
                ui.notification_show(f"Error loading session log: {e}", type="error")
            

app = App(app_ui, server)
