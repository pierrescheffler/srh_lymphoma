# Libraries
import numpy as np
import torch
import pandas
from pathlib import Path
import re
import itertools
import pydicom
from ctran import ctranspath
from sklearn import metrics
import matplotlib.pyplot as plt
import utils


def sort_study_images(study_folder=Path("~")):
    # Takes a folder containing a study as input and returns a list of
    # dictionaries, 1 dictionary per image series. Each dictionary contains
    # lists of file paths for that correspond to the dictionary key.

    # Compile regex
    id_pattern = re.compile("\d+_\d+")

    # Get file paths
    file_paths = list(study_folder.glob("**/img*_*_*.dcm"))

    # Remove any duplicates with identical file basename
    for file_path in file_paths:
        if [i.name for i in file_paths].count(file_path.name) > 1:
            file_paths.remove(file_path)

    # Extract indices, rows and columns
    suid, parent, series, image, rows, columns, patname, seriesno = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for file_path in file_paths:
        parent_folder = file_path.parent.name
        series_id, image_id = id_pattern.search(file_path.name)[0].split(sep="_")
        dicom = pydicom.dcmread(file_path, stop_before_pixels=True)
        try:
            rows.append(dicom.Rows)
            columns.append(dicom.Columns)
            patname.append(dicom.PatientName)
            seriesno.append(dicom.SeriesNumber)
        except:
            continue
        parent.append(parent_folder)
        series.append(int(series_id))
        suid.append(str(parent_folder) + "-" + str(series_id))
        image.append(int(image_id))
    file_metadata = sorted(
        list(
            zip(
                suid,
                image,
                rows,
                columns,
                file_paths,
                patname,
                seriesno,
                parent,
                series,
            )
        )
    )
    suid_keys, image_metadata = [], []
    for suid_key, group in itertools.groupby(file_metadata, lambda x: x[0]):
        suid_keys.append(suid_key)
        image_metadata.append(list(group))
    grouped_metadata = list(zip(suid_keys, image_metadata))

    # Loop over sample indices
    dict_list = []
    for suid_key, image_metadata in grouped_metadata:
        # Get series description
        try:
            series_description = pydicom.dcmread(
                image_metadata[0][4], stop_before_pixels=True
            ).SeriesDescription
        except:
            series_description = ""
        # As strips are 1000 pixels wide, stitched images will be wider
        stitched_images = [i[4] for i in image_metadata if i[3] > 1000]
        # Strips are 1000 pixels wide and higher than they are wide
        strips = [i[4] for i in image_metadata if i[3] == 1000 and i[2] > i[3]]
        strips_with_metadata = [
            i for i in image_metadata if i[3] == 1000 and i[2] > i[3]
        ]
        # Auto focus images are 1000 pixels wide and wider than they are high
        auto_focus = [i[4] for i in image_metadata if i[3] == 1000 and i[2] < i[3]]
        # Sample images are less wide than strips
        sample_images = [i[4] for i in image_metadata if i[3] < 1000]
        # If there are 2 channels, they are CH2 and CH3, if there are 5,
        # 3 ALA channels are added.
        CH2 = []
        CH3 = []
        ALA1 = []
        ALA2 = []
        ALA3 = []
        if strips:
            n_channels = len(strips) // (int(strips_with_metadata[0][2]) // 1000)
            if n_channels == 2:
                CH2 = [i for ind, i in enumerate(strips) if ind % 2 == 0]
                CH3 = [i for ind, i in enumerate(strips) if ind % 2 == 1]
            elif n_channels == 5:
                CH2 = [i for ind, i in enumerate(strips) if ind % 5 == 0]
                CH3 = [i for ind, i in enumerate(strips) if ind % 5 == 1]
                ALA1 = [i for ind, i in enumerate(strips) if ind % 5 == 2]
                ALA2 = [i for ind, i in enumerate(strips) if ind % 5 == 3]
                ALA3 = [i for ind, i in enumerate(strips) if ind % 5 == 4]
        analysis = []
        HE = []
        ALA = []
        for i in stitched_images:
            # Analysis images have ImageComments or UnformattedTextValue tags
            try:
                analysis_text = pydicom.dcmread(
                    i, stop_before_pixels=True
                ).ImageComments
                analysis.append(analysis_text)
            except:
                try:
                    analysis_text = pydicom.dcmread(
                        i, stop_before_pixels=True
                    ).UnformattedTextValue
                    analysis.append(analysis_text)
                except:
                    # ALA images have an empty blue channel
                    try:
                        if np.array(pydicom.dcmread(i).pixel_array)[:, :, 2].any():
                            HE.append(i)
                        else:
                            ALA.append(i)
                    except:
                        pass
        # Collect all information in a dictionary
        dict = {
            "study": study_folder.name,
            "series": image_metadata[0][8],
            "parent": image_metadata[0][7],
            "suid": suid_key,
            "patname": image_metadata[0][5],
            "seriesno": image_metadata[0][6],
            "series_description": series_description,
            "height": image_metadata[0][2],
            "width": image_metadata[0][3],
            "stitched_images": stitched_images,
            "HE": HE,
            "analysis": analysis,
            "ALA": ALA,
            "auto_focus": auto_focus,
            "strips": strips,
            "CH2": CH2,
            "CH3": CH3,
            "ALA1": ALA1,
            "ALA2": ALA2,
            "ALA3": ALA3,
            "sample_images": sample_images,
        }
        dict_list.append(dict)

    return dict_list


# Get IDs of lymphomas and glioblastomas
lymphoma_sheet = pandas.read_excel("NIO_Lymphome.xlsx")
glioblastoma_sheet = pandas.read_excel("NIO_GBMs_STX.xlsx")
lymphoma_list = lymphoma_sheet.iloc[1:, 1].values.astype(str).tolist()
glioblastoma_list = glioblastoma_sheet.iloc[1:, 1].values.astype(str).tolist()

# Add label 0 for lymphoma and 1 for glioblastoma
lymphoma_list = [[x, 0] for x in lymphoma_list]
glioblastoma_list = [[x, 1] for x in glioblastoma_list]

# Combine the lists
ID_list = lymphoma_list + glioblastoma_list

# Create a 60:20:20 split for the lymphoma and glioblastoma IDs
from sklearn.model_selection import train_test_split

train_ID, test_ID = train_test_split(ID_list, test_size=0.2, random_state=0)
train_ID, val_ID = train_test_split(train_ID, test_size=0.25, random_state=0)

# Get paths for IDs
data_path = r"lymphoma_gbm_images/"


# Get all files whose names start with the ID and pair them with the label
def get_images(ID_list):
    import glob

    file_label_list = []
    for ID in ID_list:
        file_list = []
        file_list = file_list + sum(
            [
                x["HE"]
                for x in sort_study_images(Path(data_path + "pat_" + ID[0].zfill(4)))
            ],
            [],
        )
        file_list = file_list + sum(
            [
                x["HE"]
                for x in sort_study_images(Path(data_path + "pat " + ID[0].zfill(2)))
            ],
            [],
        )
        file_label_list = file_label_list + [[x, ID[1]] for x in file_list]
    return file_label_list


# Get files for train, val and test sets

train_files = get_images(train_ID)
val_files = get_images(val_ID)
test_files = get_images(test_ID)


# Functions
def generate_images(data, nr_crops=1, crop_size=(224, 224, 3)):

    import numpy as np
    import pydicom
    import random
    from scipy.ndimage import rotate

    file_path = data[0]
    label = data[1]
    crop_tensor = np.empty(shape=(nr_crops,) + crop_size, dtype="uint8")
    crop_counter = 0
    # Discard top 300 pixels of the image due to watermark
    picture = pydicom.dcmread(file_path).pixel_array[300:, :]
    while crop_counter < nr_crops:
        seed_x = random.randint(0, picture.shape[0] - crop_size[0])
        seed_y = random.randint(0, picture.shape[1] - crop_size[1])
        crop = picture[
            seed_x : (seed_x + crop_size[0]), seed_y : (seed_y + crop_size[1])
        ]
        # Only add the crop if it is not white
        if np.mean(crop) < 200:
            crop_tensor[crop_counter] = crop
            crop_counter += 1
    return np.stack(crop_tensor), np.repeat(label, nr_crops)


# Variables
nr_crops = 250

# Create patches
train_patches = np.empty(
    shape=(nr_crops * len(train_files),) + (224, 224, 3), dtype="uint8"
)
train_labels = np.empty(shape=(nr_crops * len(train_files),), dtype="uint8")
for i in range(len(train_files)):
    (
        train_patches[i * nr_crops : (i + 1) * nr_crops],
        train_labels[i * nr_crops : (i + 1) * nr_crops],
    ) = generate_images(train_files[i], nr_crops=nr_crops, crop_size=(224, 224, 3))
    print(i)
train_patches = torch.tensor(train_patches / 255, dtype=torch.float).permute(0, 3, 1, 2)
val_patches = np.empty(
    shape=(nr_crops * len(val_files),) + (224, 224, 3), dtype="uint8"
)
val_labels = np.empty(shape=(nr_crops * len(val_files),), dtype="uint8")
for i in range(len(val_files)):
    (
        val_patches[i * nr_crops : (i + 1) * nr_crops],
        val_labels[i * nr_crops : (i + 1) * nr_crops],
    ) = generate_images(val_files[i], nr_crops=nr_crops, crop_size=(224, 224, 3))
    print(i)
val_patches = torch.tensor(val_patches / 255, dtype=torch.float).permute(0, 3, 1, 2)
test_patches = np.empty(
    shape=(nr_crops * len(test_files),) + (224, 224, 3), dtype="uint8"
)
test_labels = np.empty(shape=(nr_crops * len(test_files),), dtype="uint8")
for i in range(len(test_files)):
    (
        test_patches[i * nr_crops : (i + 1) * nr_crops],
        test_labels[i * nr_crops : (i + 1) * nr_crops],
    ) = generate_images(test_files[i], nr_crops=nr_crops, crop_size=(224, 224, 3))
    print(i)
test_patches = torch.tensor(test_patches / 255, dtype=torch.float).permute(0, 3, 1, 2)


# Create pytorch dataloader
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


# Create datasets
train_dataset = CustomDataset(data=train_patches, labels=train_labels)
val_dataset = CustomDataset(data=val_patches, labels=val_labels)
test_dataset = CustomDataset(data=test_patches, labels=test_labels)

# Load CTransPath model
model = ctranspath()

PATH = r"best_model.pth"

# Define loss function and optimizer

import torch.optim as optim
import torch.nn as nn


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

model.cuda().train()

for epoch in range(10):  # loop over the dataset multiple times

    train_loss = 0.0
    val_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
            running_loss = 0.0

    for i, data in enumerate(val_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels.long())

        # print statistics
        val_loss += loss.item()
    print(
        f"Epoch {epoch + 1} | Train Loss: {train_loss / len(train_loader):.3f} | Val Loss: {val_loss / len(val_loader):.3f}"
    )

    # Save model if validation loss is lower than previous lowest
    if epoch == 0:
        lowest_val_loss = val_loss
    elif val_loss < lowest_val_loss:
        lowest_val_loss = val_loss
        torch.save(model.state_dict(), PATH)

print("Finished Training")


# Load best model
model.load_state_dict(torch.load(PATH))


# Get predictions
model.eval().cuda()

test_output_list = []
label_output_list = []

with torch.no_grad():
    for i, data in enumerate(test_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.cuda()
        label_output_list.append(labels)
        test_output_list.append(model(inputs.cuda()).detach().cpu().numpy())

output_tensor = np.concatenate(test_output_list)

output_per_image = np.zeros(shape=((output_tensor.shape[0] // 250), 2))
for i in range(output_tensor.shape[0] // 250):
    output_per_image[i] = np.mean(output_tensor[i * 250 : (i + 1) * 250], axis=0)

y_score = torch.nn.functional.softmax(torch.tensor(output_per_image)).cpu().numpy()
y_pred = np.argmax(y_score, axis=1)
y_true = np.concatenate(label_output_list)[::250]
accuracy = metrics.accuracy_score(y_true, y_pred)
print(accuracy)

# Get graph showing accuracy per image
import matplotlib.pyplot as plt

zipped = list(zip(y_true, y_score))

# Sort first by true label, then by descending probability of true label
zipped.sort(key=lambda x: (x[0], -x[1][x[0]]))

# Separate the data into two lists
lymphoma = [x for x in zipped if x[0] == 0]
glioblastoma = [x for x in zipped if x[0] == 1]

# Create a panel with 2 subplots side by side
fig, axs = plt.subplots(1, 2, figsize=(14, 7))

# Plot two stacked bar charts, one for each class
for ax, data, title in zip(
    axs, [lymphoma, glioblastoma], ["Lymphomas", "Glioblastomas"]
):
    ax.bar(
        [x + 1 for x in range(len(data))],
        [x[1][0] for x in data],
        color="red",
        label="Predicted Lymphoma",
        width=0.8,
        tick_label=[f"{x + 1}" for x in range(len(data))],
    )
    ax.bar(
        [x + 1 for x in range(len(data))],
        [x[1][1] for x in data],
        bottom=[x[1][0] for x in data],
        color="blue",
        label="Predicted Glioblastoma",
        width=0.8,
        tick_label=[f"{x + 1}" for x in range(len(data))],
    )
    ax.set_title(title)
    ax.set_xlabel("Image")
    ax.set_ylabel("Probability")


# Plot a legend in the upper right corner outside the graph
plt.legend(loc="upper right", bbox_to_anchor=(0.2, -0.1))
plt.savefig("lymphoma_glioblastoma_probabilities.png", dpi=300)


# Create ROC curve in black and white for publication using torchmetrics
from torchmetrics import ROC

roc = ROC(task="binary")
roc.update(torch.tensor(y_score[:, 1]), torch.tensor(y_true))
roc.compute()

roc.plot()

from torchmetrics.classification import BinaryROC

metric = BinaryROC()
metric.update(torch.tensor(y_score[:, 1]), torch.tensor(y_true))

# Plot ROC curve. The title is changed to "ROC Curve" and the color is changed to black.

fig, ax = metric.plot(score=True)
ax.set_title("ROC curve")
ax.set_facecolor("white")

# Set the color of the ROC curve to black
ax.get_lines()[0].set_color("black")

# Set the color of the line in the legend to black
ax.get_legend().get_lines()[0].set_color("black")

plt.show()

# Get heatmaps

plt.imshow(
    utils.get_heatmaps(model, utils.load_image(str(train_files[110][0])), (224, 224))[1]
)

# Get confusion matrix from torchmetrics
from torchmetrics import ConfusionMatrix

confusion_matrix = ConfusionMatrix(task="binary", threshold=0.1)
confusion_matrix.update(torch.tensor(y_score[:, 1]), torch.tensor(y_true))
fig, ax = confusion_matrix.plot(labels=["Lymphoma", "Glioblastoma"])

# Set colormap to Blues
ax.set_facecolor("white")
ax.get_images()[0].set_cmap("Blues")
plt.savefig("confusion_matrix.png")

# Plot confusion matrix using sklearn with Blues colormap
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_true, y_score)
sns.heatmap(
    cm,
    annot=True,
    cmap="viridis",
    xticklabels=["Lymphoma", "Glioblastoma"],
    yticklabels=["Lymphoma", "Glioblastoma"],
)
plt.xlabel("Predicted", size=14)
plt.ylabel("True", size=14)
plt.savefig("confusion_matrix.png", dpi=300)

# Get plot of F1 score by threshold from 0 to 1 in steps of 0.1
from torchmetrics.classification import BinaryF1Score

values = []
for i in range(11):
    metric = BinaryF1Score(threshold=1 - (i / 10))
    values.append(metric(torch.tensor(y_score[:, 1]), torch.tensor(y_true)))
fig, ax = metric.plot(values)

# Label the x-axis "Threshold" and the y-axis "F1 Score"
ax.set_xlabel("Threshold")
ax.set_ylabel("F1 Score")

# Set the x axis labels to 0.0, 0.1, 0.2, ..., 1.0
ax.set_xticks(range(11))
ax.set_xticklabels([f"{i / 10:.1f}" for i in range(11)])
# Color the plot black
ax.set_facecolor("white")
ax.get_lines()[0].set_color("black")
