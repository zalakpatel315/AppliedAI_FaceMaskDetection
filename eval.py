from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score,classification_report
import seaborn as s
from dataProcess import load_data, im_convert, validation_loader
import torch
import numpy as np
import matplotlib.pyplot as plt
from train_model import CNN


def generate_sample_preds(test_load, model, path):
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    output = model(images)
    _, preds = torch.max(output, 1)

    fig = plt.figure(figsize=(25, 4))
    classes = ['Cloth Mask', 'FFP2 Mask', 'FFP2 Mask With Valve', 'Surgical Mask', 'Without Mask']
    for idx in np.arange(20):
        ax = fig.add_subplot(2, 10, idx + 1, xticks=[], yticks=[])
        ax.set_title(classes[preds[idx]])
        plt.imsave( path + classes[preds[idx]] + str(idx) + '.png',
                   im_convert(images[idx]))
    Accuracy = accuracy_score(preds, labels)
    print("Accuracy of Current Batch: ", Accuracy)
    confusion = confusion_matrix(preds, labels)
    s.set(font_scale=1.5)
    s.heatmap(
            confusion_matrix(labels, preds),
            annot=True,
            annot_kws={"size": 16},
            cmap="Blues"
    )
    plt.imsave(path+'confusion_matrix.png', confusion)
    report = classification_report(
            labels,
            preds,
            target_names=classes
    )
    print(report)

if __name__== "__main__":
    test_loader = load_data("/Users/jhanviarora/Desktop/Project/output/test/")
    test_loader = validation_loader(test_loader, 214)
    model = torch.load('/Users/jhanviarora/Desktop/Project/output_models/ep15bs100.h5')
    generate_sample_preds(test_loader, model, '/Users/jhanviarora/Desktop/Project/Output_Predictions/')